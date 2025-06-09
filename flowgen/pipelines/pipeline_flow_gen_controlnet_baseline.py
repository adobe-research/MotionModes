# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass
import gc
import sys 
import os 
import torch.utils.checkpoint as checkpoint
from torch.utils.checkpoint import checkpoint_sequential
import torch.nn.functional as nnf
from torch.optim.adam import Adam

import matplotlib.pyplot as plt

from torch.cuda.amp import autocast
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch import randn_like

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from ..models.unet import UNet3DConditionModel
from ..motion_control.motion_losses import combined_optical_flow_loss, save_flow_grid_pillow_arrow
from datetime import datetime

#from ...animation.utils.util import save_flow_grid_pillow_arrow

# motion_path = 'generated_motions'
# if not os.path.exists(motion_path):
#     os.makedirs(motion_path)
#     print(f"Created directory: {motion_path}")
# else:
#     print(f"Directory already exists: {motion_path}")

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def print_tensors_on_gpu():
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            print(f"Tensor: {obj.shape}, Type: {obj.dtype}, Memory: {obj.element_size() * obj.nelement() / 1024**2:.2f} MB")

def print_unet_memory_usage(model):
    total_memory = 0
    total_grad_memory = 0

    for name, param in model.named_parameters():
        param_memory = param.nelement() * param.element_size()  # Memory used by the parameter
        total_memory += param_memory
        
        if param.grad is not None:
            grad_memory = param.grad.nelement() * param.grad.element_size()  # Memory used by the gradient
            total_grad_memory += grad_memory

    total_memory_mb = total_memory / 1024**2  # Convert to MB
    total_grad_memory_mb = total_grad_memory / 1024**2  # Convert to MB

    print(f"Memory used by UNet parameters: {total_memory_mb:.2f} MB")
    print(f"Memory used by UNet gradients: {total_grad_memory_mb:.2f} MB")
    print(f"Total memory used by UNet (params + grads): {(total_memory_mb + total_grad_memory_mb):.2f} MB")


def print_tensor_memory_usage(local_vars):
    total_memory = 0
    total_grad_memory = 0
    
    print("\nMemory usage of tensors and their gradients:")
    for var_name, var in local_vars.items():
        if torch.is_tensor(var):
            tensor_memory = var.nelement() * var.element_size()
            total_memory += tensor_memory
            
            print(f"Tensor {var_name}: {var.shape}, dtype: {var.dtype}, requires_grad: {var.requires_grad}, memory: {tensor_memory / 1024**2:.2f} MB")
            
            if var.grad is not None:
                grad_memory = var.grad.nelement() * var.grad.element_size()
                total_grad_memory += grad_memory
                print(f"  -> Gradient: {var.grad.shape}, dtype: {var.grad.dtype}, memory: {grad_memory / 1024**2:.2f} MB")

    print(f"\nTotal memory used by tensors: {total_memory / 1024**2:.2f} MB")
    print(f"Total memory used by gradients: {total_grad_memory / 1024**2:.2f} MB")
    print(f"Combined memory (tensors + gradients): {(total_memory + total_grad_memory) / 1024**2:.2f} MB")

def print_gradients(model):
    print("Gradients in the model:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: {param.grad.shape}, dtype: {param.grad.dtype}")

def cads_linear_schedule(t, tau1 = 0.6, tau2 = 0.9):
        """ CADS annealing schedule function """
        if t <= tau1:
                return 1.0
        if t>= tau2:
                return 0.0
        gamma = (tau2-t)/(tau2-tau1)
        return gamma

def add_noise(y, gamma, noise_scale = 0.25, psi = 1.0, rescale=False):
        """ CADS adding noise to the condition

        Arguments:
        y: Input conditioning
        gamma: Noise level w.r.t t
        noise_scale (float): Noise scale
        psi (float): Rescaling factor
        rescale (bool): Rescale the condition
        """
        y_mean, y_std = torch.mean(y), torch.std(y)
        y = np.sqrt(gamma) * y + noise_scale * np.sqrt(1-gamma) * randn_like(y)
        if rescale:
                y_scaled = (y - torch.mean(y)) / torch.std(y) * y_std + y_mean
                if not torch.isnan(y_scaled).any():
                        y = psi * y_scaled + (1 - psi) * y
                else:
                        logger.debug("Warning: NaN encountered in rescaling")
        return y


def cads_annealing(text_embeddings, t, sampling_step, total_sampling_step):
    t = 1.0 - max(min(sampling_step / total_sampling_step, 1.0), 0.0)
    print(t)
    gamma = cads_linear_schedule(t) 
    text_embeddings = add_noise(text_embeddings, gamma)
    return text_embeddings
    
@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]



class NullInversion:
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):

        latents = torch.cat([self.latents_img, latents], dim=1)
        control_input = self.control
        latent_model_input = latents

        latent_model_input = self.scheduler.scale_model_input(
            latent_model_input, t
        )
        noise_pred = self.model.unet(
            latent_model_input,
            t,
            self.stride,
            encoder_hidden_states=context,
            control=control_input,
        ).sample.to(dtype=latents.dtype)
        # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        # noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)


        # noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):

        if context is None:
            context = self.context

        guidance_scale = 1 if is_forward else 7.5


        latents = torch.cat([self.latents_img, latents], dim=1)
        latent_model_input = (
            torch.cat([latents] * 2) 
        )
        control = self.control
        control_input = (
                    torch.cat([control] * 2)
                    if (control is not None)
                    else control
                )

        latent_model_input = self.model.scheduler.scale_model_input(
            latent_model_input, t
        )
        noise_pred = self.model.unet(
            latent_model_input,
            t,
            self.stride,
            encoder_hidden_states=context,
            control=control_input,
        ).sample.to(dtype=latents.dtype)

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents[:,4:,...])
        else:
            latents = self.prev_step(noise_pred, t, latents[:,4:,...])
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(25):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, latent):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * 25)
        for i in range(25):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                print('curr loss')
                print(loss_item)
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list
    
    def invert(self, ref_flow_latent, prompt: str, offsets=(0,0,0,0), num_inner_steps=20, early_stop_epsilon=1e-5, verbose=False):
        # self.init_prompt(prompt)
        # ptp_utils.register_attention_control(self.model, None)
        # image_gt = ref_flow
        if verbose:
            print("DDIM inversion...")
        # image_rec, ddim_latents = self.ddim_inversion(image_gt)
        ddim_latents = self.ddim_loop(ref_flow_latent)
        # if verbose:
        #     print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return ddim_latents[-1], uncond_embeddings

    def __init__(self, model, latents_img, control, text_embeddings):
        # scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
        #                           set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.latents_img = latents_img
        self.stride = torch.tensor([list(range(8, 121, 8))]).cuda()
        self.control = control
        # self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = text_embeddings



class FlowGenPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae_img: AutoencoderKL,
        vae_flow: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if (
            hasattr(scheduler.config, "steps_offset")
            and scheduler.config.steps_offset != 1
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if (
            hasattr(scheduler.config, "clip_sample")
            and scheduler.config.clip_sample is True
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(
            unet.config, "_diffusers_version"
        ) and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse(
            "0.9.0.dev0"
        )
        is_unet_sample_size_less_64 = (
            hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate(
                "sample_size<64", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae_img=vae_img,
            vae_flow=vae_flow,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae_flow.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae_flow.enable_slicing()

    def disable_vae_slicing(self):
        self.vae_flow.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_videos_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_videos_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents, stay_on_device=False):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        #video_length = 4
        # video = self.vae.decode(latents).sample
        # print('latents shape')
        video = []

        def decode_forward(*inputs):
           return self.vae_flow.decode(*inputs).sample #.to(dtype=latents_dtype)
        def freeze_decoder(decoder):
            for param in decoder.parameters():
                param.requires_grad = False
        
        # Use this function before your training loop
        freeze_decoder(self.vae_flow)
        
        if(stay_on_device):
            #return latents
            #self.vae_flow.train()
            # # set self.vae to require grad
            # for param in self.vae_flow.parameters():
            #     param.requires_grad = True 
            # print('latents shape')
            # print(latents.shape)
            for frame_idx in tqdm(range(latents.shape[0])):
                # frame = self.vae_flow.decode(latents[frame_idx : frame_idx + 1]).sample
                frame = checkpoint.checkpoint(decode_forward, latents[frame_idx : frame_idx + 1])
                #frame = frame.sample

                video.append(frame)
                del frame  # Ensure the intermediate tensor is deleted
                torch.cuda.empty_cache()      
                #print('printing frame')
                #print(frame_idx)
                # video.append(
                #     self.vae_flow.decode(latents[frame_idx : frame_idx + 1]).sample
                # )
            video = torch.cat(video)
            video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
            return video

        else:                    
            for frame_idx in tqdm(range(latents.shape[0])):
                    # print(frame_idx)
                    video.append(
                        self.vae_flow.decode(latents[frame_idx : frame_idx + 1]).sample
                    )
                
            # if(stay_on_device):
            #     curr_flow = self.vae_flow.decode(latents[frame_idx : frame_idx + 1]).sample
            #     curr_flow = rearrange(curr_flow, "(b f) c h w -> b c f h w", f = 1)
            #     return curr_flow
            video = torch.cat(video)
            video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
            #video.
        # video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.detach().cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        video_length,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(
                        shape, generator=generator[i], device=rand_device, dtype=dtype
                    )
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(
                    shape, generator=generator, device=rand_device, dtype=dtype
                ).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                )
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def __call__(
        self,
        prompt: Union[str, List[str]],
        stride: Optional[torch.FloatTensor],
        video_length: Optional[int],
        first_frame: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        brush_mask: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        control: Optional[torch.FloatTensor] = None,
        guidance: bool = False,
        particle_guidance: bool = False,
        controlnet_switch: bool = False,
        output_dir: Optional[str] = "",
        **kwargs,
    ):

        with torch.no_grad():
            
            # Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor
    
            # Check inputs. Raise error if not correct
            self.check_inputs(prompt, height, width, callback_steps)
    
            # Define call parameters
            # batch_size = 1 if isinstance(prompt, str) else len(prompt)
            batch_size = 1
            if latents is not None:
                batch_size = latents.shape[0]
            if isinstance(prompt, list):
                batch_size = len(prompt)
    
            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0
    
            # Encode input prompt
            prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
            if negative_prompt is not None:
                negative_prompt = (
                    negative_prompt
                    if isinstance(negative_prompt, list)
                    else [negative_prompt] * batch_size
                )
            text_embeddings = self._encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
            )
    
            # Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            # Prepare extra step kwargs.
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    
            # Prepare latent variables
            # generator = torch.Generator(device = 'cuda')
            # seed = 3
            # print('SEED IS')
            # print(seed)
            # generator = generator.manual_seed(seed)
            num_channels_latents = self.unet.in_channels

            # latents = self.prepare_latents(
            #     1,
            #     num_channels_latents,
            #     video_length,
            #     height,
            #     width,
            #     latent_dtype,
            #     device,
            #     None,
            #     latents,
            # )

            # latents = latents[:, :4, ...]
            # latents_dtype = latents.dtype


            # flow_shape = (1,
            #          2,
            #     height,
            #     width,
            # )
        
            if(len(first_frame.shape) == 4):
                first_frame = first_frame[None].to(device)[0]
            else:
                first_frame = first_frame[None].to(device)
                    
            latents_img = self.vae_img.encode(first_frame.float()).latent_dist
            #latents_img.requires_grad_(True)
            latents_img = latents_img.sample() * 0.18215
            latents_img = rearrange(latents_img, "(b f) c h w -> b c f h w", f=1)
            latents_img = latents_img.repeat(1, 1, video_length, 1, 1)
        
            best_score = 1000000
            num_samples = 5
            selected_latent = None

            motion_path = os.path.join(output_dir, 'generated_motions')
            if not os.path.exists(motion_path):
                os.makedirs(motion_path)
                print(f"Created directory: {motion_path}")
            else:
                print(f"Directory already exists: {motion_path}")


            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                None,
            )
            short_latents = latents[:, 4:, ...]
            latents_dtype = latents.dtype 
            
        
            # Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        
        # for downsample_block in self.unet.down_blocks:
        #     self.unet._set_gradient_checkpointing(downsample_block, True)

        # self.unet._set_gradient_checkpointing(self.unet.mid_block, True)

        # for upsample_block in self.unet.up_blocks:
        #     self.unet._set_gradient_checkpointing(upsample_block, True)
            

        # def unet_forward(*inputs):
        #     return self.unet(*inputs).sample.to(dtype=latents_dtype)

        gc.collect()
        torch.cuda.empty_cache()

        sparse_flow = control[:,:4,...].clone()
        sparse_mask = control[:,4:,...].clone()


        #del control

        # vis_brush_mask = brush_mask.squeeze(0).squeeze(0).squeeze(0).detach().cpu().numpy()
        # print(vis_brush_mask.shape)
        # from PIL import Image
        # Image.fromarray((vis_brush_mask*255.0).astype(np.uint8)).save('brush_mask.png')

        total_loss = 0.0
        total_norm = 0.0
        
                                            
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                print('pre unet')
                if(latents.shape[1]==8):
                    short_latents = latents[:, 4:, ...].detach().clone().requires_grad_(True)
                else:
                    short_latents = latents.detach().clone().requires_grad_(True)


                control = torch.cat([sparse_flow, sparse_mask], dim = 1)    

                if(not controlnet_switch):
                    control = None
                
            
                latents = torch.cat([latents_img, short_latents], dim=1)
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                control_input = (
                    torch.cat([control] * 2)
                    if (do_classifier_free_guidance and control is not None)
                    else control
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )



                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        stride,
                        encoder_hidden_states=text_embeddings,
                        control=control_input,
                    ).sample.to(dtype=latents_dtype)
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    del latent_model_input
                    # break
                                    
                with torch.no_grad():            

                    latents[:, 4:, ...] = short_latents

                    original_pred = self.scheduler.step(
                        noise_pred, t, latents[:, 4:, ...], **extra_step_kwargs
                    ).pred_original_sample 

                    latents = self.scheduler.step(
                        noise_pred, t, latents[:, 4:, ...], **extra_step_kwargs
                    ).prev_sample

    
                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)
                    #noise_pred.detach_()
                    del noise_pred, noise_pred_text, noise_pred_uncond
                    video = self.decode_latents(original_pred)
                    video = torch.from_numpy(video)  
                
                torch.cuda.empty_cache()
                gc.collect()

        # Post-processing
        with torch.no_grad():
            
            if(latents.shape[1]==8):
                latents = latents[:,4:,...]
        
            video = self.decode_latents(latents)
            num_files = len([f for f in os.listdir(motion_path) if os.path.isfile(os.path.join(motion_path, f))])

            np.save(os.path.join(output_dir, 'flow.npy'), video)

            # if(particle_guidance):
            #     np.save(motion_path + '/' + str(num_files) + '.npy', video)
            #     np.savetxt(motion_path + '/' + str(num_files) + '_loss_val.txt', np.array([total_loss.detach().cpu().numpy(), total_norm.detach().cpu().numpy()]))
            
            # print(video.mean())

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)
            
            save_flow_grid_pillow_arrow(video[0], os.path.join(output_dir,'flow.gif'))

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video), latents

def detach_all(*tensors):
    return [t.detach() if isinstance(t, torch.Tensor) else t for t in tensors]
