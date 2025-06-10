# Motion Modes

Preliminary code release for the paper "Motion Modes: What Could Happen Next?" at CVPR 2025. 

Karran Pandey, Matheus Gadelha, Yannick Hold-Geoffroy, Karan Singh, Niloy J. Mitra, Paul Guerrero

We will add parameter controls and potential integration with other models in future code updates.

## Usage

1. Install environments
```shell
conda env create -f environment.yaml
```
2. Download models
```shell
git clone https://huggingface.co/wangfuyun/Motion-I2V
```
3. Set input data path in ```frame_data.json``` - look at current path for example setup.

4. Run the code
```shell
python motion_discovery.py 
```
## References

We use Motion-I2V as our backbone for motion generation. 

```bib
@article{shi2024motion,
            title={Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling},
            author={Shi, Xiaoyu and Huang, Zhaoyang and Wang, Fu-Yun and Bian, Weikang and Li, Dasong and Zhang, Yi and Zhang, Manyuan and Cheung, Ka Chun and See, Simon and Qin, Hongwei and others},
            journal={SIGGRAPH 2024},
            year={2024}
            }
}
```
