# dreamfields-torch (WIP)

A pytorch implementation of [dreamfields](https://github.com/google-research/google-research/tree/master/dreamfields) with modifications, as described in [Zero-Shot Text-Guided Object Generation with Dream Fields](https://arxiv.org/abs/2112.01455).

An example of a generated neural field by prompt "cthulhu" viewed in real-time:

https://user-images.githubusercontent.com/25863658/158593558-a52fe215-4276-41eb-a588-cf60c9461cf3.mp4

# Install

The code framework is based on [torch-ngp](https://github.com/ashawkey/torch-ngp).

```bash
pip install -r requirements.txt

# (optional) install the tcnn backbone
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
Tested on Ubuntu with torch 1.10 & CUDA 11.3 on TITAN RTX.

# Usage

First time running will take some time to compile the CUDA extensions.

```bash
# with GUI
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py "cthulhu" --workspace trial --cuda_ray --gui --fp16

# with CMD
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py "cthulhu" --workspace trial --cuda_ray --fp16
```

check the `scripts` directory for more provided examples.

# Difference from the original implementation
* mip-nerf is not implemented, currently only the original nerf is supported.

# Update Logs
* 3.16: basic reproduction.


# Acknowledgement

* The great paper and official JAX implementation of [dreamfields](https://ajayj.com/dreamfields):
    ```
    @article{jain2021dreamfields,
    author = {Jain, Ajay and Mildenhall, Ben and Barron, Jonathan T. and Abbeel, Pieter and Poole, Ben},
    title = {Zero-Shot Text-Guided Object Generation with Dream Fields},
    journal = {arXiv},
    month = {December},
    year = {2021},
    }   
    ```

* The GUI is developed with [DearPyGui](https://github.com/hoffstadt/DearPyGui).
