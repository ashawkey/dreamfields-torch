#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=4 python main_nerf.py --text "an illustration of a pumpkin on the vine" --workspace trial_ppk --cuda_ray --fp16