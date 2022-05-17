#! /bin/bash

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py --text "cthulhu" --workspace trial_cth --cuda_ray --fp16 #--gui
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=4 python main_nerf.py --text "cthulhu" --workspace trial_cc_cth --cuda_ray --fp16 --cc

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py --text "cthulhu" --workspace trial_bird --cuda_ray --fp16 --seed 16 # --gui
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=4 python main_nerf.py --text "an illustration of a pumpkin on the vine" --workspace trial_ppk --cuda_ray --fp16