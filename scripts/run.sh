#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py --text "bouquet of flowers sitting in a clear glass vase" --workspace trial_seed42_flowers --cuda_ray --fp16 # --gui
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py --text "a small green vase displays some small yellow blooms" --workspace trial_seed42_vase --cuda_ray --fp16 # --gui
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py --text "a slug crawling on the ground around flower petals" --workspace trial_seed42_slug --cuda_ray --fp16 # --gui
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py --text "a man" --workspace trial_seed42_man --cuda_ray --fp16 # --gui
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py --text "armchair in the sahpe of an avocado" --workspace trial_seed42_avocado_chair --cuda_ray --fp16 # --gui
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py --text "teapot in the sahpe of an avocado" --workspace trial_seed42_avocado_teapot --cuda_ray --fp16 # --gui
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py --text "a bird that has many colors on it" --workspace trial_seed42_bird --cuda_ray --fp16 # --gui
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py --text "a blue jug in a garden filled with mud" --workspace trial_seed42_jug --cuda_ray --fp16 # --gui
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py --text "cthulhu" --workspace trial_seed42_cthulhu --cuda_ray --fp16 # --gui