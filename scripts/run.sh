#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py "an armchair in the shape of an avocado" --workspace trial --cuda_ray --fp16