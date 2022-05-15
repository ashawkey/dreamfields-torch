#! /bin/bash

# "a bird that has many colors on it"
# "an illustration of a pumpkin on the vine"
# "an armchair in the shape of an avocado"
# "a bouquet of roses in a vase"
# "a sculpture of a rooster."
# "a tray that has meat and carrots on a table."
# "a small green vase displays some small yellow blooms."
# "a high-quality 3d render of a jenga tower"
# "a slug crawling on the ground around flower petals."
# "bouquet of flowers sitting in a clear glass vase."

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py "a fox" --workspace trial --cuda_ray --gui
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py "a rainbow cat" --workspace trial --fp16 --ff --gui --seed 10
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py "a watermelon with a knife on it" --workspace trial --fp16 --ff --cuda_ray --gui --seed 42

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py --text "a red fox." --workspace trial --cuda_ray --gui --fp16 --seed 2
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py --image ./data/redfox.jpg --workspace trial --cuda_ray --gui --fp16 --seed 2 #--dir_text
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py --text "a red fox in a black sofa." --workspace trial --cuda_ray --gui --fp16 --seed 43 #--dir_text


OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py --text "an illustration of a pumpkin on the vine" --workspace trial --cuda_ray --gui --seed 42