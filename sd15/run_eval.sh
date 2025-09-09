#!/usr/bin/env bash
set -euo pipefail


echo "[1/2] Generating images with SD 1.5"

python generate_images_sd15.py \
    --csv_path MS-COCO_val2014_30k_captions.csv \
    --batch_size 24 \
    --resolution 512 \
    --steps 50 \
    --max_samples 5000 \
    --output_dir generated_images_512_sd15_step50_cfg6
    
# echo "[2/2] Calculating FID scores..."

python fid_val_v2.py \
    --generated_dir generated_images_512_sd15_step50_cfg6 \
    --csv_path MS-COCO_val2014_30k_captions.csv \
    --max_images 5000 \
    --use_30k_subset \
    --download \
    --save_real_stats \
    --save_gen_stats \
    --batch_size 12

# echo "Evaluation pipeline completed."

# FID:  100.62430369126565

# FID:  58.82201993812208
