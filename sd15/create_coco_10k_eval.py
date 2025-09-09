import json
import pandas as pd
import random
from pathlib import Path
import os

def create_coco_10k_eval():
    # load annotation
    with open('data/coco/annotations/captions_val2017.json', 'r') as f:
        coco_data = json.load(f)
    
    # image_id to filename mapping
    images = {img['id']: img['file_name'] for img in coco_data['images']}

    captions_data = []
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id in images:
            image_path = f"data/coco/val2017/{images[image_id]}"
            if os.path.exists(image_path):  # check if image file exists
                captions_data.append({
                    'image_id': image_id,
                    'image_path': image_path,
                    'caption': annotation['caption']
                })

    print(f"Found {len(captions_data)} valid caption-image pairs") # check

    # sample 10k random captions
    random.seed(42)
    if len(captions_data) >= 10000:
        coco_10k = random.sample(captions_data, 10000)
    else:
        coco_10k = captions_data
        print(f"Warning: Only {len(captions_data)} samples available, using all")
    
    df = pd.DataFrame(coco_10k)
    df.to_csv('data/coco_10k_eval.csv', index=False)

    print(f"Created COCO 10k evaluation dataset: {len(df)} samples")
    print(f"Saved to: data/coco_10k_eval.csv")

    os.makedirs('generated_images', exist_ok=True)
    os.makedirs('real_images', exist_ok=True)

    return df

if __name__ == "__main__":
    df = create_coco_10k_eval()
    print(f"Sample caption: {df.iloc[0]['caption']}")