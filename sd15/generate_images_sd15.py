import torch
from diffusers import StableDiffusionPipeline
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import argparse
import time

def generate_images_from_csv(csv_path, output_dir, batch_size=4, resolution=256, num_inference_steps=20, max_samples=10000):
    
    print("Loading Stable Diffusion 1.5...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker=None
    )
    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)  
    
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} captions")
    
    if max_samples and len(df) > max_samples:
        df = df.head(max_samples)
        print(f"Limited to first {max_samples} samples")
    
    os.makedirs(output_dir, exist_ok=True)
    
    generated_count = 0
    failed_count = 0
    start_time = time.time()
    
    for i in tqdm(range(0, len(df), batch_size), desc="Generating images"):
        batch_df = df.iloc[i:i+batch_size]
        captions = batch_df['text'].tolist()  # 'text' == caption column
        image_ids = batch_df['image_id'].tolist()
        
        try:
            with torch.autocast("cuda"):
                images = pipeline(
                    prompt=captions,
                    height=resolution,
                    width=resolution,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=6,
                    num_images_per_prompt=1
                ).images
            
            # save images
            for img, image_id in zip(images, image_ids):
                output_path = os.path.join(output_dir, f"{image_id:012d}.png")
                img.save(output_path)
                generated_count += 1
                
        except Exception as e:
            print(f"Error generating batch {i//batch_size}: {e}")
            failed_count += batch_size
            continue
        
        if generated_count % 24 == 0:
            elapsed = time.time() - start_time
            rate = generated_count / elapsed if elapsed > 0 else 0
            print(f"Generated {generated_count} images ({rate:.1f} img/sec)")
    
    total_time = time.time() - start_time
    print(f"\nCompleted!")
    print(f"Generated: {generated_count} images")
    print(f"Failed: {failed_count} images") 
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Average rate: {generated_count/total_time:.1f} images/second")
    
    return generated_count

def main():
    parser = argparse.ArgumentParser(description="Generate images from CSV captions using SD 1.5")
    parser.add_argument("--csv_path", default="MS-COCO_val2014_30k_captions.csv", help="Path to CSV file")
    parser.add_argument("--output_dir", default="generated_images_step50_cfg7_5", help="Output directory for images")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for generation")
    parser.add_argument("--resolution", type=int, default=256, help="Output image resolution")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum number of samples to process (default: 10000)")
    
    args = parser.parse_args()
    

    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file {args.csv_path} not found")
        return
    
    
    generate_images_from_csv(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        resolution=args.resolution,
        num_inference_steps=args.steps,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()