#!/usr/bin/env python3

from __future__ import annotations
import argparse
import os
import time
from typing import List, Optional

import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image

from diffusers import DiffusionPipeline

# Optional schedulers
try:
    from diffusers import DPMSolverMultistepScheduler
except Exception:  # pragma: no cover
    DPMSolverMultistepScheduler = None  # type: ignore

try:
    from diffusers import EulerAncestralDiscreteScheduler
except Exception:  # pragma: no cover
    EulerAncestralDiscreteScheduler = None  # type: ignore

# Fast attention path via PyTorch SDPA
try:
    from diffusers.models.attention_processor import AttnProcessor2_0
except Exception:  # pragma: no cover
    AttnProcessor2_0 = None  # type: ignore


# ------------------------------
# Pipeline factory
# ------------------------------

def get_sdxl_pipeline(
    scheduler_name: str = "dpmpp",
    torch_dtype: torch.dtype = torch.float16,
    device: Optional[str] = "cuda",
    offload: bool = False,
    enable_tiling: bool = True,
    enable_slicing: bool = True,
    try_xformers: bool = True,
) -> DiffusionPipeline:

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16" if torch_dtype == torch.float16 else None,
    )

    # Choose scheduler
    s = (scheduler_name or "dpmpp").lower()
    if s == "dpmpp" and DPMSolverMultistepScheduler is not None:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras=True
        )
    elif s in {"euler_a", "euler"} and EulerAncestralDiscreteScheduler is not None:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    # else: keep existing

    # VAE fp32 to avoid black images
    # VAE fp32 safely (prevents Half/Float conv2d mismatch)
    try:
        pipe.upcast_vae()  # diffusers >= 0.29
    except Exception:
        # Fallback if older diffusers: still move VAE to fp32
        pipe.vae.to(dtype=torch.float32)

    # Fast attention path
    if AttnProcessor2_0 is not None:
        try:
            pipe.unet.set_attn_processor(AttnProcessor2_0())
        except Exception:
            pass

    # Perf tweaks
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Move to GPU or enable offload
    if offload:
        # Prefer no offload on Windows unless VRAM-starved
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            if device:
                pipe.to(device)
    else:
        if device:
            pipe.to(device)

    # VAE memory helpers
    if enable_tiling:
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
    if enable_slicing:
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

    # (Optional) xFormers — OFF by default because it can stall on Windows
    if try_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    pipe.set_progress_bar_config(disable=True)
    return pipe


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_caption_sidecar(path_no_ext: str, caption: str, model_name: str, res: int, steps: int) -> None:
    sidecar = f"{path_no_ext}_caption.txt"
    with open(sidecar, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}")
        f.write(f"Caption: {caption}")
        f.write(f"Resolution: {res}x{res}")
        f.write(f"Steps: {steps}")



# ------------------------------
# Batch generation
# ------------------------------

def generate_images_sdxl(
    csv_path: str,
    output_dir: str,
    batch_size: int = 2,
    resolution: int = 1024,
    num_inference_steps: int = 40,
    max_samples: int = 3500,
    scheduler: str = "dpmpp",
    dtype: str = "fp16",
    guidance_scale: float = 5.0,
    offload: bool = False,
    seed: Optional[int] = 42,
    save_captions: bool = False,
    offset: int = 0
) -> int:
    """Batch-generate images from CSV with columns [image_id, text]."""

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    td = dtype_map.get(dtype.lower(), torch.float16)

    ##############
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    if not {"image_id", "text"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: image_id, text")

    total = len(df)

    # (Optional) make order deterministic across PCs; comment this out if you want raw CSV order
    # df = df.sort_values("image_id").reset_index(drop=True)

    # Apply offset + limit safely
    start = offset
    end = start + max_samples if max_samples is not None else total
    end = min(end, total)
    df = df.iloc[start:end].reset_index(drop=True)
    #print(f"Using rows [{start}:{end}) of {total} -> {len(df)} samples")
    #print(df)
    ###############
    
    print("Loading SDXL pipeline...")
    pipe = get_sdxl_pipeline(
        scheduler_name=scheduler,
        torch_dtype=td,
        device="cuda",
        offload=offload,
    )

    # print(f"Loading dataset from {csv_path}...")
    # df = pd.read_csv(csv_path)
    # if not {"image_id", "text"}.issubset(df.columns):
    #     raise ValueError("CSV must contain columns: image_id, text")

    # total = len(df)
    # if total > max_samples:
    #     df = df.head(max_samples)
    #     print(f"Using first {max_samples} samples (of {total})")


    _ensure_dir(output_dir)

    generated = 0
    failed = 0
    start_time = time.time()

    i = 0
    # make sure we use the offset + max_samples correctly
    print(f"Using rows [{start}:{end}) of {total} -> {len(df)} samples")
    print(df)
    print(len(df))
    while i < len(df):
        batch_df = df.iloc[i : i + batch_size]
        captions: List[str] = batch_df["text"].astype(str).tolist()
        image_ids: List[int] = batch_df["image_id"].astype(int).tolist()

        try:
            generator = None
            if seed is not None:
                generator = torch.Generator(device="cuda").manual_seed(seed + i)

            print(f"Generating batch {i // batch_size + 1} with {len(captions)} samples...")
            out = pipe(
                prompt=captions,
                height=resolution,
                width=resolution,
                # original_size=(1024, 1024),
                # target_size=(512, 512),
                crops_coords_top_left=(0, 0),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
                generator=generator,
                output_type="pil",
            )
            images: List[Image.Image] = out.images

            for img, image_id in zip(images, image_ids):
                path = os.path.join(output_dir, f"{int(image_id):012d}.png")
                print("Saving to", path)
                img.save(path)
                if save_captions:
                    _save_caption_sidecar(
                        os.path.join(output_dir, f"{int(image_id):012d}"),
                        df.loc[df["image_id"] == image_id, "text"].values[0],
                        "SDXL Base 1.0",
                        resolution,
                        num_inference_steps,
                    )
                generated += 1
                print("Saved:", generated)

            i += batch_size

        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda error" in msg:
                torch.cuda.empty_cache()
                new_bs = max(1, batch_size // 2)
                if new_bs == batch_size:
                    failed += len(captions)
                    i += batch_size
                    print(f"OOM on batch @idx {i}; skipping {len(captions)} samples")
                else:
                    print(f"OOM detected. Reducing batch size {batch_size} -> {new_bs}")
                    batch_size = new_bs
                continue
            else:
                failed += len(captions)
                i += batch_size
                print(f"Error on batch {i // batch_size}: {e}")
                continue

        # Progress
        if generated and generated % 10 == 0:
            elapsed = time.time() - start_time
            rate = generated / max(elapsed, 1e-6)
            remaining = min(max_samples, len(df)) - generated
            eta = remaining / max(rate, 1e-6)
            print(
                f"Generated {generated}/{min(max_samples, len(df))} images "
                f"({rate:.2f} img/sec, ETA: {eta/3600:.1f}h)"
            )

    total_time = time.time() - start_time
    print("SDXL Generation Completed!")
    print(f"Generated: {generated} images")
    print(f"Failed: {failed} images")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Average rate: {generated/max(total_time, 1e-6):.2f} images/second")

    return generated


# ------------------------------
# Single image generation
# ------------------------------

def generate_single_image_sdxl(
    image_id: int = 623,
    csv_path: str = "MS-COCO_val2014_30k_captions.csv",
    output_dir: str = "test_single_sdxl",
    resolution: int = 512,
    num_inference_steps: int = 40,
    scheduler: str = "dpmpp",
    dtype: str = "fp16",
    guidance_scale: float = 5.0,
    offload: bool = False,
    seed: Optional[int] = 42,
) -> Optional[str]:
    print("Loading SDXL for single image generation...")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    td = dtype_map.get(dtype.lower(), torch.float16)

    pipe = get_sdxl_pipeline(
        scheduler_name=scheduler,
        torch_dtype=td,
        device="cuda",
        offload=offload,
    )

    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    if not {"image_id", "text"}.issubset(df.columns):
        print("Error: CSV must contain columns: image_id, text")
        return None

    target_row = df[df["image_id"] == image_id]
    if target_row.empty:
        print(f"Error: Image ID {image_id} not found in CSV!")
        return None

    caption = str(target_row.iloc[0]["text"])  # ensure str
    print(f"Image ID: {image_id}")
    print(f"Caption: {caption}")

    _ensure_dir(output_dir)

    print(f"Generating with SDXL: {num_inference_steps} steps at {resolution}x{resolution}...")
    start = time.time()

    try:
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        out = pipe(
            prompt=caption,
            height=resolution,
            width=resolution,
            original_size=(1024, 1024),
            target_size=(512, 512),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil",
        )
        image: Image.Image = out.images[0]

        out_path = os.path.join(output_dir, f"SDXL_{int(image_id):012d}.png")
        image.save(out_path)

        print(f"Success! SDXL image saved to: {out_path}")
        print(f"Generation time: {time.time() - start:.2f} seconds")
        return out_path

    except Exception as e:
        print(f"Error generating SDXL image: {e}")
        return None


# ------------------------------
# Compare SD 1.5 vs SDXL
# ------------------------------

def compare_models_single(
    image_id: int = 550403,
    csv_path: str = "MS-COCO_val2014_30k_captions.csv",
    resolution: int = 1024,
    steps: int = 40,
    seed: Optional[int] = 42,
):
    print("=== Generating with SD 1.5 ===")
    from diffusers import StableDiffusionPipeline

    sd15 = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to("cuda")
    try:
        sd15.safety_checker = None
    except Exception:
        pass

    df = pd.read_csv(csv_path)
    if not {"image_id", "text"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: image_id, text")
    target_row = df[df["image_id"] == image_id]
    if target_row.empty:
        raise ValueError(f"Image ID {image_id} not found in CSV!")

    caption = str(target_row.iloc[0]["text"])
    print(f"Caption: {caption}")

    gen = torch.Generator(device="cuda").manual_seed(seed or 42)

    img15 = sd15(
        prompt=caption,
        height=256,
        width=256,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=gen,
        output_type="pil",
    ).images[0]

    img15_path = f"comparison_SD15_{int(image_id)}.png"
    img15.save(img15_path)
    print("SD1.5 image saved")

    del sd15
    torch.cuda.empty_cache()

    print("=== Generating with SDXL ===")
    out_path = generate_single_image_sdxl(
        image_id=image_id,
        csv_path=csv_path,
        output_dir="comparison",
        resolution=resolution,
        num_inference_steps=steps,
        scheduler="dpmpp",
        dtype="fp16",
        guidance_scale=5.0,
        offload=False,
        seed=seed,
    )

    print("Comparison complete! Check:")
    print(f"- {img15_path} (256x256)")
    if out_path:
        print(f"- {out_path} ({resolution}x{resolution})")


# ------------------------------
# Minimal test
# ------------------------------

def test_basic_sdxl():
    print("Testing basic SDXL...")
    pipe = get_sdxl_pipeline(scheduler_name="dpmpp", torch_dtype=torch.float16, device="cuda")

    prompt = "a red apple"
    out = pipe(
        prompt,
        height=512,
        width=512,
        num_inference_steps=20,
        guidance_scale=5.0,
        output_type="pil",
    )
    image: Image.Image = out.images[0]

    import numpy as np
    arr = np.array(image)
    print(f"Image mode: {image.mode}")
    print(f"Image size: {image.size}")
    print(f"Array shape: {arr.shape}, min/max: {arr.min()}/{arr.max()}, unique: {len(np.unique(arr))}")

    image.save("debug_sdxl_test.png")
    print("Saved debug_sdxl_test.png")


# ------------------------------
# CLI
# ------------------------------

def main():
    p = argparse.ArgumentParser(description="Generate images using SDXL from COCO captions")
    p.add_argument("--csv_path", default="MS-COCO_val2014_30k_captions.csv", help="Path to CSV file (needs image_id,text)")
    p.add_argument("--output_dir", default="generated_images_SDXL_3_5k", help="Output directory")
    p.add_argument("--batch_size", type=int, default=2, help="Batch size (SDXL is VRAM-heavy)")
    p.add_argument("--resolution", type=int, default=1024, help="Generation resolution (SDXL native is 1024)")
    p.add_argument("--steps", type=int, default=40, help="Number of diffusion steps")
    p.add_argument("--max_samples", type=int, default=3500, help="Max samples for batch mode")
    p.add_argument("--scheduler", choices=["dpmpp", "euler_a", "euler"], default="dpmpp", help="Scheduler choice")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16", help="Computation dtype for UNet/text enc")
    p.add_argument("--guidance", type=float, default=5.0, help="Guidance scale (SDXL works well ~4.5–6.0)")
    p.add_argument("--offload", action="store_true", help="Enable CPU offload (avoid on Windows unless VRAM-starved)")
    p.add_argument("--seed", type=int, default=42, help="Base random seed (None for random)")
    p.add_argument("--save_captions", action="store_true", help="Save sidecar caption .txt files next to images")
    p.add_argument("--offset", type=int, default=0, help="Row offset into the CSV before sampling")


    # modes
    p.add_argument("--mode", choices=["batch", "single", "compare", "test"], default="batch", help="Run mode")
    p.add_argument("--image_id", type=int, default=550403, help="Image ID for single/compare mode")

    args = p.parse_args()

    if args.mode == "batch":
        if not os.path.exists(args.csv_path):
            print(f"Error: CSV file {args.csv_path} not found!")
            return
        generate_images_sdxl(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            resolution=args.resolution,
            num_inference_steps=args.steps,
            max_samples=args.max_samples,
            scheduler=args.scheduler,
            dtype=args.dtype,
            guidance_scale=args.guidance,
            offload=args.offload,
            seed=args.seed,
            save_captions=args.save_captions,
            offset=args.offset,
        )

    elif args.mode == "single":
        generate_single_image_sdxl(
            image_id=args.image_id,
            csv_path=args.csv_path,
            output_dir="test_single_sdxl",
            resolution=args.resolution,
            num_inference_steps=args.steps,
            scheduler=args.scheduler,
            dtype=args.dtype,
            guidance_scale=args.guidance,
            offload=args.offload,
            seed=args.seed,
        )

    elif args.mode == "compare":
        compare_models_single(
            image_id=args.image_id,
            csv_path=args.csv_path,
            resolution=args.resolution,
            steps=args.steps,
            seed=args.seed,
        )

    elif args.mode == "test":
        test_basic_sdxl()


if __name__ == "__main__":
    main()
