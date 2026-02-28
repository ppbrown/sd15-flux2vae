#!/bin/env python

# This is a specialised verson of the tools in
#  https://github.com/ppbrown/ai-training/tree/main/trainer/cache-utils
# hardcoded to use Flux2 type vae (class AutoencoderKLFlux2)
# It will generate flux2 vae cache files for all png and jpg images found under
#   --data_root
# It will optionally write a "preview" of what the vae cache re-expands to.
#  (This tests how well the vae can reconstitute any one particular image)
#
# WARNING: This tool is intended for use with model training. As such, it will rescale
# and center-crop images to the size it expects.
# Default size is 512x512 pixels

device = "cuda"
import argparse

def parse_args():
    global device

    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="black-forest-labs/FLUX.2-klein-4B",
        help="HF repo or local dir(expects full model unless you use --vae)",
    )
    p.add_argument("--vae", action="store_true", help="Treat model as direct vae, not full pipeline")
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--writepreview",
        action="store_true",
        default=False,
        help="Add a webp view of the cache data. Useful for quality checking.",
    )
    p.add_argument(
        "--data_root",
        required=True,
        help="Directory containing images (recursively searched)",
    )
    p.add_argument(
        "--out_suffix",
        default=".img_flux2",
        help="File suffix for saved latents (default: .img_flux2)",
    )
    p.add_argument("--target_width", type=int, default=512, help="Width Resolution for images (default: 512)")
    p.add_argument("--target_height", type=int, default=512, help="Height Resolution for images (default: 512)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--extensions", nargs="+", default=["jpg", "jpeg", "png"])
    p.add_argument("--custom", action="store_true", help="Treat model as custom pipeline")
    args = p.parse_args()

    if args.cpu:
        device = "cpu"

    return args


# do early for fast usage return
args = parse_args()

######################################################################

from pathlib import Path
import os
import sys
import subprocess

# This stuff required for the "deterministic" settings
ENV_VAR = "CUBLAS_WORKSPACE_CONFIG"
DESIRED = ":4096:8"  # Or ":16:8" if preferred

if os.environ.get(ENV_VAR) != DESIRED:
    os.environ[ENV_VAR] = DESIRED
    print(f"[INFO] Setting {ENV_VAR}={DESIRED}.")


from tqdm.auto import tqdm

import torch
import torchvision.transforms as TVT
from torchvision.transforms import InterpolationMode as IM
import torchvision.transforms.functional as F

import safetensors.torch as st
from diffusers import AutoencoderKLFlux2
from PIL import Image

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def find_images(input_dir, exts):
    images = []
    for ext in exts:
        images += list(Path(input_dir).rglob(f"*.{ext}"))
    return sorted(images)


# Resize to height, while preserving aspect ratio. Then crop to width.
def make_cover_resize_center_crop(target_width: int, target_height: int):
    def _f(img):
        src_height, src_width = img.height, img.width
        scale = max(target_width / src_width, target_height / src_height)
        resized_width, resized_height = round(src_width * scale), round(src_height * scale)
        img2 = F.resize(img, (resized_height, resized_width), interpolation=IM.BICUBIC, antialias=True)
        return F.center_crop(img2, (target_height, target_width))

    return _f


def get_transform(width, height):
    return TVT.Compose(
        [
            lambda im: im.convert("RGB"),
            make_cover_resize_center_crop(width, height),
            TVT.ToTensor(),
            # Must be in [-1, 1] before VAE encode
            TVT.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def load_vae_fp32(model_id: str, direct_vae_repo: bool, VAE_CLASS):
    global device

    if direct_vae_repo:
        vae = VAE_CLASS.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
    else:
        vae = VAE_CLASS.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        )

    vae.to(device)
    vae.eval()
    return vae


def tensor_to_pil_rgb(img_chw: torch.Tensor) -> Image.Image:
    """
    img_chw: float tensor in [0, 1], shape (3, H, W)
    """
    img_chw = img_chw.clamp(0.0, 1.0)
    img_hwc_u8 = (img_chw.permute(1, 2, 0) * 255.0).round().to(torch.uint8).cpu().numpy()
    # Pillow 13 deprecates the explicit mode= parameter here; omit it.
    return Image.fromarray(img_hwc_u8)


@torch.no_grad()
def main():
    global args

    VAE_CLASS = AutoencoderKLFlux2

    vae = load_vae_fp32(args.model, args.vae, VAE_CLASS)

    # Collect images
    all_image_paths = find_images(args.data_root, args.extensions)
    image_paths = []
    skipped = 0
    for path in all_image_paths:
        out_path = path.with_name(path.stem + args.out_suffix)
        if out_path.exists():
            skipped += 1
            continue
        image_paths.append(path)

    if not image_paths:
        print("No new images to process (all cache files exist).")
        return
    if skipped:
        print(f"Skipped {skipped} files with existing cache.")

    tfm = get_transform(args.target_width, args.target_height)

    print(f"Processing {len(image_paths)} images from {args.data_root}")
    print("Batch size is", args.batch_size)
    print(f"Using {args.model} to create ({args.out_suffix}) caches...")
    if args.writepreview:
        print("(Also Writing WEBP cache reconstruction files)")
    print("")

    for i in tqdm(range(0, len(image_paths), args.batch_size)):
        batch_paths = image_paths[i : i + args.batch_size]
        batch_imgs = []
        valid_paths = []

        for path in batch_paths:
            try:
                img = Image.open(path)
                batch_imgs.append(tfm(img))
                valid_paths.append(path)
            except Exception as e:
                print(f"Could not load {path}: {e}")

        if not batch_imgs:
            continue

        batch_tensor = torch.stack(batch_imgs).to(device)  # (B, C, H, W)

        latents = vae.encode(batch_tensor).latent_dist.mean  # raw latent, no scaling

        # Optional preview: decode in-memory latents back to pixels and save as WEBP
        if args.writepreview:
            latents_for_decode = latents

            decoded = vae.decode(latents_for_decode).sample

            # decoded is typically in [-1, 1]; convert to [0, 1]
            decoded_01 = (decoded / 2.0 + 0.5).clamp(0.0, 1.0).detach().cpu()

            for j, path in enumerate(valid_paths):
                preview_path = path.with_name(path.stem + args.out_suffix + ".webp")
                pil = tensor_to_pil_rgb(decoded_01[j])
                # Use lossless to avoid WEBP artifacts affecting your qualitative VAE check.
                pil.save(str(preview_path), format="WEBP", lossless=True, quality=100, method=6)

        # Save latents (make contiguous so safetensors is happy)
        latents_cpu = latents.detach().cpu().contiguous()

        for j, path in enumerate(valid_paths):
            out_path = path.with_name(path.stem + args.out_suffix)
            st.save_file({"latent": latents_cpu[j]}, str(out_path))


if __name__ == "__main__":
    main()
