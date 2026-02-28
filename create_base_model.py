#!/usr/bin/env python3

"""
Download sd1.5 model
replace vae with flux2 klein vae
tweak unet config to match
tweak unet layers conv_in and conv_out to match
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline

try:
    # Present in newer diffusers builds that support FLUX.2
    from diffusers import AutoencoderKLFlux2
except Exception as e:
    raise SystemExit(
        "Your diffusers install does not expose AutoencoderKLFlux2.\n"
        "Upgrade diffusers to a version that supports FLUX.2-klein.\n"
        f"Import error: {e}"
    )


NEW_C = 32
OUT_DIR_DEFAULT = "sd-flux2-alpha00"
SD15_DEFAULT = "stable-diffusion-v1-5/stable-diffusion-v1-5"
FLUX2_KLEIN_DEFAULT = "black-forest-labs/FLUX.2-klein-4B"


def _new_conv_like(conv: nn.Conv2d, *, in_c: int, out_c: int) -> nn.Conv2d:
    if conv.groups != 1:
        raise ValueError(f"Unsupported: conv.groups={conv.groups} (expected 1)")
    return nn.Conv2d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)


def widen_unet_latent_io_to_32(unet) -> None:
    old_in = int(unet.config.in_channels)
    old_out = int(unet.config.out_channels)
    if old_in != 4 or old_out != 4:
        raise ValueError(f"Expected SD1.5 UNet in/out = 4/4, got {old_in}/{old_out}")

    # Convert conv_in: (320, 4, k, k) -> (320, 32, k, k)
    old = unet.conv_in
    new = _new_conv_like(old, in_c=NEW_C, out_c=old.out_channels)
    with torch.no_grad():
        new.weight.zero_()
        if new.bias is not None:
            new.bias.copy_(old.bias)
        new.weight[:, :4, :, :].copy_(old.weight[:, :4, :, :])
    unet.conv_in = new

    old = unet.conv_out
    new = _new_conv_like(old, in_c=old.in_channels, out_c=NEW_C)
    with torch.no_grad():
        new.weight.zero_()
        if new.bias is not None:
            new.bias.zero_()

        # copy original 0..3
        new.weight[:4, :, :, :].copy_(old.weight[:4, :, :, :])
        if old.bias is not None:
            new.bias[:4].copy_(old.bias[:4])

        # IMPORTANT: repeat outputs 0..3 into 4..31 so extra channels are denoised from day 0.
        # This gives better training base than random Kaiming init.
        # But only for conv_out specifically
        for c in range(4, NEW_C):
            src = c % 4
            new.weight[c, :, :, :].copy_(old.weight[src, :, :, :])
            if old.bias is not None:
                new.bias[c].copy_(old.bias[src])

    unet.conv_out = new
    unet.register_to_config(in_channels=NEW_C, out_channels=NEW_C)
    print("UNet: widened latent I/O 4->32 (conv_in/conv_out only).")


def load_flux2_klein_vae(repo_id: str, *, dtype: torch.dtype) -> AutoencoderKLFlux2:
    vae = AutoencoderKLFlux2.from_pretrained(repo_id, subfolder="vae", torch_dtype=dtype)

    # StableDiffusionPipeline expects vae.config.scaling_factor to exist and uses it
    # to scale latents before/after decode/encode. Flux2 VAE configs do not include
    # SD’s 0.18215 convention, so force to 1.0 for this hybrid pipeline.
    vae.register_to_config(scaling_factor=1.0)
    return vae


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sd15", default=SD15_DEFAULT, help="SD1.5 diffusers repo or local dir")
    ap.add_argument("--flux2", default=FLUX2_KLEIN_DEFAULT, help="Flux2 klein repo (VAE is in subfolder 'vae')")
    ap.add_argument("--out", default=OUT_DIR_DEFAULT, help="Output directory")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32", help="Weights dtype to save")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16

    # Load SD1.5 pipeline
    pipe = StableDiffusionPipeline.from_pretrained(args.sd15, torch_dtype=dtype)

    # Patch UNet to 32ch latent I/O
    widen_unet_latent_io_to_32(pipe.unet)

    # Replace VAE with Flux2 klein VAE (32ch)
    flux_vae = load_flux2_klein_vae(args.flux2, dtype=dtype)
    pipe.register_modules(vae=flux_vae)

    # Refresh derived pipeline attributes that depend on VAE
    pipe.vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)

    pipe.save_pretrained(str(out_dir), safe_serialization=True)
    print(f"Saved hybrid pipeline to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
