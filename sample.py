#!/usr/bin/env python3

"""
Helpful util created primarily for checkpoint sampling while you are
in the middle of a training run.
This is why it defaults to device=cpu.
However, when not doing a run, you can run it faster with
  --cuda
"""

import argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="Directory with the model checkpoint")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cfg", type=float, default=7)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--prompt", nargs="+", type=str, help="Prompt(s) to use for sample image")
    return ap.parse_args()

args = parse_args()

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import sys, os

MODEL_DIR = "sd-flux2-phase0-notext/checkpoint-00200"

PROMPTS = ["A collie dog","A beautiful woman"]
STEPS = 30
OUTFILE = "sample"

def main() -> int:
#    device = "cuda" if torch.cuda.is_available() else "cpu"
    global args, PROMPTS

    if args.cuda:
        device = "cuda"
    else:
        device = "cpu"
    print("Using device:", device)

    #dtype = torch.float16 if device == "cuda" else torch.float32
    # DO NOT do the above, it has overflow problems with sd15
    dtype = torch.float32

    MODEL_DIR = args.dir
    print("Using", MODEL_DIR)
    if not os.path.exists(MODEL_DIR):
        raise SystemExit("Error: non-existant directory", MODEL_DIR)

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_DIR, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    g = torch.Generator(device=device).manual_seed(args.seed)

    if args.prompt:
        PROMPTS = args.prompt

    for i in range(len(PROMPTS)):
        with torch.inference_mode():
            img = pipe(
                prompt=PROMPTS[i],
                num_inference_steps=STEPS,
                guidance_scale=args.cfg,
                generator=g,
            ).images[0]
        outpath=MODEL_DIR + f"/" + OUTFILE + f"_{i}_g{args.cfg}.png"
        img.save(outpath)
        print(f"saved: {outpath}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
