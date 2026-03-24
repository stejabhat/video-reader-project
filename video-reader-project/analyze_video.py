#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image

# FastVLM imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

# ---------------------------------------------------------------------------
# Frame Extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path, frames_dir, fps=1):
    Path(frames_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))

    frame_paths = []
    frame_idx = saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            path = os.path.join(frames_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(path, frame)
            frame_paths.append(path)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"[INFO] Extracted {saved} frames")
    return frame_paths


# ---------------------------------------------------------------------------
# Load Model (ONCE)
# ---------------------------------------------------------------------------

def load_model(model_path):
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    print(f"[INFO] Loading {model_name}...")

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name, device="mps"
    )

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    print("[INFO] Model loaded ✓")
    return tokenizer, model, image_processor


# ---------------------------------------------------------------------------
# Preprocess Frame
# ---------------------------------------------------------------------------

def preprocess_frame(image_path, image_processor, model):
    image = Image.open(image_path).convert("RGB")

    # Resize for speed (important)
    image = image.resize((224, 224))

    image_tensor = process_images([image], image_processor, model.config)[0]
    return image, image_tensor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(image, image_tensor, tokenizer, model, prompt, conv_mode="qwen_2"):
    if model.config.mm_use_im_start_end:
        prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
    else:
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)

    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to("mps")

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to("mps", dtype=torch.float16),
            image_sizes=[image.size],
            do_sample=False,
            temperature=0.2,
            max_new_tokens=128,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# 🧠 CLEAN + COMPRESS OUTPUT (KEY PART)
# ---------------------------------------------------------------------------

def clean_descriptions(results):
    cleaned = []

    for r in results:
        r = r.strip()

        # Skip near-duplicate consecutive outputs
        if cleaned and r.lower()[:60] == cleaned[-1].lower()[:60]:
            continue

        cleaned.append(r)

    return cleaned


def compress_descriptions(descriptions):
    summary_tags = []

    for d in descriptions:
        d = d.lower()

        if "walk" in d:
            summary_tags.append("person walking")
        elif "sit" in d:
            summary_tags.append("person sitting at bar")
        elif "conversation" in d or "talk" in d:
            summary_tags.append("people having conversation")
        elif "stand" in d:
            summary_tags.append("person standing")

    return list(dict.fromkeys(summary_tags))


def build_final_summary(descriptions):
    cleaned = clean_descriptions(descriptions)
    compressed = compress_descriptions(cleaned)

    if not compressed:
        return "No clear actions detected."

    return "Video shows: " + ", ".join(compressed)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--model-path",
        default="checkpoints/llava-fastvithd_0.5b_stage3",
    )
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--frames-dir", default="frames")
    parser.add_argument(
        "--prompt",
        default="This is a frame from a video. Describe the main action happening.",
    )

    args = parser.parse_args()

    # STEP 1: Extract frames
    frames = extract_frames(args.video, args.frames_dir, fps=args.fps)

    # STEP 2: Load model ONCE
    tokenizer, model, image_processor = load_model(args.model_path)

    # STEP 3: Process frames
    results = []

    for i, frame_path in enumerate(frames):
        print(f"[FRAME {i+1}/{len(frames)}]")

        # preprocess
        image, image_tensor = preprocess_frame(frame_path, image_processor, model)

        # inference
        output = run_inference(
            image,
            image_tensor,
            tokenizer,
            model,
            args.prompt,
        )

        print("→", output, "\n")

        results.append(output)

    # STEP 4: Final Summary
    print("\n========== FINAL SUMMARY ==========")
    summary = build_final_summary(results)
    print(summary)


if __name__ == "__main__":
    main()