# generation.py

import random
import numpy as np
import torch
from diffusers.utils import load_image
from .style_template import styles
from .watermark import add_watermark
from .face_utils import extract_left_right_embeddings



MAX_SEED = np.iinfo(np.int32).max


def validate_trigger_word(pipe, prompt):
    token_id = pipe.tokenizer.convert_tokens_to_ids(pipe.trigger_word)
    ids = pipe.tokenizer.encode(prompt)

    if token_id not in ids:
        raise ValueError(f"Trigger word '{pipe.trigger_word}' missing in prompt: {prompt}")

    if ids.count(token_id) > 1:
        raise ValueError(f"Multiple trigger words '{pipe.trigger_word}' found in prompt: {prompt}")


def apply_style(style_name, positive, negative):
    default = "Photographic (Default)"
    p, n = styles.get(style_name, styles[default])
    return p.replace("{prompt}", positive), n + " " + negative


def generate_images(pipe, face_detector, config):
    # Load images
    input_images = [load_image(p) for p in config["INPUT_IMAGES"]]

    # Extract embeddings
    id_left, id_right = extract_left_right_embeddings(face_detector, input_images[0])

    # Seed
    seed = config["SEED"] if config["SEED"] is not None else random.randint(0, MAX_SEED)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # Merge step
    start_merge_step = int(float(config["STYLE_STRENGTH_RATIO"]) / 100 * config["NUM_STEPS"])
    start_merge_step = min(start_merge_step, 30)

    # Output containers
    left_results = []
    right_results = []

    # LEFT FACE
    for prompt_text in config["PROMPTS_FACE_LEFT"]:
        validate_trigger_word(pipe, prompt_text)
        prompt, neg = apply_style(config["STYLE_NAME"], prompt_text, config["NEGATIVE_PROMPT"])

        imgs = pipe(
            prompt=prompt,
            width=config["OUTPUT_WIDTH"],
            height=config["OUTPUT_HEIGHT"],
            input_id_images=input_images,
            negative_prompt=neg,
            num_images_per_prompt=config["NUM_OUTPUTS"],
            num_inference_steps=config["NUM_STEPS"],
            start_merge_step=start_merge_step,
            generator=generator,
            guidance_scale=config["GUIDANCE_SCALE"],
            id_embeds=id_left,
        ).images

        left_results.append((prompt_text, imgs))

    # RIGHT FACE
    for prompt_text in config["PROMPTS_FACE_RIGHT"]:
        validate_trigger_word(pipe, prompt_text)
        prompt, neg = apply_style(config["STYLE_NAME"], prompt_text, config["NEGATIVE_PROMPT"])

        imgs = pipe(
            prompt=prompt,
            width=config["OUTPUT_WIDTH"],
            height=config["OUTPUT_HEIGHT"],
            input_id_images=input_images,
            negative_prompt=neg,
            num_images_per_prompt=config["NUM_OUTPUTS"],
            num_inference_steps=config["NUM_STEPS"],
            start_merge_step=start_merge_step,
            generator=generator,
            guidance_scale=config["GUIDANCE_SCALE"],
            id_embeds=id_right,
        ).images

        right_results.append((prompt_text, imgs))

    return left_results, right_results, seed
