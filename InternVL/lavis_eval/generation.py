"""InternVL-native image preprocessing + generation helper.

The prompt fed to ``model.chat()`` is just ``<image>\\n{question}`` — the
InternVL chat template wraps it appropriately. Optionally, a single-word
short-answer hint can be appended (``Answer the question using a single
word or phrase.``); this is the prompt InternVL is officially evaluated
with on VQAv2 / OK-VQA / GQA in VLMEvalKit, so it remains a pure-InternVL
structure (no BLIP-style ``Short answer:`` wording).
"""
from __future__ import annotations

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_pixel_values(image, input_size=448, max_num=12):
    transform = _build_transform(input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(img) for img in images])


SHORT_ANSWER_HINT = "Answer the question using a single word or phrase."


def generate_response(
    model,
    tokenizer,
    image,
    question: str,
    device,
    max_new_tokens: int = 10,
    num_beams: int = 5,
    max_tiles: int = 12,
    short_hint: bool = True,
):
    """Run a single image+question through InternVL.

    When ``short_hint`` is True (default for VQA tasks), the prompt is::

        <image>
        {question}
        Answer the question using a single word or phrase.

    which is the InternVL VQA evaluation prompt used in VLMEvalKit. When
    ``short_hint`` is False, the prompt is the bare ``<image>\\n{question}``.
    """
    dtype = next(model.parameters()).dtype
    pixel_values = load_pixel_values(image, input_size=448, max_num=max_tiles)
    pixel_values = pixel_values.to(device).to(dtype)

    generation_config = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=num_beams,
    )
    if short_hint:
        prompt = f"<image>\n{question}\n{SHORT_ANSWER_HINT}"
    else:
        prompt = f"<image>\n{question}"

    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)

    return response
