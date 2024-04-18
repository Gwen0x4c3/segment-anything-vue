import os
import cv2
import time
import clip
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    gray_image = Image.new("RGB", image.size, (128, 128, 128))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    gray_image.paste(segmented_image, mask=transparency_mask_image)
    return gray_image


@torch.no_grad()
def image_text_match(cropped_objects, text_query):
    preprocessed_images = [preprocess(image).to(device) for image in cropped_objects]
    tokenized_text = clip.tokenize([text_query]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100. * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)


def get_id_photo_output(image, text):
    """
    Get the special size and background photo.

    Args:
        img(numpy:ndarray): The image array.
        size(str): The size user specified.
        bg(str): The background color user specified.
        download_size(str): The size for image saving.

    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    print("masks num: {}".format(len(masks)))

    cropped_objects = []
    image_pil = Image.fromarray(image)
    for mask in masks:
        bbox = [mask["bbox"][0], mask["bbox"][1], mask["bbox"][0] + mask["bbox"][2],
                mask["bbox"][1] + mask["bbox"][3]]
        cropped_objects.append(segment_image(image_pil, mask["segmentation"]).crop(bbox))

    scores = image_text_match(cropped_objects, str(text))
    text_matching_masks = []
    for idx, score in enumerate(scores):
        if score < 0.05:
            continue
        text_matching_mask = Image.fromarray(masks[idx]["segmentation"].astype('uint8') * 255)
        text_matching_masks.append(text_matching_mask)

    alpha_image = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
    alpha_color = (255, 0, 0, 180)

    draw = ImageDraw.Draw(alpha_image)
    for text_matching_mask in text_matching_masks:
        draw.bitmap((0, 0), text_matching_mask, fill=alpha_color)

    result_image = Image.alpha_composite(image_pil.convert('RGBA'), alpha_image)
    print(type(result_image), result_image.size)
    return result_image


if __name__ == "__main__":
    print("=== Initializing models...")
    model = "vit_h"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_generator = SamAutomaticMaskGenerator(
        sam_model_registry[model](checkpoint="../checkpoints/sam_vit_h_4b8939.pth").to(device))
    model, preprocess = clip.load("ViT-B/32", device=device, download_root='../checkpoints')
    print("=== Models have been loaded.")
    print("=== Processing image - {}.".format(time.time()))
    img = cv2.imread('../test/wyc.jpg')
    result_img = get_id_photo_output(img, 'dolls')
    print("=== Done - {}.".format(time.time()))
    plt.imshow(result_img)
    plt.show()