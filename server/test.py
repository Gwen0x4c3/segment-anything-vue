import os
import time

import torch
from PIL import Image
import numpy as np
import io
import base64
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from pycocotools import mask as mask_utils
import lzstring


def init():
    # your model path
    checkpoint = "checkpoints/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device='cuda')
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return predictor, mask_generator


predictor, mask_generator = init()

last_image = ""
last_logit = None

@torch.no_grad()
def process_image(body: dict):
    global last_image, last_logit
    print("start processing image", time.time())
    path = body["path"]
    is_first_segment = False
    # 看上次分割的图片是不是该图片
    if path != last_image: # 不是该图片，重新生成图像embedding
        pil_image = Image.open(path)
        np_image = np.array(pil_image)
        predictor.set_image(np_image)
        last_image = path
        is_first_segment = True
        print("第一次识别该图片，获取embedding中")
    # 获取mask
    clicks = body["clicks"]
    input_points = []
    input_labels = []
    for click in clicks:
        input_points.append([click["x"], click["y"]])
        input_labels.append(click["clickType"])
    print("input_points:{}, input_labels:{}".format(input_points, input_labels))
    input_points = np.array(input_points)
    input_labels = np.array(input_labels)
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        mask_input=last_logit[None, :, :] if not is_first_segment else None,
        multimask_output=is_first_segment # 第一次产生3个结果，选择最优的
    )
    # 设置mask_input，为下一次做准备
    best = np.argmax(scores)
    last_logit = logits[best, :, :]
    masks = masks[best, :, :]
    print("shape:{}".format(masks.shape))
    print(np.asfortranarray(masks).shape)
    source_mask = mask_utils.encode(np.asfortranarray(masks))["counts"].decode("utf-8")
    lzs = lzstring.LZString()
    encoded = lzs.compressToEncodedURIComponent(source_mask)
    print(len(encoded))
    print("process finished", time.time())
    return masks.shape, encoded

if __name__ == "__main__":
    for i in range(3):
        shape, encoded = process_image({
            "path": r"D:\project\code\segment-anything-main\test\wc.jpg",
            "clicks": [{"x": 100 + i * 100, "y": 100 + i * 100, "clickType": i % 2}]
        })