import os
import shutil
import time

from PIL import Image
import numpy as np
import io
import base64
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from pycocotools import mask as mask_utils
import lzstring


def init():
    # your model path
    # checkpoint = "checkpoints/sam_vit_b_01ec64.pth"
    # checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
    checkpoint = "checkpoints/sam_vit_l_0b3195.pth"
    model_type = "vit_l"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device='cuda')
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return predictor, mask_generator


predictor, mask_generator = init()

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/upload", StaticFiles(directory="upload"), name="upload")
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

last_image = ""
last_logit = None

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    print("上传图片", file.filename)
    file_path = os.path.join("upload", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return JSONResponse(content={"src": "http://localhost:8006/upload/" + file.filename, "path": os.path.abspath(file_path)})

@app.get("/img/{path}")
async def get_image(path: str):
    file_path = os.path.join("upload", path)
    return FileResponse(file_path)

@app.post("/segment")
def process_image(body: dict):
    global last_image, last_logit
    print("start processing image", time.time())
    path = body["path"]
    is_first_segment = False
    # 看上次分割的图片是不是该图片
    if path != last_image:  # 不是该图片，重新生成图像embedding
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
        multimask_output=is_first_segment  # 第一次产生3个结果，选择最优的
    )
    # 设置mask_input，为下一次做准备
    best = np.argmax(scores)
    last_logit = logits[best, :, :]
    masks = masks[best, :, :]
    # print(mask_utils.encode(np.asfortranarray(masks))["counts"])
    # numpy_array = np.frombuffer(mask_utils.encode(np.asfortranarray(masks))["counts"], dtype=np.uint8)
    # 打印numpy数组作为uint8array的格式
    # print("Uint8Array([" + ", ".join(map(str, numpy_array)) + "])")
    source_mask = mask_utils.encode(np.asfortranarray(masks))["counts"].decode("utf-8")
    # print(source_mask)
    lzs = lzstring.LZString()
    encoded = lzs.compressToEncodedURIComponent(source_mask)
    print("process finished", time.time())
    return {"shape": masks.shape, "mask": encoded}


@app.get("/everything")
def segment_everything(path: str):
    start_time = time.time()
    print("start segment_everything", start_time)
    pil_image = Image.open(path)
    np_image = np.array(pil_image)
    masks = mask_generator.generate(np_image)
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]), dtype=np.uint8)
    for idx, ann in enumerate(sorted_anns, 0):
        img[ann['segmentation']] = idx % 255 + 1  # color can only be in range [1, 255]
    # 压缩数组
    result = my_compress(img)
    end_time = time.time()
    print("finished segment_everything", end_time)
    print("time cost", end_time - start_time)
    return {"shape": img.shape, "mask": result}


@app.get('/automatic_masks')
def automatic_masks(path: str):
    pil_image = Image.open(path)
    np_image = np.array(pil_image)
    mask = mask_generator.generate(np_image)
    sorted_anns = sorted(mask, key=(lambda x: x['area']), reverse=True)
    lzs = lzstring.LZString()
    res = []
    for ann in sorted_anns:
        m = ann['segmentation']
        source_mask = mask_utils.encode(m)['counts'].decode("utf-8")
        encoded = lzs.compressToEncodedURIComponent(source_mask)
        r = {
            "encodedMask": encoded,
            "point_coord": ann['point_coords'][0],
        }
        res.append(r)
    return res


def my_compress(img):
    result = []
    last_pixel = img[0][0]
    count = 0
    for line in img:
        for pixel in line:
            if pixel == last_pixel:
                count += 1
            else:
                result.append(count)
                result.append(int(last_pixel))
                last_pixel = pixel
                count = 1
    result.append(count)
    result.append(int(last_pixel))
    return result
