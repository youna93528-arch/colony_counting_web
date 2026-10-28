import sys
import os
import io
import base64
import threading

import torch
import numpy as np
import cv2

from PIL import Image, ImageDraw
from flask import Flask, request, jsonify, render_template_string

# ⭐ 관련 라이브러리 정의 추가
# =========================================================
import json
from pycocotools import mask as mask_utils
# =========================================================
# ⭐ 종료


# 모델 추론에 사용될 가중치 파일이 있는 폴더
sys.path.append(
    r"C:\Users\KETI\Desktop\SAM3 미생물 검출\sam3"
)

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


bpe_path = r"C:\Users\KETI\Desktop\SAM3 미생물 검출\sam3\sam3\assets\bpe_simple_vocab_16e6.txt.gz"

checkpoint_path = r"C:\Users\KETI\Desktop\SAM3 미생물 검출\sam3\sam3\logs\checkpoints\checkpoint.pt"

# 추론할 이미지 경로
image_path = r"C:\Users\KETI\Desktop\SAM3 미생물 검출\1. test 원본 이미지 (어노테이션 포함)\train_1-1-C-a-1-1-2-_png.rf.1af23b6c2d9f2d1148bd983aaf3830ed.jpg"

# ⭐ 전역변수 및 경로 추가
# =========================================================
# GT 어노테이션 경로
gt_json_path = r"C:\Users\KETI\Desktop\SAM3 미생물 검출\1. test 원본 이미지 (어노테이션 포함)\_annotations.coco.json"
EVAL_IOU_THRESHOLD = 0.5
# =========================================================
# ⭐ 종료


PROMPT_MAP = {
    "C.a": "C.a",
    "S.a": "S.a",
    "E.c": "E.c",
    "P.g": "P.g",
    "B.c": "B.c",
    "P.a": "P.a",
    "A.b": "A.b",

    "S.a_TNTC": "S.a_TNTC",
    "E.c_TNTC": "E.c_TNTC",
    "P.g_TNTC": "P.g_TNTC",
    "P.a_TNTC": "P.a_TNTC",
    "B.c_TNTC": "B.c_TNTC",
    "C.a_TNTC": "C.a_TNTC",
    "A.b_TNTC": "A.b_TNTC",
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


def resize_long_side(image, target_size=1008):

    w, h = image.size

    long_side = max(w, h)

    if long_side == target_size:
        return image

    scale = target_size / long_side

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    return image.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )


def masks_to_numpy(masks):

    if masks is None:
        return []

    if torch.is_tensor(masks):
        masks = (
            masks
            .detach()
            .cpu()
            .numpy()
        )

    result = []

    for mask in masks:

        if mask.ndim == 3:
            mask = mask[0]

        mask = (
            mask > 0.5
        ).astype(np.uint8)

        result.append(mask)

    return result


def resize_mask_if_needed(
    mask,
    target_w,
    target_h
):

    if mask.shape == (
        target_h,
        target_w
    ):
        return mask.astype(bool)

    mask_img = Image.fromarray(
        (
            mask * 255
        ).astype(np.uint8)
    )

    mask_img = mask_img.resize(
        (
            target_w,
            target_h
        ),
        Image.Resampling.NEAREST
    )

    return (
        np.array(mask_img)
        > 0
    )


def draw_mask_with_contour(
    result,
    mask_bool,
    fill_color,
    contour_color,
    alpha=0.22,
    halo_width=5,
    contour_width=3
):

    mask_uint8 = (
        mask_bool.astype(np.uint8)
        * 255
    )

    # 내부는 반투명 색상
    result[mask_bool] = (
        result[mask_bool].astype(np.float32)
        * (1.0 - alpha)
        +
        np.array(
            fill_color,
            dtype=np.float32
        )
        * alpha
    ).astype(np.uint8)


    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # 바깥쪽 검정 halo
    # RGB numpy image라서 RGB 순서 사용
    cv2.drawContours(
        result,
        contours,
        -1,
        (0, 0, 0),
        halo_width,
        lineType=cv2.LINE_AA
    )


    # 안쪽 실제 색 contour
    cv2.drawContours(
        result,
        contours,
        -1,
        contour_color,
        contour_width,
        lineType=cv2.LINE_AA
    )


def create_overlay_image(
    base_image,
    text_masks=None,
    visual_masks=None,
    boxes=None
):

    if text_masks is None:
        text_masks = []

    if visual_masks is None:
        visual_masks = []

    if boxes is None:
        boxes = []


    result = np.array(
        base_image
    ).copy()


    height, width = (
        result.shape[:2]
    )


    # Text Prompt 결과
    # 붉은 반투명 fill
    # 검정 halo
    # 밝은 빨간 outline
    for mask in text_masks:

        mask_bool = resize_mask_if_needed(
            mask,
            width,
            height
        )


        draw_mask_with_contour(
            result=result,
            mask_bool=mask_bool,

            fill_color=(
                255,
                40,
                40
            ),

            contour_color=(
                255,
                20,
                20
            ),

            alpha=0.20,

            halo_width=6,

            contour_width=3
        )


    # Visual / Box Prompt 결과
    # 초록 반투명 fill
    # 검정 halo
    # 밝은 초록 outline
    for mask in visual_masks:

        mask_bool = resize_mask_if_needed(
            mask,
            width,
            height
        )


        draw_mask_with_contour(
            result=result,
            mask_bool=mask_bool,

            fill_color=(
                30,
                255,
                80
            ),

            contour_color=(
                0,
                255,
                60
            ),

            alpha=0.20,

            halo_width=6,

            contour_width=3
        )


    pil_result = Image.fromarray(
        result
    )


    draw = ImageDraw.Draw(
        pil_result
    )


    # 사용자가 선택한 Bounding Box
    for box in boxes:

        x1, y1, x2, y2 = box


        # 검정 halo
        draw.rectangle(
            [
                int(x1) - 2,
                int(y1) - 2,
                int(x2) + 2,
                int(y2) + 2
            ],
            outline=(
                0,
                0,
                0
            ),
            width=7
        )


        # 노란 box
        draw.rectangle(
            [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ],
            outline=(
                255,
                255,
                0
            ),
            width=4
        )


    return pil_result


def pil_to_base64(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/png;base64,"
        + encoded
    )
    
# ⭐ 여기부터 추가된 함수 블록
# =========================================================
# IoU 구하는 함수
def mask_iou(
    mask_a,
    mask_b
):
    mask_a = mask_a.astype(
        bool
    )

    mask_b = mask_b.astype(
        bool
    )

    if mask_a.shape != mask_b.shape:

        mask_a = resize_mask_if_needed(
            mask_a,
            target_w=mask_b.shape[1],
            target_h=mask_b.shape[0]
        )

    intersection = np.logical_and(
        mask_a,
        mask_b
    ).sum()

    union = np.logical_or(
        mask_a,
        mask_b
    ).sum()

    if union == 0:
        return 0.0

    return (
        float(intersection)
        /
        float(union)
    )
    
    
# Text - Box 중복 검출 제거 함수 
def remove_visual_duplicates_with_text(
    visual_masks,
    text_masks,
    iou_threshold=0.5
):
    unique_visual_masks = []

    for visual_idx, visual_mask in enumerate(
        visual_masks
    ):
        max_iou = 0.0

        for text_idx, text_mask in enumerate(
            text_masks
        ):
            # 혹시 크기가 다르면 visual mask를
            # text mask 크기에 맞춤
            if visual_mask.shape != text_mask.shape:
                visual_mask_resized = (
                    resize_mask_if_needed(
                        visual_mask,
                        target_w=text_mask.shape[1],
                        target_h=text_mask.shape[0]
                    )
                )
            else:
                visual_mask_resized = (
                    visual_mask.astype(bool)
                )

            text_mask_bool = (
                text_mask.astype(bool)
            )

            iou = mask_iou(
                visual_mask_resized,
                text_mask_bool
            )

            max_iou = max(
                max_iou,
                iou
            )

        # Text Prompt 객체와 많이 겹치지 않는 경우만 유지
        if max_iou < iou_threshold:
            unique_visual_masks.append(
                visual_mask
            )

            print(
                f"[KEEP] Visual {visual_idx} "
                f"| max IoU={max_iou:.4f}"
            )

        else:
            print(
                f"[REMOVE] Visual {visual_idx} "
                f"| max IoU={max_iou:.4f}"
            )

    return unique_visual_masks


# GT랑 추론된 값이랑 비교해서 Precision / Recall / F1-Score 구하는 함수
def load_coco_gt_masks(
    json_path,
    current_image_path,
    target_w,
    target_h
):
    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:
        coco_data = json.load(f)

    current_filename = os.path.basename(
        current_image_path
    )

    print()
    print("=" * 60)
    print("[GT LOAD]")
    print("Current image:", current_filename)


    matched_image = None

    for image_info in coco_data["images"]:

        coco_filename = os.path.basename(
            image_info["file_name"]
        )

        if coco_filename == current_filename:
            matched_image = image_info
            break

    if matched_image is None:
        raise ValueError(
            f"COCO JSON에서 현재 이미지를 찾을 수 없습니다:\n"
            f"{current_filename}"
        )

    image_id = matched_image["id"]

    original_w = matched_image["width"]
    original_h = matched_image["height"]

    print("COCO image_id:", image_id)
    print(
        "COCO original size:",
        original_w,
        original_h
    )


    annotations = [
        ann
        for ann in coco_data["annotations"]
        if ann["image_id"] == image_id
    ]

    gt_masks = []

    for ann in annotations:

        segmentation = ann.get(
            "segmentation"
        )

        if segmentation is None:
            continue


        if isinstance(segmentation, list):

            rles = mask_utils.frPyObjects(
                segmentation,
                original_h,
                original_w
            )

            rle = mask_utils.merge(
                rles
            )


        elif isinstance(segmentation, dict):

            rle = segmentation.copy()

            if isinstance(
                rle.get("counts"),
                list
            ):
                rle = mask_utils.frPyObjects(
                    rle,
                    original_h,
                    original_w
                )

        else:
            continue

        gt_mask = mask_utils.decode(
            rle
        )

        # 간혹 H x W x N 형태
        if gt_mask.ndim == 3:
            gt_mask = (
                np.any(
                    gt_mask,
                    axis=2
                )
            )

        gt_mask = (
            gt_mask > 0
        ).astype(
            np.uint8
        )

        # 모델 입력 이미지 크기가 1008 resize이므로
        # GT도 동일 크기로 resize
        gt_mask = resize_mask_if_needed(
            gt_mask,
            target_w=target_w,
            target_h=target_h
        )

        gt_masks.append(
            gt_mask.astype(
                np.uint8
            )
        )

    print(
        "GT objects:",
        len(gt_masks)
    )

    return gt_masks


# Prediction ↔ GT 1:1 Matching
def evaluate_masks(
    pred_masks,
    gt_masks,
    iou_threshold=0.5,
    title="Evaluation"
):
    num_pred = len(
        pred_masks
    )

    num_gt = len(
        gt_masks
    )

    # 모든 Prediction-GT IoU 계산
    candidates = []

    for pred_idx, pred_mask in enumerate(
        pred_masks
    ):

        for gt_idx, gt_mask in enumerate(
            gt_masks
        ):

            iou = mask_iou(
                pred_mask,
                gt_mask
            )

            if iou >= iou_threshold:

                candidates.append(
                    (
                        iou,
                        pred_idx,
                        gt_idx
                    )
                )

    # 높은 IoU부터 매칭
    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    matched_pred = set()
    matched_gt = set()

    matches = []

    for (
        iou,
        pred_idx,
        gt_idx
    ) in candidates:

        if pred_idx in matched_pred:
            continue

        if gt_idx in matched_gt:
            continue

        matched_pred.add(
            pred_idx
        )

        matched_gt.add(
            gt_idx
        )

        matches.append(
            (
                pred_idx,
                gt_idx,
                iou
            )
        )

    # TP / FP / FN
    tp = len(
        matches
    )

    fp = (
        num_pred
        -
        tp
    )

    fn = (
        num_gt
        -
        tp
    )

    # Precision / Recall / F1
    if tp + fp > 0:

        precision = (
            tp
            /
            (tp + fp)
        )

    else:
        precision = 0.0

    if tp + fn > 0:

        recall = (
            tp
            /
            (tp + fn)
        )

    else:
        recall = 0.0

    if precision + recall > 0:

        f1 = (
            2
            *
            precision
            *
            recall
            /
            (
                precision
                +
                recall
            )
        )

    else:
        f1 = 0.0


    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        "IoU threshold:",
        iou_threshold
    )

    print(
        "Prediction objects:",
        num_pred
    )

    print(
        "GT objects        :",
        num_gt
    )

    print()

    print(
        f"TP "
        f"(실제 객체를 제대로 맞춘 것): "
        f"{tp}"
    )

    print(
        f"FP "
        f"(실제 없는데 있다고 잡은 것): "
        f"{fp}"
    )

    print(
        f"FN "
        f"(실제 있는데 못 잡은 것): "
        f"{fn}"
    )

    print()

    print(
        f"Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"F1-score  : "
        f"{f1:.4f}"
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,

        "precision": precision,
        "recall": recall,
        "f1": f1,

        "matches": matches,

        "num_pred": num_pred,
        "num_gt": num_gt
    }

# =========================================================
# ⭐ 종료


print()
print("Loading SAM3 model...")
print()


model = build_sam3_image_model(
    bpe_path=bpe_path
)


if os.path.exists(
    checkpoint_path
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    trained_weights = (
        checkpoint["model"]
    )

    model_dict = (
        model.state_dict()
    )

    updated_weights = {
        k: v
        for k, v
        in trained_weights.items()
        if (
            k in model_dict
            and
            v.shape
            ==
            model_dict[k].shape
        )
    }


    model.load_state_dict(
        updated_weights,
        strict=False
    )


    print(
        f"Checkpoint loaded "
        f"(Epoch: "
        f"{checkpoint.get('epoch', 'unknown')})"
    )

else:

    print(
        "WARNING: Checkpoint not found"
    )


model.to(device)

model.eval()


processor = Sam3Processor(
    model,
    confidence_threshold=0.5,
    device=device
)


with Image.open(
    image_path
) as img:

    original_image = (
        img.convert("RGB")
    )


original_image = resize_long_side(
    original_image,
    target_size=1008
)


IMAGE_W, IMAGE_H = (
    original_image.size
)

# ⭐ GT 마스크 로드 함수 추가
# GT 마스크 로드
# =========================================================
gt_masks = load_coco_gt_masks(
    json_path=gt_json_path,
    current_image_path=image_path,
    target_w=IMAGE_W,
    target_h=IMAGE_H
)
# =========================================================
# ⭐ 종료

print(
    "Image size:",
    IMAGE_W,
    IMAGE_H
)


app = Flask(
    __name__
)


state_lock = (
    threading.Lock()
)


text_state = None

visual_state = None


text_masks = []

visual_masks = []


current_prompt = ""

current_boxes = []


original_base64 = (
    pil_to_base64(
        original_image
    )
)


HTML = """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>
SAM3 Interactive Prompt
</title>


<style>

body {

    background: #181818;

    color: white;

    font-family: Arial, sans-serif;

    margin: 0;

    padding: 0;
}


.container {

    width: 1100px;

    margin: 20px auto;
}


.title {

    font-size: 28px;

    font-weight: bold;

    margin-bottom: 10px;
}


.description {

    color: #bbbbbb;

    margin-bottom: 20px;

    line-height: 1.6;
}


.controls {

    display: flex;

    gap: 10px;

    align-items: center;

    margin-bottom: 15px;
}


input {

    width: 260px;

    padding: 10px;

    font-size: 18px;

    border-radius: 6px;

    border: 1px solid #666;

    background: #282828;

    color: white;
}


input:focus {

    outline: 2px solid #4c9aff;

    border-color: #4c9aff;
}


button {

    padding: 10px 18px;

    font-size: 16px;

    cursor: pointer;

    border-radius: 6px;

    border: none;
}


#runText {

    background: #1976d2;

    color: white;
}


#runText:disabled {

    opacity: 0.5;

    cursor: not-allowed;
}


#reset {

    background: #555555;

    color: white;
}


#status {

    background: #242424;

    padding: 12px;

    border-radius: 6px;

    margin-bottom: 12px;

    min-height: 24px;

    font-size: 17px;

    font-weight: bold;
}


.legend {

    display: flex;

    gap: 25px;

    margin-bottom: 12px;
}


.legend-item {

    display: flex;

    align-items: center;

    gap: 7px;
}


.color-box {

    width: 20px;

    height: 20px;

    border: 2px solid black;
}


.red {

    background: rgb(
        255,
        30,
        30
    );
}


.green {

    background: rgb(
        0,
        255,
        60
    );
}


.yellow {

    background: rgb(
        255,
        255,
        0
    );
}


.canvas-wrapper {

    position: relative;

    display: inline-block;

    border: 3px solid #666;

    background: black;
}


canvas {

    display: block;

    max-width: 100%;

    cursor: crosshair;
}


.help {

    margin-top: 15px;

    line-height: 1.8;

    color: #cccccc;
}


.count-box {

    margin-top: 10px;

    padding: 12px;

    background: #242424;

    border-radius: 6px;

    display: flex;

    gap: 35px;

    font-size: 17px;

    font-weight: bold;
}


</style>

</head>


<body>


<div class="container">


<div class="title">

SAM3 Interactive Text + Visual Prompt

</div>


<div class="description">

Text Prompt로 1차 검출 후,
놓친 객체에 Bounding Box를 그리면
해당 객체를 Visual Prompt로 사용하여
비슷한 객체를 추가 검출합니다.

</div>


<div class="controls">


<label>

Text Prompt:

</label>


<input
    id="prompt"
    type="text"
    maxlength="1"
    placeholder="Prompt 입력 (예: A)"
    autocomplete="off"
/>


<button id="runText">

Text Inference

</button>


<button id="reset">

Reset

</button>


</div>


<div class="legend">


<div class="legend-item">

<div class="color-box red"></div>

Text Prompt Detection

</div>


<div class="legend-item">

<div class="color-box green"></div>

Visual / Box Prompt Detection

</div>


<div class="legend-item">

<div class="color-box yellow"></div>

Selected Bounding Box

</div>


</div>


<div id="status">

Text Prompt를 입력하세요.

</div>


<div class="canvas-wrapper">

<canvas
    id="canvas"
    width="{{ width }}"
    height="{{ height }}"
>
</canvas>

</div>


<div class="count-box">

<div>

Text:
<span id="textCount">
0
</span>

</div>


<div>

Visual:
<span id="visualCount">
0
</span>

</div>


<div>

Boxes:
<span id="boxCount">
0
</span>

</div>


</div>


<div class="help">

1. Prompt 입력창에 <b>A</b> 입력<br>

2. Enter 또는 Text Inference 클릭<br>

3. 빨간 외곽선으로 Text Prompt 검출 결과 확인<br>

4. 놓친 객체를 마우스로 드래그<br>

5. 마우스를 놓으면 Visual Box Prompt 추론 자동 실행<br>

6. 추가 검출 결과는 초록 외곽선으로 표시<br>

7. 다른 객체에 Box를 계속 추가할 수 있음

</div>


</div>


<script>


const canvas =
    document.getElementById(
        "canvas"
    );


const ctx =
    canvas.getContext(
        "2d"
    );


const promptInput =
    document.getElementById(
        "prompt"
    );


const runTextButton =
    document.getElementById(
        "runText"
    );


const resetButton =
    document.getElementById(
        "reset"
    );


const status =
    document.getElementById(
        "status"
    );


const textCount =
    document.getElementById(
        "textCount"
    );


const visualCount =
    document.getElementById(
        "visualCount"
    );


const boxCount =
    document.getElementById(
        "boxCount"
    );


let currentImage =
    new Image();


let drawing =
    false;


let startX =
    0;


let startY =
    0;


let currentX =
    0;


let currentY =
    0;


let textInferenceDone =
    false;


function loadImage(src) {

    currentImage =
        new Image();


    currentImage.onload =
        function() {

            ctx.clearRect(
                0,
                0,
                canvas.width,
                canvas.height
            );


            ctx.drawImage(
                currentImage,
                0,
                0,
                canvas.width,
                canvas.height
            );
        };


    currentImage.src =
        src;
}


loadImage(
    "{{ image_data }}"
);


function getCanvasPosition(event) {

    const rect =
        canvas.getBoundingClientRect();


    const scaleX =
        canvas.width
        /
        rect.width;


    const scaleY =
        canvas.height
        /
        rect.height;


    return {

        x:
            (
                event.clientX
                -
                rect.left
            )
            *
            scaleX,

        y:
            (
                event.clientY
                -
                rect.top
            )
            *
            scaleY
    };
}


async function runTextInference() {

    let prompt =
        promptInput
        .value
        .trim()
        .toUpperCase();


    promptInput.value =
        prompt;


    if (!prompt) {

        status.innerText =
            "Text Prompt를 입력하세요.";

        return;
    }


    status.innerText =
        `Prompt "${prompt}" 추론 중...`;


    runTextButton.disabled =
        true;


    try {


        const response =
            await fetch(
                "/text_infer",
                {

                    method:
                        "POST",

                    headers: {

                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({

                            prompt:
                                prompt
                        })
                }
            );


        const data =
            await response.json();


        if (!data.success) {

            status.innerText =
                data.error;

            return;
        }


        loadImage(
            data.image
        );


        textCount.innerText =
            data.text_count;


        visualCount.innerText =
            0;


        boxCount.innerText =
            0;


        status.innerText =
            `Prompt "${prompt}" → ${data.text_count}개 검출 | 놓친 객체를 Box로 지정하세요.`;


        textInferenceDone =
            true;


    } catch (error) {


        status.innerText =
            "ERROR: "
            + error;


    } finally {


        runTextButton.disabled =
            false;
    }
}


runTextButton.addEventListener(
    "click",
    runTextInference
);


promptInput.addEventListener(
    "input",
    function() {

        this.value =
            this.value
            .toUpperCase();
    }
);


promptInput.addEventListener(
    "keydown",
    function(event) {

        if (
            event.key
            ===
            "Enter"
        ) {

            runTextInference();
        }
    }
);


canvas.addEventListener(
    "mousedown",
    function(event) {


        if (
            !textInferenceDone
        ) {

            status.innerText =
                "먼저 Text Prompt 추론을 실행하세요.";

            return;
        }


        const pos =
            getCanvasPosition(
                event
            );


        startX =
            pos.x;


        startY =
            pos.y;


        currentX =
            pos.x;


        currentY =
            pos.y;


        drawing =
            true;
    }
);


canvas.addEventListener(
    "mousemove",
    function(event) {


        if (!drawing) {

            return;
        }


        const pos =
            getCanvasPosition(
                event
            );


        currentX =
            pos.x;


        currentY =
            pos.y;


        ctx.clearRect(
            0,
            0,
            canvas.width,
            canvas.height
        );


        ctx.drawImage(
            currentImage,
            0,
            0,
            canvas.width,
            canvas.height
        );


        // 검정 halo
        ctx.strokeStyle =
            "black";


        ctx.lineWidth =
            8;


        ctx.strokeRect(
            startX,
            startY,
            currentX - startX,
            currentY - startY
        );


        // 노란 box
        ctx.strokeStyle =
            "yellow";


        ctx.lineWidth =
            4;


        ctx.strokeRect(
            startX,
            startY,
            currentX - startX,
            currentY - startY
        );
    }
);


canvas.addEventListener(
    "mouseup",
    async function(event) {


        if (!drawing) {

            return;
        }


        drawing =
            false;


        const pos =
            getCanvasPosition(
                event
            );


        const endX =
            pos.x;


        const endY =
            pos.y;


        const x1 =
            Math.min(
                startX,
                endX
            );


        const y1 =
            Math.min(
                startY,
                endY
            );


        const x2 =
            Math.max(
                startX,
                endX
            );


        const y2 =
            Math.max(
                startY,
                endY
            );


        if (
            x2 - x1 < 5
            ||
            y2 - y1 < 5
        ) {


            loadImage(
                currentImage.src
            );


            status.innerText =
                "Bounding Box가 너무 작습니다.";


            return;
        }


        status.innerText =
            "Visual Box Prompt 추론 중...";


        try {


            const response =
                await fetch(
                    "/box_infer",
                    {

                        method:
                            "POST",

                        headers: {

                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify({

                                x1:
                                    x1,

                                y1:
                                    y1,

                                x2:
                                    x2,

                                y2:
                                    y2
                            })
                    }
                );


            const data =
                await response.json();


            if (!data.success) {


                status.innerText =
                    data.error;


                loadImage(
                    currentImage.src
                );


                return;
            }


            loadImage(
                data.image
            );


            textCount.innerText =
                data.text_count;


            visualCount.innerText =
                data.visual_count;


            boxCount.innerText =
                data.box_count;


            status.innerText =
                `Visual Prompt → ${data.visual_count}개 검출 | Text ${data.text_count} + Visual ${data.visual_count}`;


        } catch (error) {


            status.innerText =
                "ERROR: "
                + error;


            loadImage(
                currentImage.src
            );
        }
    }
);


canvas.addEventListener(
    "mouseleave",
    function() {


        if (drawing) {


            drawing =
                false;


            loadImage(
                currentImage.src
            );
        }
    }
);


resetButton.addEventListener(
    "click",
    async function() {


        const response =
            await fetch(
                "/reset",
                {

                    method:
                        "POST"
                }
            );


        const data =
            await response.json();


        loadImage(
            data.image
        );


        promptInput.value =
            "";


        textCount.innerText =
            0;


        visualCount.innerText =
            0;


        boxCount.innerText =
            0;


        textInferenceDone =
            false;


        status.innerText =
            "Reset 완료. Text Prompt를 입력하세요.";
    }
);


</script>


</body>

</html>
"""


@app.route("/")
def index():

    return render_template_string(
        HTML,

        image_data=
            original_base64,

        width=
            IMAGE_W,

        height=
            IMAGE_H
    )


@app.route(
    "/text_infer",
    methods=["POST"]
)

# ⭐ text_infer 함수 수정
# ===================================
def text_infer():

    global text_state
    global visual_state

    global text_masks
    global visual_masks

    global current_prompt
    global current_boxes
    global text_metrics 
    
    data = request.get_json()


    user_prompt = (
        data
        .get(
            "prompt",
            ""
        )
        .strip()
        .upper()
    )


    if not user_prompt:

        return jsonify(
            success=False,
            error="Text Prompt가 비어 있습니다."
        )


    # A -> P.a
    # GUI나 터미널에는 실제 이름 출력하지 않음
    model_prompt = (
        PROMPT_MAP.get(
            user_prompt
        )
    )


    if model_prompt is None:

        return jsonify(
            success=False,
            error=(
                "지원하지 않는 Prompt입니다. "
                "A를 입력하세요."
            )
        )


    try:


        with state_lock:


            print()
            print(
                "=" * 60
            )

            print(
                "[TEXT PROMPT]"
            )

            print(
                "Blind Prompt:",
                user_prompt
            )


            with torch.no_grad():


                text_state = (
                    processor.set_image(
                        original_image
                    )
                )


                processor.reset_all_prompts(
                    text_state
                )


                # 여기에서만 내부적으로 P.a 전달
                text_state = (
                    processor.set_text_prompt(
                        state=
                            text_state,

                        prompt=
                            model_prompt
                    )
                )
                
            # Text Prompt 결과 가져오기
            masks = (
                text_state.get(
                    "masks"
                )
            )

            text_masks = (
                masks_to_numpy(
                    masks
                )
            )


            # Text Prompt 단독 성능 평가
            text_metrics = evaluate_masks(
                pred_masks=text_masks,
                gt_masks=gt_masks,
                iou_threshold=EVAL_IOU_THRESHOLD,
                title="[TEXT PROMPT ONLY]"
            )


            # Box Prompt 관련 상태 초기화
            visual_state = None
            visual_masks = []
            current_boxes = []

            current_prompt = user_prompt
    
            visual_masks = []

            current_boxes = []

            # 실제 클래스명이 아닌 A 저장
            current_prompt = (
                user_prompt
            )


            print(
                "Text detections:",
                len(text_masks)
            )


            result_image = (
                create_overlay_image(

                    base_image=
                        original_image,

                    text_masks=
                        text_masks,

                    visual_masks=
                        [],

                    boxes=
                        []
                )
            )


            encoded = (
                pil_to_base64(
                    result_image
                )
            )


        return jsonify(

            success=True,

            image=
                encoded,

            text_count=
                len(text_masks),

            display_prompt=
                user_prompt
        )


    except Exception as e:


        print()
        print(
            "TEXT ERROR:"
        )

        print(
            repr(e)
        )


        return jsonify(
            success=False,
            error=str(e)
        )


@app.route(
    "/box_infer",
    methods=["POST"]
)
# =============================================
# ⭐ 종료

# ⭐ box_infer 수정 시작
# =============================================
def box_infer():

    global text_state
    global visual_state

    global text_masks
    global visual_masks

    global current_boxes
    global text_metrics


    if text_state is None:

        return jsonify(
            success=False,
            error=(
                "먼저 Text Prompt 추론을 "
                "실행하세요."
            )
        )


    data = request.get_json()


    try:


        x1 = float(
            data["x1"]
        )

        y1 = float(
            data["y1"]
        )

        x2 = float(
            data["x2"]
        )

        y2 = float(
            data["y2"]
        )


        x1 = max(
            0,
            min(
                x1,
                IMAGE_W - 1
            )
        )


        y1 = max(
            0,
            min(
                y1,
                IMAGE_H - 1
            )
        )


        x2 = max(
            0,
            min(
                x2,
                IMAGE_W - 1
            )
        )


        y2 = max(
            0,
            min(
                y2,
                IMAGE_H - 1
            )
        )


        box_width = (
            x2 - x1
        )


        box_height = (
            y2 - y1
        )


        if (
            box_width <= 1
            or
            box_height <= 1
        ):


            return jsonify(
                success=False,
                error="Box 크기가 너무 작습니다."
            )


        center_x = (
            x1 + x2
        ) / 2.0


        center_y = (
            y1 + y2
        ) / 2.0


        normalized_box = [

            center_x
            / IMAGE_W,

            center_y
            / IMAGE_H,

            box_width
            / IMAGE_W,

            box_height
            / IMAGE_H
        ]


        print()
        print(
            "=" * 60
        )

        print(
            "[VISUAL BOX PROMPT]"
        )


        print(
            "Pixel box:",
            [
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2)
            ]
        )


        print(
            "Normalized [cx, cy, w, h]:",
            [
                round(v, 5)
                for v
                in normalized_box
            ]
        )


        with state_lock:


            with torch.no_grad():


                if visual_state is None:


                    print(
                        "Creating new VISUAL state"
                    )


                    visual_state = (
                        processor.set_image(
                            original_image
                        )
                    )


                    processor.reset_all_prompts(
                        visual_state
                    )


                visual_state = (
                    processor.add_geometric_prompt(
                        box=
                            normalized_box,

                        label=
                            True,

                        state=
                            visual_state
                    )
                )

            # Box Prompt 원본 결과
            masks = (
                visual_state.get(
                    "masks"
                )
            )

            raw_visual_masks = (
                masks_to_numpy(
                    masks
                )
            )

            print(
                "Raw Visual detections:",
                len(raw_visual_masks)
            )


            # Text Prompt 결과와 중복되는 Visual 객체 제거
            # Mask IoU >= 0.5 → 중복 제거
            visual_masks = (
                remove_visual_duplicates_with_text(
                    visual_masks=raw_visual_masks,
                    text_masks=text_masks,
                    iou_threshold=0.5
                )
            )

            print(
                "Unique Visual detections:",
                len(visual_masks)
            )


            # Text + Box 최종 객체
            final_masks = (
                list(text_masks)
                +
                list(visual_masks)
            )


            # Text + Box 최종 성능 평가
            final_metrics = evaluate_masks(
                pred_masks=final_masks,
                gt_masks=gt_masks,
                iou_threshold=EVAL_IOU_THRESHOLD,
                title="[TEXT + BOX FINAL]"
            )


            # Text Only vs Text + Box 성능 비교
            print()
            print("=" * 70)
            print("TEXT ONLY vs TEXT + BOX")
            print("=" * 70)

            print(
                f"{'Metric':<15}"
                f"{'Text Only':>15}"
                f"{'Text + Box':>15}"
                f"{'Difference':>15}"
            )

            print("-" * 60)

            print(
                f"{'TP':<15}"
                f"{text_metrics['tp']:>15}"
                f"{final_metrics['tp']:>15}"
                f"{final_metrics['tp'] - text_metrics['tp']:>+15}"
            )

            print(
                f"{'FP':<15}"
                f"{text_metrics['fp']:>15}"
                f"{final_metrics['fp']:>15}"
                f"{final_metrics['fp'] - text_metrics['fp']:>+15}"
            )

            print(
                f"{'FN':<15}"
                f"{text_metrics['fn']:>15}"
                f"{final_metrics['fn']:>15}"
                f"{final_metrics['fn'] - text_metrics['fn']:>+15}"
            )

            print(
                f"{'Precision':<15}"
                f"{text_metrics['precision']:>15.4f}"
                f"{final_metrics['precision']:>15.4f}"
                f"{final_metrics['precision'] - text_metrics['precision']:>+15.4f}"
            )

            print(
                f"{'Recall':<15}"
                f"{text_metrics['recall']:>15.4f}"
                f"{final_metrics['recall']:>15.4f}"
                f"{final_metrics['recall'] - text_metrics['recall']:>+15.4f}"
            )

            print(
                f"{'F1-score':<15}"
                f"{text_metrics['f1']:>15.4f}"
                f"{final_metrics['f1']:>15.4f}"
                f"{final_metrics['f1'] - text_metrics['f1']:>+15.4f}"
            )

            print()
            print(
                "Box Prompt로 추가된 고유 객체:",
                len(visual_masks)
            )


            current_boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            )


            print(
                "Text detections:",
                len(text_masks)
            )


            print(
                "Visual detections:",
                len(visual_masks)
            )


            print(
                "Boxes:",
                len(current_boxes)
            )


            result_image = (
                create_overlay_image(

                    base_image=
                        original_image,

                    text_masks=
                        text_masks,

                    visual_masks=
                        visual_masks,

                    boxes=
                        current_boxes
                )
            )


            encoded = (
                pil_to_base64(
                    result_image
                )
            )


        return jsonify(

            success=True,

            image=
                encoded,

            text_count=
                len(text_masks),

            visual_count=
                len(visual_masks),

            box_count=
                len(current_boxes)
        )


    except Exception as e:


        print()
        print(
            "BOX ERROR:"
        )

        print(
            repr(e)
        )


        return jsonify(
            success=False,
            error=str(e)
        )


@app.route(
    "/reset",
    methods=["POST"]
)
# =============================================
# ⭐ 종료

# ⭐ text_metrics 전역변수만 추가 되었습니다!
# =============================================
def reset():

    global text_state
    global visual_state

    global text_masks
    global visual_masks

    global current_prompt
    global current_boxes
    
    # ⭐ 전역 변수 추가
    global text_metrics
    
    # =============================================
    # ⭐ 종료



    with state_lock:


        text_state = None
        text_metrics = None
        visual_state = None


        text_masks = []

        visual_masks = []


        current_prompt = ""

        current_boxes = []


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    print()
    print(
        "[RESET]"
    )


    return jsonify(

        success=True,

        image=
            original_base64
    )


if __name__ == "__main__":

    print()
    print(
        "=" * 60
    )

    print(
        "SAM3 Interactive GUI"
    )

    print(
        "=" * 60
    )

    print()

    print(
        "Server:"
    )

    print(
        "http://0.0.0.0:7860"
    )

    print()

    print(
        "1. Blind Prompt A 입력"
    )

    print(
        "2. Text Inference"
    )

    print(
        "3. 놓친 객체에 Bounding Box"
    )

    print(
        "4. Visual Prompt 자동 추론"
    )

    print()


    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False,
        threaded=True
    )