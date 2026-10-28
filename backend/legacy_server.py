from pathlib import Path
from services.backend_guard import acquire_backend_guard

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 7860
_backend_guard = acquire_backend_guard(Path(__file__).resolve().parents[1], host=SERVER_HOST, port=SERVER_PORT)

import io
import base64

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from flask import Flask, jsonify, render_template_string, request

from api.session_routes import create_session_blueprint
from services.runtime_observability import install_runtime_observability
from security.file_validation import MAX_UPLOAD_BYTES
from services.session_manager import SessionInferenceManager
from services.sam3_service import (
    InvalidBoxError,
    PromptRequiredError,
    Sam3InferenceService,
    clear_cuda_cache,
)


# ============================================================
# 2. 환경 설정
# ============================================================


# ============================================================
# 4. 이미지 처리 함수
# ============================================================


def pil_to_base64(image):
    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


def resize_mask_if_needed(mask, target_width, target_height):
    mask = np.asarray(mask)

    if mask.shape == (target_height, target_width):
        return mask.astype(bool)

    mask_image = Image.fromarray(
        (mask.astype(np.uint8) * 255)
    )

    mask_image = mask_image.resize(
        (target_width, target_height),
        Image.Resampling.NEAREST
    )

    return np.asarray(mask_image) > 0


def draw_mask_with_contour(
    result,
    mask_bool,
    fill_color,
    contour_color,
    alpha=0.2,
    halo_width=6,
    contour_width=3
):
    mask_uint8 = mask_bool.astype(np.uint8) * 255

    # 마스크 내부 반투명 색상
    result[mask_bool] = (
        result[mask_bool].astype(np.float32) * (1.0 - alpha)
        + np.asarray(fill_color, dtype=np.float32) * alpha
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 외곽선 검은색 테두리
    cv2.drawContours(
        result,
        contours,
        -1,
        (0, 0, 0),
        halo_width,
        lineType=cv2.LINE_AA
    )

    # 실제 색상 외곽선
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
    text_masks = text_masks or []
    visual_masks = visual_masks or []
    boxes = boxes or []

    result = np.asarray(base_image).copy()
    height, width = result.shape[:2]

    # Text Prompt 결과: 빨간색
    for mask in text_masks:
        mask_bool = resize_mask_if_needed(
            mask,
            width,
            height
        )

        draw_mask_with_contour(
            result=result,
            mask_bool=mask_bool,
            fill_color=(255, 40, 40),
            contour_color=(255, 20, 20),
            alpha=0.20,
            halo_width=6,
            contour_width=3
        )

    # Visual/Box Prompt 결과: 초록색
    for mask in visual_masks:
        mask_bool = resize_mask_if_needed(
            mask,
            width,
            height
        )

        draw_mask_with_contour(
            result=result,
            mask_bool=mask_bool,
            fill_color=(30, 255, 80),
            contour_color=(0, 255, 60),
            alpha=0.20,
            halo_width=6,
            contour_width=3
        )

    result_image = Image.fromarray(result)
    draw = ImageDraw.Draw(result_image)

    # 사용자가 선택한 Bounding Box: 노란색
    for box in boxes:
        x1, y1, x2, y2 = box

        # 검은색 바깥 테두리
        draw.rectangle(
            [
                int(x1) - 2,
                int(y1) - 2,
                int(x2) + 2,
                int(y2) + 2
            ],
            outline=(0, 0, 0),
            width=7
        )

        # 노란색 박스
        draw.rectangle(
            [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ],
            outline=(255, 255, 0),
            width=4
        )

    return result_image


# ============================================================
# 6. 프로그램 초기화
# ============================================================

sam3_service = Sam3InferenceService()
original_image = sam3_service.original_image
IMAGE_WIDTH = sam3_service.image_width
IMAGE_HEIGHT = sam3_service.image_height
ORIGINAL_BASE64 = pil_to_base64(original_image)


# ============================================================
# 7. Flask와 추론 상태
# ============================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 1024 * 1024
install_runtime_observability(app, Path(__file__).resolve().parents[1])

session_manager = SessionInferenceManager(
    processor=sam3_service.processor,
    gpu_lock=sam3_service.lock,
)

app.register_blueprint(
    create_session_blueprint(
        manager=session_manager,
        create_overlay_image=create_overlay_image,
        pil_to_base64=pil_to_base64,
    )
)


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(
        success=False,
        error="업로드 요청 크기가 제한을 초과했습니다.",
    ), 413


# ============================================================
# 8. 웹페이지
# ============================================================

HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>CosMi-SAM 미생물 콜로니 검출</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            background: #10141c;
            color: #f5f7fb;
            font-family:
                Arial,
                "Malgun Gothic",
                sans-serif;
        }

        header {
            padding: 24px 30px;
            background: #172033;
            border-bottom: 1px solid #2c3951;
        }

        header h1 {
            margin: 0 0 8px;
            font-size: 26px;
        }

        header p {
            margin: 0;
            color: #abb6ca;
            line-height: 1.6;
        }

        main {
            width: min(1200px, calc(100% - 32px));
            margin: 24px auto 50px;
        }

        .panel {
            padding: 20px;
            margin-bottom: 18px;
            background: #172033;
            border: 1px solid #2c3951;
            border-radius: 12px;
        }

        .panel-title {
            margin: 0 0 14px;
            font-size: 19px;
        }

        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
        }

        label {
            font-weight: bold;
        }

        input {
            width: 210px;
            padding: 11px 13px;
            background: #0f1624;
            color: white;
            border: 1px solid #43516b;
            border-radius: 7px;
            font-size: 16px;
        }

        input:focus {
            outline: 2px solid #4396ff;
        }

        button {
            padding: 11px 18px;
            border: 0;
            border-radius: 7px;
            color: white;
            font-size: 15px;
            font-weight: bold;
            cursor: pointer;
        }

        button:disabled {
            opacity: 0.55;
            cursor: wait;
        }

        #runText {
            background: #1976d2;
        }

        #resetButton {
            background: #586274;
        }

        #status {
            margin-top: 14px;
            min-height: 48px;
            padding: 13px;
            background: #0f1624;
            border-radius: 7px;
            color: #d7e4fa;
            line-height: 1.5;
        }

        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 22px;
            margin-bottom: 14px;
        }

        .legend-item {
            display: flex;
            gap: 8px;
            align-items: center;
            color: #ced6e5;
        }

        .color {
            width: 19px;
            height: 19px;
            border: 2px solid black;
        }

        .red {
            background: #ff2525;
        }

        .green {
            background: #00ff3c;
        }

        .yellow {
            background: #ffff00;
        }

        .canvas-container {
            max-width: 100%;
            overflow: auto;
            padding: 8px;
            background: black;
            border: 1px solid #46546c;
            border-radius: 8px;
        }

        canvas {
            display: block;
            max-width: 100%;
            height: auto;
            cursor: crosshair;
        }

        .counts {
            display: grid;
            grid-template-columns:
                repeat(3, minmax(150px, 1fr));
            gap: 12px;
            margin-top: 14px;
        }

        .count-card {
            padding: 15px;
            text-align: center;
            background: #0f1624;
            border-radius: 8px;
        }

        .count-label {
            display: block;
            margin-bottom: 7px;
            color: #aeb9cb;
        }

        .count-value {
            font-size: 25px;
            font-weight: bold;
        }

        .guide {
            color: #c7d0df;
            line-height: 1.9;
        }

        .guide ol {
            margin: 0;
            padding-left: 24px;
        }

        @media (max-width: 700px) {
            .counts {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>

<body>
    <header>
        <h1>CosMi-SAM 미생물 콜로니 검출</h1>

        <p>
            Text Prompt로 콜로니를 1차 검출하고,
            놓친 콜로니를 Bounding Box로 지정하여
            Visual Prompt 추론을 수행합니다.
        </p>
    </header>

    <main>
        <section class="panel">
            <h2 class="panel-title">1. Text Prompt 추론</h2>

            <div class="controls">
                <label for="promptInput">Prompt</label>

                <input
                    id="promptInput"
                    type="text"
                    maxlength="1"
                    placeholder="A를 입력하세요"
                    autocomplete="off"
                >

                <button id="runText">
                    Text Inference
                </button>

                <button id="resetButton">
                    Reset
                </button>
            </div>

            <div id="status">
                A를 입력한 다음 Text Inference 버튼을 누르세요.
            </div>
        </section>

        <section class="panel">
            <h2 class="panel-title">2. 검출 결과 및 Visual Prompt</h2>

            <div class="legend">
                <div class="legend-item">
                    <span class="color red"></span>
                    Text Prompt 검출
                </div>

                <div class="legend-item">
                    <span class="color green"></span>
                    Visual Prompt 검출
                </div>

                <div class="legend-item">
                    <span class="color yellow"></span>
                    사용자가 지정한 Box
                </div>
            </div>

            <div class="canvas-container">
                <canvas
                    id="canvas"
                    width="{{ width }}"
                    height="{{ height }}"
                ></canvas>
            </div>

            <div class="counts">
                <div class="count-card">
                    <span class="count-label">
                        Text 검출 수
                    </span>

                    <span
                        id="textCount"
                        class="count-value"
                    >0</span>
                </div>

                <div class="count-card">
                    <span class="count-label">
                        Visual 검출 수
                    </span>

                    <span
                        id="visualCount"
                        class="count-value"
                    >0</span>
                </div>

                <div class="count-card">
                    <span class="count-label">
                        지정한 Box 수
                    </span>

                    <span
                        id="boxCount"
                        class="count-value"
                    >0</span>
                </div>
            </div>
        </section>

        <section class="panel guide">
            <h2 class="panel-title">사용 순서</h2>

            <ol>
                <li>Prompt 입력창에 A를 입력합니다.</li>
                <li>Text Inference 버튼을 누릅니다.</li>
                <li>빨간색 영역으로 1차 검출 결과를 확인합니다.</li>
                <li>놓친 콜로니를 마우스로 드래그해 Box를 그립니다.</li>
                <li>Box 입력이 끝나면 Visual Prompt 추론이 실행됩니다.</li>
                <li>Visual Prompt 결과는 초록색으로 표시됩니다.</li>
                <li>다른 객체에도 Box를 계속 추가할 수 있습니다.</li>
            </ol>
        </section>
    </main>

    <script>
        const canvas =
            document.getElementById("canvas");

        const ctx =
            canvas.getContext("2d");

        const promptInput =
            document.getElementById("promptInput");

        const runTextButton =
            document.getElementById("runText");

        const resetButton =
            document.getElementById("resetButton");

        const statusElement =
            document.getElementById("status");

        const textCountElement =
            document.getElementById("textCount");

        const visualCountElement =
            document.getElementById("visualCount");

        const boxCountElement =
            document.getElementById("boxCount");

        let currentImage = new Image();

        let drawing = false;
        let startX = 0;
        let startY = 0;
        let currentX = 0;
        let currentY = 0;

        let textInferenceDone = false;
        let requestRunning = false;


        function setStatus(message) {
            statusElement.textContent = message;
        }


        function loadImage(source) {
            const nextImage = new Image();

            nextImage.onload = function () {
                currentImage = nextImage;

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

            nextImage.src = source;
        }


        function getCanvasPosition(event) {
            const rect =
                canvas.getBoundingClientRect();

            const scaleX =
                canvas.width / rect.width;

            const scaleY =
                canvas.height / rect.height;

            return {
                x: (
                    event.clientX - rect.left
                ) * scaleX,

                y: (
                    event.clientY - rect.top
                ) * scaleY
            };
        }


        function redrawCurrentImage() {
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
        }


        loadImage("{{ image_data }}");


        promptInput.addEventListener(
            "input",
            function () {
                this.value =
                    this.value.toUpperCase();
            }
        );


        promptInput.addEventListener(
            "keydown",
            function (event) {
                if (event.key === "Enter") {
                    runTextInference();
                }
            }
        );


        runTextButton.addEventListener(
            "click",
            runTextInference
        );


        async function runTextInference() {
            if (requestRunning) {
                return;
            }

            const prompt =
                promptInput.value.trim().toUpperCase();

            promptInput.value = prompt;

            if (!prompt) {
                setStatus("Prompt A를 입력하세요.");
                return;
            }

            if (prompt !== "A") {
                setStatus("현재 버전에서는 A Prompt만 지원합니다.");
                return;
            }

            requestRunning = true;
            runTextButton.disabled = true;

            setStatus(
                `Prompt "${prompt}"로 Text 추론 중입니다.`
            );

            try {
                const response = await fetch(
                    "/text_infer",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            prompt: prompt
                        })
                    }
                );

                const data = await response.json();

                if (!data.success) {
                    setStatus(
                        `Text 추론 오류: ${data.error}`
                    );

                    return;
                }

                loadImage(data.image);

                textCountElement.textContent =
                    data.text_count;

                visualCountElement.textContent = 0;
                boxCountElement.textContent = 0;

                textInferenceDone = true;

                setStatus(
                    `Text Prompt 검출 완료: `
                    + `${data.text_count}개가 검출되었습니다. `
                    + `놓친 객체에 Box를 그리세요.`
                );
            } catch (error) {
                setStatus(
                    `서버 통신 오류: ${error}`
                );
            } finally {
                requestRunning = false;
                runTextButton.disabled = false;
            }
        }


        canvas.addEventListener(
            "mousedown",
            function (event) {
                if (requestRunning) {
                    setStatus(
                        "현재 추론 중입니다. 잠시 기다려 주세요."
                    );

                    return;
                }

                if (!textInferenceDone) {
                    setStatus(
                        "먼저 Text Prompt 추론을 실행하세요."
                    );

                    return;
                }

                const position =
                    getCanvasPosition(event);

                startX = position.x;
                startY = position.y;

                currentX = position.x;
                currentY = position.y;

                drawing = true;
            }
        );


        canvas.addEventListener(
            "mousemove",
            function (event) {
                if (!drawing) {
                    return;
                }

                const position =
                    getCanvasPosition(event);

                currentX = position.x;
                currentY = position.y;

                redrawCurrentImage();

                // 검은색 테두리
                ctx.strokeStyle = "black";
                ctx.lineWidth = 8;

                ctx.strokeRect(
                    startX,
                    startY,
                    currentX - startX,
                    currentY - startY
                );

                // 노란색 Box
                ctx.strokeStyle = "yellow";
                ctx.lineWidth = 4;

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
            async function (event) {
                if (!drawing) {
                    return;
                }

                drawing = false;

                const position =
                    getCanvasPosition(event);

                const x1 =
                    Math.min(startX, position.x);

                const y1 =
                    Math.min(startY, position.y);

                const x2 =
                    Math.max(startX, position.x);

                const y2 =
                    Math.max(startY, position.y);

                if (
                    x2 - x1 < 5
                    || y2 - y1 < 5
                ) {
                    redrawCurrentImage();

                    setStatus(
                        "Bounding Box가 너무 작습니다."
                    );

                    return;
                }

                await runBoxInference(
                    x1,
                    y1,
                    x2,
                    y2
                );
            }
        );


        canvas.addEventListener(
            "mouseleave",
            function () {
                if (drawing) {
                    drawing = false;
                    redrawCurrentImage();
                }
            }
        );


        async function runBoxInference(
            x1,
            y1,
            x2,
            y2
        ) {
            if (requestRunning) {
                return;
            }

            requestRunning = true;
            runTextButton.disabled = true;

            setStatus(
                "Visual Box Prompt 추론 중입니다."
            );

            try {
                const response = await fetch(
                    "/box_infer",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            x1: x1,
                            y1: y1,
                            x2: x2,
                            y2: y2
                        })
                    }
                );

                const data = await response.json();

                if (!data.success) {
                    redrawCurrentImage();

                    setStatus(
                        `Visual 추론 오류: ${data.error}`
                    );

                    return;
                }

                loadImage(data.image);

                textCountElement.textContent =
                    data.text_count;

                visualCountElement.textContent =
                    data.visual_count;

                boxCountElement.textContent =
                    data.box_count;

                setStatus(
                    `Visual Prompt 검출 완료: `
                    + `${data.visual_count}개 검출, `
                    + `${data.box_count}개 Box 입력`
                );
            } catch (error) {
                redrawCurrentImage();

                setStatus(
                    `서버 통신 오류: ${error}`
                );
            } finally {
                requestRunning = false;
                runTextButton.disabled = false;
            }
        }


        resetButton.addEventListener(
            "click",
            async function () {
                if (requestRunning) {
                    return;
                }

                requestRunning = true;
                runTextButton.disabled = true;
                resetButton.disabled = true;

                setStatus("상태를 초기화하는 중입니다.");

                try {
                    const response = await fetch(
                        "/reset",
                        {
                            method: "POST"
                        }
                    );

                    const data =
                        await response.json();

                    if (!data.success) {
                        setStatus(
                            `초기화 오류: ${data.error}`
                        );

                        return;
                    }

                    loadImage(data.image);

                    promptInput.value = "";

                    textCountElement.textContent = 0;
                    visualCountElement.textContent = 0;
                    boxCountElement.textContent = 0;

                    textInferenceDone = false;

                    setStatus(
                        "초기화 완료. Prompt A를 입력하세요."
                    );
                } catch (error) {
                    setStatus(
                        `초기화 통신 오류: ${error}`
                    );
                } finally {
                    requestRunning = false;
                    runTextButton.disabled = false;
                    resetButton.disabled = false;
                }
            }
        );
    </script>
</body>
</html>
"""


# ============================================================
# 9. Flask 라우트
# ============================================================

@app.route("/")
def index():
    return render_template_string(
        HTML,
        image_data=ORIGINAL_BASE64,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT
    )


@app.route("/text_infer", methods=["POST"])
def text_infer():
    data = request.get_json(silent=True) or {}
    user_prompt = str(data.get("prompt", "")).strip().upper()

    try:
        result = sam3_service.run_text_prompt(user_prompt)
        result_image = create_overlay_image(
            base_image=original_image,
            text_masks=result["text_masks"],
            visual_masks=result["visual_masks"],
            boxes=result["boxes"],
        )
        encoded_image = pil_to_base64(result_image)

        return jsonify(
            success=True,
            image=encoded_image,
            text_count=result["text_count"],
            display_prompt=result["display_prompt"],
        )

    except ValueError as error:
        return jsonify(success=False, error=str(error)), 400

    except torch.cuda.OutOfMemoryError as error:
        print()
        print("CUDA 메모리 부족:")
        print(repr(error))

        clear_cuda_cache()

        return jsonify(
            success=False,
            error=(
                "GPU 메모리가 부족합니다. "
                "Chrome, PowerPoint 등 GPU를 사용하는 프로그램을 "
                "종료한 후 서버를 다시 실행하세요."
            )
        ), 500

    except Exception as error:
        print()
        print("TEXT 추론 오류:")
        print(repr(error))

        clear_cuda_cache()

        return jsonify(
            success=False,
            error=str(error)
        ), 500


@app.route("/box_infer", methods=["POST"])
def box_infer():
    data = request.get_json(silent=True) or {}

    try:
        result = sam3_service.run_box_prompt(
            x1=data["x1"],
            y1=data["y1"],
            x2=data["x2"],
            y2=data["y2"],
        )
        result_image = create_overlay_image(
            base_image=original_image,
            text_masks=result["text_masks"],
            visual_masks=result["visual_masks"],
            boxes=result["boxes"],
        )
        encoded_image = pil_to_base64(result_image)

        return jsonify(
            success=True,
            image=encoded_image,
            text_count=result["text_count"],
            visual_count=result["visual_count"],
            box_count=result["box_count"],
        )

    except KeyError:
        return jsonify(
            success=False,
            error="Bounding Box 좌표가 올바르지 않습니다."
        ), 400

    except (PromptRequiredError, InvalidBoxError) as error:
        return jsonify(success=False, error=str(error)), 400

    except (ValueError, TypeError):
        return jsonify(
            success=False,
            error="Bounding Box 좌표는 숫자여야 합니다."
        ), 400

    except torch.cuda.OutOfMemoryError as error:
        print()
        print("CUDA 메모리 부족:")
        print(repr(error))

        clear_cuda_cache()

        return jsonify(
            success=False,
            error=(
                "Visual Prompt 추론 중 GPU 메모리가 부족해졌습니다. "
                "다른 프로그램을 종료하고 서버를 다시 실행하세요."
            )
        ), 500

    except Exception as error:
        print()
        print("BOX 추론 오류:")
        print(repr(error))

        clear_cuda_cache()

        return jsonify(
            success=False,
            error=str(error)
        ), 500


@app.route("/reset", methods=["POST"])
def reset():
    try:
        sam3_service.reset()

        return jsonify(
            success=True,
            image=ORIGINAL_BASE64
        )

    except Exception as error:
        return jsonify(
            success=False,
            error=str(error)
        ), 500


# ============================================================
# 10. 서버 실행
# ============================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("CosMi-SAM 미생물 콜로니 검출 서버")
    print("=" * 70)
    print(f"브라우저 주소: http://localhost:{SERVER_PORT}")
    print()
    print("사용 순서")
    print("1. 브라우저에서 위 주소를 엽니다.")
    print("2. Prompt 입력창에 A를 입력합니다.")
    print("3. Text Inference 버튼을 누릅니다.")
    print("4. 놓친 콜로니에 Bounding Box를 그립니다.")
    print()
    print("서버를 종료하려면 PowerShell에서 Ctrl+C를 누르세요.")
    print()

    app.run(
        host=SERVER_HOST,
        port=SERVER_PORT,
        debug=False,
        threaded=True,
        use_reloader=False
    )
