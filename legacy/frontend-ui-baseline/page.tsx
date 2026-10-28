"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import type {
  ChangeEvent,
  DragEvent,
  MouseEvent as ReactMouseEvent,
} from "react";


type SessionResult = {
  success: boolean;
  session_id: string;
  image?: string;
  prompt: string;
  text_done: boolean;
  text_count: number;
  visual_count: number;
  box_count: number;
  boxes: number[][];
  width: number;
  height: number;
  original_width: number;
  original_height: number;
  original_filename: string;
  detected_format: string;
  file_size: number;
  confidence_threshold: number;
  mask_threshold: number;
  error?: string;
};


type TabId =
  | "analysis"
  | "pipeline"
  | "algorithm";


const PROMPT_OPTIONS = [
  {
    value: "P.g",
    label: "Pseudomonas gergoviae (P.g)",
  },
  {
    value: "P.a",
    label: "Pseudomonas aeruginosa (P.a)",
  },
  {
    value: "C.a",
    label: "Candida albicans (C.a)",
  },
  {
    value: "S.a",
    label: "Staphylococcus aureus (S.a)",
  },
  {
    value: "E.c",
    label: "Escherichia coli (E.c)",
  },
  {
    value: "B.c",
    label: "Burkholderia cepacia (B.c)",
  },
  {
    value: "A.b",
    label: "Aspergillus brasiliensis (A.b)",
  },
];


const PIPELINE_STEPS = [
  "이미지 업로드",
  "검증·리사이즈",
  "Text Prompt",
  "Visual Prompt",
  "Mask-NMS",
  "최종 계수",
];


async function readApiResponse(
  response: Response,
): Promise<SessionResult> {
  const data =
    await response.json();

  if (!response.ok || !data.success) {
    throw new Error(
      data.error
      ?? "서버 요청에 실패했습니다.",
    );
  }

  return data;
}


export default function Home() {
  const canvasRef =
    useRef<HTMLCanvasElement>(null);

  const imageRef =
    useRef<HTMLImageElement | null>(null);

  const fileInputRef =
    useRef<HTMLInputElement>(null);

  const dragStartRef =
    useRef<{ x: number; y: number } | null>(
      null,
    );

  const [activeTab, setActiveTab] =
    useState<TabId>("analysis");

  const [sessionId, setSessionId] =
    useState("");

  const [selectedPrompt, setSelectedPrompt] =
    useState("P.g");

  const [imageData, setImageData] =
    useState("");

  const [result, setResult] =
    useState<SessionResult | null>(null);

  const [status, setStatus] =
    useState(
      "검사 이미지를 업로드해 분석을 시작하세요.",
    );

  const [activeStep, setActiveStep] =
    useState(0);

  const [busy, setBusy] =
    useState(false);

  const [draggingFile, setDraggingFile] =
    useState(false);

  const [drawingBox, setDrawingBox] =
    useState(false);

  const [textVisible, setTextVisible] =
    useState(true);

  const [visualVisible, setVisualVisible] =
    useState(true);

  const [boxVisible, setBoxVisible] =
    useState(true);


  const drawImage = useCallback(
    (source: string) => {
      const canvas =
        canvasRef.current;

      if (!canvas || !source) {
        return;
      }

      const image =
        new Image();

      image.onload = () => {
        canvas.width =
          result?.width
          ?? image.naturalWidth;

        canvas.height =
          result?.height
          ?? image.naturalHeight;

        const context =
          canvas.getContext("2d");

        if (!context) {
          return;
        }

        context.clearRect(
          0,
          0,
          canvas.width,
          canvas.height,
        );

        context.drawImage(
          image,
          0,
          0,
          canvas.width,
          canvas.height,
        );

        imageRef.current =
          image;
      };

      image.src = source;
    },
    [result?.height, result?.width],
  );


  useEffect(() => {
    if (imageData) {
      drawImage(imageData);
    }
  }, [drawImage, imageData]);


  const getCanvasPoint = (
    event: ReactMouseEvent<HTMLCanvasElement>,
  ) => {
    const canvas =
      canvasRef.current;

    if (!canvas) {
      return {
        x: 0,
        y: 0,
      };
    }

    const rect =
      canvas.getBoundingClientRect();

    return {
      x:
        (event.clientX - rect.left)
        * (canvas.width / rect.width),

      y:
        (event.clientY - rect.top)
        * (canvas.height / rect.height),
    };
  };


  const redrawCanvas = () => {
    const canvas =
      canvasRef.current;

    const image =
      imageRef.current;

    if (!canvas || !image) {
      return;
    }

    const context =
      canvas.getContext("2d");

    if (!context) {
      return;
    }

    context.clearRect(
      0,
      0,
      canvas.width,
      canvas.height,
    );

    context.drawImage(
      image,
      0,
      0,
      canvas.width,
      canvas.height,
    );
  };


  const uploadImage = async (
    file: File,
  ) => {
    const allowedTypes = [
      "image/jpeg",
      "image/png",
      "image/tiff",
    ];

    if (!allowedTypes.includes(file.type)) {
      setStatus(
        "JPG, PNG, TIFF 이미지만 업로드할 수 있습니다.",
      );

      return;
    }

    if (
      file.size
      > 30 * 1024 * 1024
    ) {
      setStatus(
        "이미지는 최대 30MB까지 업로드할 수 있습니다.",
      );

      return;
    }

    setBusy(true);
    setActiveStep(0);
    setStatus(
      "이미지를 안전하게 검증하고 있습니다.",
    );

    try {
      if (sessionId) {
        await fetch(
          `/api/sessions/${sessionId}`,
          {
            method: "DELETE",
          },
        );
      }

      const formData =
        new FormData();

      formData.append(
        "image",
        file,
      );

      const response =
        await fetch(
          "/api/sessions",
          {
            method: "POST",
            body: formData,
          },
        );

      const data =
        await readApiResponse(response);

      setSessionId(
        data.session_id,
      );

      setResult(data);

      setImageData(
        data.image ?? "",
      );

      setActiveStep(2);

      setStatus(
        "이미지 검증과 1008px 전처리가 완료됐습니다. 미생물 종류를 선택하세요.",
      );
    } catch (error) {
      setStatus(
        error instanceof Error
          ? error.message
          : "이미지 업로드에 실패했습니다.",
      );
    } finally {
      setBusy(false);
    }
  };


  const onFileInput = (
    event: ChangeEvent<HTMLInputElement>,
  ) => {
    const file =
      event.target.files?.[0];

    if (file) {
      void uploadImage(file);
    }

    event.target.value = "";
  };


  const onFileDrop = (
    event: DragEvent<HTMLDivElement>,
  ) => {
    event.preventDefault();

    setDraggingFile(false);

    const file =
      event.dataTransfer.files?.[0];

    if (file) {
      void uploadImage(file);
    }
  };


  const runTextInference = async () => {
    if (!sessionId) {
      setStatus(
        "먼저 이미지를 업로드하세요.",
      );

      return;
    }

    setBusy(true);
    setActiveStep(2);

    setStatus(
      `${selectedPrompt} Text Prompt 추론 중입니다.`,
    );

    try {
      const response =
        await fetch(
          `/api/sessions/${sessionId}/text-infer`,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body: JSON.stringify({
              prompt:
                selectedPrompt,
            }),
          },
        );

      const data =
        await readApiResponse(response);

      setResult(data);

      setImageData(
        data.image ?? "",
      );

      setActiveStep(3);

      setStatus(
        `Text Prompt 검출 완료: ${data.text_count}개. 놓친 콜로니에 Box를 그리세요.`,
      );
    } catch (error) {
      setStatus(
        error instanceof Error
          ? error.message
          : "Text 추론에 실패했습니다.",
      );
    } finally {
      setBusy(false);
    }
  };


  const runBoxInference = async (
    x1: number,
    y1: number,
    x2: number,
    y2: number,
  ) => {
    if (!sessionId) {
      return;
    }

    setBusy(true);
    setActiveStep(3);

    setStatus(
      "Visual Box Prompt 추론 중입니다.",
    );

    try {
      const response =
        await fetch(
          `/api/sessions/${sessionId}/box-infer`,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body: JSON.stringify({
              x1,
              y1,
              x2,
              y2,
            }),
          },
        );

      const data =
        await readApiResponse(response);

      setResult(data);

      setImageData(
        data.image ?? "",
      );

      setActiveStep(4);

      setStatus(
        `Visual Prompt 완료: 현재 ${data.visual_count}개 검출. Mask-NMS 적용 전 결과입니다.`,
      );
    } catch (error) {
      redrawCanvas();

      setStatus(
        error instanceof Error
          ? error.message
          : "Visual 추론에 실패했습니다.",
      );
    } finally {
      setBusy(false);
    }
  };


  const onCanvasMouseDown = (
    event: ReactMouseEvent<HTMLCanvasElement>,
  ) => {
    if (
      busy
      || !result?.text_done
    ) {
      setStatus(
        "먼저 Text Prompt 추론을 완료하세요.",
      );

      return;
    }

    dragStartRef.current =
      getCanvasPoint(event);

    setDrawingBox(true);
  };


  const onCanvasMouseMove = (
    event: ReactMouseEvent<HTMLCanvasElement>,
  ) => {
    if (
      !drawingBox
      || !dragStartRef.current
    ) {
      return;
    }

    const canvas =
      canvasRef.current;

    if (!canvas) {
      return;
    }

    const context =
      canvas.getContext("2d");

    if (!context) {
      return;
    }

    const point =
      getCanvasPoint(event);

    const start =
      dragStartRef.current;

    redrawCanvas();

    context.strokeStyle =
      "#111827";

    context.lineWidth =
      8;

    context.strokeRect(
      start.x,
      start.y,
      point.x - start.x,
      point.y - start.y,
    );

    context.strokeStyle =
      "#facc15";

    context.lineWidth =
      4;

    context.strokeRect(
      start.x,
      start.y,
      point.x - start.x,
      point.y - start.y,
    );
  };


  const onCanvasMouseUp = (
    event: ReactMouseEvent<HTMLCanvasElement>,
  ) => {
    const start =
      dragStartRef.current;

    if (!drawingBox || !start) {
      return;
    }

    const end =
      getCanvasPoint(event);

    setDrawingBox(false);

    dragStartRef.current =
      null;

    const x1 =
      Math.min(start.x, end.x);

    const y1 =
      Math.min(start.y, end.y);

    const x2 =
      Math.max(start.x, end.x);

    const y2 =
      Math.max(start.y, end.y);

    if (
      x2 - x1 < 5
      || y2 - y1 < 5
    ) {
      redrawCanvas();

      setStatus(
        "Bounding Box가 너무 작습니다.",
      );

      return;
    }

    void runBoxInference(
      x1,
      y1,
      x2,
      y2,
    );
  };


  const resetAnalysis = async () => {
    if (!sessionId) {
      return;
    }

    setBusy(true);

    try {
      const response =
        await fetch(
          `/api/sessions/${sessionId}/reset`,
          {
            method: "POST",
          },
        );

      const data =
        await readApiResponse(response);

      setResult(data);

      setImageData(
        data.image ?? "",
      );

      setActiveStep(2);

      setStatus(
        "분석 상태를 초기화했습니다.",
      );
    } catch (error) {
      setStatus(
        error instanceof Error
          ? error.message
          : "초기화에 실패했습니다.",
      );
    } finally {
      setBusy(false);
    }
  };


  const candidateCount =
    (result?.text_count ?? 0)
    + (result?.visual_count ?? 0);


  return (
    <div className="site-shell">
      <header className="top-nav">
        <div className="brand">
          <div className="brand-mark">
            <span />
            <span />
            <span />
            <span />
          </div>

          <div>
            <strong>
              SAM3 Colony Intelligence
            </strong>

            <small>
              미생물 콜로니 분석 워크벤치
            </small>
          </div>
        </div>

        <nav aria-label="주요 메뉴">
          <button className="nav-active">
            콜로니 분석
          </button>

          <button
            onClick={() =>
              setActiveTab("pipeline")
            }
          >
            처리 과정
          </button>

          <button
            onClick={() =>
              setActiveTab("algorithm")
            }
          >
            알고리즘
          </button>

          <button disabled>
            분석 이력
          </button>
        </nav>

        <div className="system-badge">
          <span className="status-dot" />
          SAM3 Ready
        </div>
      </header>

      <main className="page-container">
        <section className="page-heading">
          <div>
            <span className="eyebrow">
              COLONY COUNTING FRAMEWORK
            </span>

            <h1>
              SAM3 기반 미생물 콜로니 검출
            </h1>

            <p>
              Text Prompt로 초기 객체를 검출하고,
              Visual Box Prompt로 누락 객체를 보정합니다.
            </p>
          </div>

          <div className="model-summary">
            <span>
              Checkpoint
              <strong>Epoch 58</strong>
            </span>

            <span>
              Input
              <strong>1008 × 1008</strong>
            </span>

            <span>
              Device
              <strong>RTX 2080</strong>
            </span>
          </div>
        </section>

        <section className="stepper-card">
          {PIPELINE_STEPS.map(
            (step, index) => {
              const completed =
                index < activeStep;

              const current =
                index === activeStep;

              return (
                <div
                  className="step-item"
                  key={step}
                >
                  <div
                    className={[
                      "step-node",
                      completed
                        ? "step-done"
                        : "",
                      current
                        ? "step-current"
                        : "",
                    ].join(" ")}
                  >
                    {completed
                      ? "✓"
                      : index + 1}
                  </div>

                  <span>{step}</span>

                  {index
                    < PIPELINE_STEPS.length - 1
                    && (
                      <div
                        className={[
                          "step-line",
                          completed
                            ? "line-done"
                            : "",
                        ].join(" ")}
                      />
                    )}
                </div>
              );
            },
          )}
        </section>

        <div className="tab-bar">
          <button
            className={
              activeTab === "analysis"
                ? "tab-active"
                : ""
            }
            onClick={() =>
              setActiveTab("analysis")
            }
          >
            분석 워크벤치
          </button>

          <button
            className={
              activeTab === "pipeline"
                ? "tab-active"
                : ""
            }
            onClick={() =>
              setActiveTab("pipeline")
            }
          >
            처리 과정
          </button>

          <button
            className={
              activeTab === "algorithm"
                ? "tab-active"
                : ""
            }
            onClick={() =>
              setActiveTab("algorithm")
            }
          >
            알고리즘 설명
          </button>
        </div>

        {activeTab === "analysis" && (
          <section className="workspace-grid">
            <aside className="control-panel card">
              <div className="section-title">
                <span>01</span>

                <div>
                  <h2>분석 설정</h2>
                  <p>입력 이미지와 Prompt를 설정합니다.</p>
                </div>
              </div>

              <div
                className={[
                  "drop-zone",
                  draggingFile
                    ? "drop-zone-active"
                    : "",
                ].join(" ")}
                role="button"
                tabIndex={0}
                onClick={() =>
                  fileInputRef.current?.click()
                }
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    fileInputRef.current?.click();
                  }
                }}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDraggingFile(true);
                }}
                onDragLeave={() =>
                  setDraggingFile(false)
                }
                onDrop={onFileDrop}
              >
                <div className="upload-icon">
                  ↑
                </div>

                <strong>
                  이미지를 드래그하거나 클릭
                </strong>

                <span>
                  JPG · PNG · TIFF
                </span>

                <small>
                  최대 30MB · 2,500만 픽셀
                </small>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".jpg,.jpeg,.png,.tif,.tiff"
                  hidden
                  onChange={onFileInput}
                />
              </div>

              {result && (
                <div className="file-summary">
                  <strong>
                    {result.original_filename}
                  </strong>

                  <span>
                    원본 {result.original_width}
                    × {result.original_height}
                  </span>

                  <span>
                    분석 {result.width}
                    × {result.height}
                  </span>
                </div>
              )}

              <label className="field-label">
                미생물 종류
              </label>

              <select
                value={selectedPrompt}
                disabled={busy}
                onChange={(event) =>
                  setSelectedPrompt(
                    event.target.value,
                  )
                }
              >
                {PROMPT_OPTIONS.map(
                  (option) => (
                    <option
                      key={option.value}
                      value={option.value}
                    >
                      {option.label}
                    </option>
                  ),
                )}
              </select>

              <div className="threshold-grid">
                <div>
                  <span>Confidence</span>
                  <strong>
                    {result?.confidence_threshold
                      ?.toFixed(2)
                      ?? "0.50"}
                  </strong>
                </div>

                <div>
                  <span>Mask Threshold</span>
                  <strong>
                    {result?.mask_threshold
                      ?.toFixed(2)
                      ?? "0.50"}
                  </strong>
                </div>

                <div>
                  <span>Mask-NMS IoU</span>
                  <strong>0.50</strong>
                  <small>적용 예정</small>
                </div>
              </div>

              <button
                className="primary-button"
                disabled={
                  busy
                  || !sessionId
                }
                onClick={
                  runTextInference
                }
              >
                {busy
                  ? "처리 중..."
                  : "Text Prompt 분석"}
              </button>

              <button
                className="secondary-button"
                disabled={
                  busy
                  || !sessionId
                }
                onClick={
                  resetAnalysis
                }
              >
                분석 초기화
              </button>

              <div className="security-note">
                <strong>
                  안전한 이미지 처리
                </strong>

                <p>
                  파일 형식·크기·해상도를 서버에서
                  다시 검증하며 분석 세션은 사용자별로
                  격리됩니다.
                </p>
              </div>
            </aside>

            <section className="canvas-panel card">
              <div className="canvas-toolbar">
                <div>
                  <h2>분석 이미지</h2>
                  <p>
                    Text 검출 후 놓친 객체를 드래그하세요.
                  </p>
                </div>

                <div className="layer-switches">
                  <label>
                    <input
                      type="checkbox"
                      checked={textVisible}
                      onChange={(event) =>
                        setTextVisible(
                          event.target.checked,
                        )
                      }
                    />
                    Text
                  </label>

                  <label>
                    <input
                      type="checkbox"
                      checked={visualVisible}
                      onChange={(event) =>
                        setVisualVisible(
                          event.target.checked,
                        )
                      }
                    />
                    Visual
                  </label>

                  <label>
                    <input
                      type="checkbox"
                      checked={boxVisible}
                      onChange={(event) =>
                        setBoxVisible(
                          event.target.checked,
                        )
                      }
                    />
                    Box
                  </label>
                </div>
              </div>

              <div className="canvas-stage">
                {!imageData && (
                  <div className="empty-canvas">
                    <div className="empty-plate">
                      <span />
                      <span />
                      <span />
                      <span />
                      <span />
                    </div>

                    <strong>
                      분석 이미지가 없습니다
                    </strong>

                    <p>
                      왼쪽에서 이미지를 업로드하세요.
                    </p>
                  </div>
                )}

                <canvas
                  ref={canvasRef}
                  className={
                    imageData
                      ? "visible-canvas"
                      : "hidden-canvas"
                  }
                  onMouseDown={
                    onCanvasMouseDown
                  }
                  onMouseMove={
                    onCanvasMouseMove
                  }
                  onMouseUp={
                    onCanvasMouseUp
                  }
                  onMouseLeave={() => {
                    if (drawingBox) {
                      setDrawingBox(false);
                      dragStartRef.current =
                        null;
                      redrawCanvas();
                    }
                  }}
                />
              </div>

              <div className="status-strip">
                <span
                  className={
                    busy
                      ? "status-spinner"
                      : "status-check"
                  }
                >
                  {busy ? "" : "✓"}
                </span>

                <p>{status}</p>
              </div>
            </section>

            <aside className="result-panel card">
              <div className="section-title">
                <span>03</span>

                <div>
                  <h2>분석 결과</h2>
                  <p>현재 단계의 검출 통계입니다.</p>
                </div>
              </div>

              <div className="primary-count">
                <span>
                  Text Prompt 검출
                </span>

                <strong>
                  {result?.text_count ?? 0}
                </strong>

                <small>objects</small>
              </div>

              <div className="result-stats">
                <div>
                  <span>Visual 출력</span>
                  <strong>
                    {result?.visual_count ?? 0}
                  </strong>
                </div>

                <div>
                  <span>입력 Box</span>
                  <strong>
                    {result?.box_count ?? 0}
                  </strong>
                </div>

                <div>
                  <span>통합 후보</span>
                  <strong>
                    {candidateCount}
                  </strong>
                </div>
              </div>

              <div className="nms-card">
                <div>
                  <span>Mask-NMS</span>
                  <em>SHADOW MODE</em>
                </div>

                <p>
                  기존 Text·Visual 값을 유지하며,
                  중복 제거 결과는 별도 항목으로 추가됩니다.
                </p>

                <div className="nms-row">
                  <span>중복 제거</span>
                  <strong>대기</strong>
                </div>

                <div className="nms-row final-row">
                  <span>최종 고유 개수</span>
                  <strong>—</strong>
                </div>
              </div>

              <div className="legend-list">
                <div>
                  <span className="legend-red" />
                  Text Prompt 마스크
                </div>

                <div>
                  <span className="legend-green" />
                  Visual Prompt 마스크
                </div>

                <div>
                  <span className="legend-yellow" />
                  사용자 Bounding Box
                </div>
              </div>

              <a
                className={[
                  "download-button",
                  imageData
                    ? ""
                    : "download-disabled",
                ].join(" ")}
                href={
                  imageData || undefined
                }
                download="sam3-colony-result.png"
              >
                결과 이미지 다운로드
              </a>

              <button
                className="csv-button"
                disabled
                title="Mask-NMS와 검출 목록 API 연결 후 활성화됩니다."
              >
                CSV 다운로드 준비 중
              </button>
            </aside>
          </section>
        )}

        {activeTab === "pipeline" && (
          <section className="info-card card">
            <h2>실시간 분석 파이프라인</h2>

            <div className="pipeline-list">
              {PIPELINE_STEPS.map(
                (step, index) => (
                  <div key={step}>
                    <span>
                      {index + 1}
                    </span>

                    <div>
                      <strong>{step}</strong>

                      <p>
                        {
                          [
                            "검사 이미지를 안전하게 업로드합니다.",
                            "RGB 변환 후 긴 변을 1008px로 조정합니다.",
                            "선택한 미생물 Prompt로 초기 마스크를 생성합니다.",
                            "사용자 Box를 기하학 Prompt로 전달합니다.",
                            "Mask IoU 0.50 기준으로 중복 후보를 비교합니다.",
                            "최종 고유 마스크 수를 콜로니 개수로 표시합니다.",
                          ][index]
                        }
                      </p>
                    </div>
                  </div>
                ),
              )}
            </div>
          </section>
        )}

        {activeTab === "algorithm" && (
          <section className="info-card card">
            <h2>적용 알고리즘</h2>

            <div className="algorithm-grid">
              <article>
                <span>01</span>
                <h3>Text Prompt Detection</h3>
                <p>
                  미생물 종류를 언어 Prompt로 입력해
                  SAM3가 초기 객체 후보와 마스크를 생성합니다.
                </p>
              </article>

              <article>
                <span>02</span>
                <h3>Visual Prompt Refinement</h3>
                <p>
                  놓친 콜로니에 Bounding Box를 지정해
                  기하학적 단서를 추가합니다.
                </p>
              </article>

              <article>
                <span>03</span>
                <h3>Mask-NMS</h3>
                <p>
                  Text와 Visual 마스크의 IoU를 비교해
                  중복 후보를 제거합니다.
                </p>
              </article>

              <article>
                <span>04</span>
                <h3>Colony Counting</h3>
                <p>
                  중복 제거 후 남은 고유 마스크 수를
                  최종 콜로니 개수로 계산합니다.
                </p>
              </article>
            </div>

            <div className="truth-note">
              <strong>구현 상태 안내</strong>

              <p>
                현재 Text Prompt와 Visual Box Prompt는 실제
                SAM3 체크포인트로 동작합니다. Mask-NMS와 최종
                고유 계수는 기존 검출값을 보존한 상태에서 다음
                단계에 Shadow Mode로 추가됩니다.
              </p>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}