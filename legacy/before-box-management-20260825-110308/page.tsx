"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import type {
  ChangeEvent,
  DragEvent,
  MouseEvent as ReactMouseEvent,
} from "react";

import {
  MaskNmsShadowPanel,
} from "./components/MaskNmsShadowPanel";

import type {
  MaskNmsShadowResult,
} from "./components/MaskNmsShadowPanel";

import {
  downloadAnalysisCsv,
} from "./utils/csvExport";

type ProcessingMetadata = {
  schema_version: number;
  stage:
    | "upload"
    | "text_prompt"
    | "visual_prompt"
    | "reset";
  duration_ms: number;
  steps: Record<string, number>;
  algorithm_preserved: boolean;
  measurement_only: boolean;
  nms_mode: "not_applied" | "shadow";
};


type ResultLayers = {
  base: string;
  text: string;
  visual: string;
  boxes: string;
};


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
  text_scores: number[];
  visual_scores: number[];
  width: number;
  height: number;
  original_width: number;
  original_height: number;
  original_filename: string;
  detected_format: string;
  file_size: number;
  confidence_threshold: number;
  mask_threshold: number;
  allowed_prompts?: string[];
  processing?: ProcessingMetadata;
  mask_nms_shadow?: MaskNmsShadowResult;
  layers?: ResultLayers;
  error?: string;
};


type BasicResponse = {
  success: boolean;
  error?: string;
};


type TabId =
  | "analysis"
  | "pipeline"
  | "algorithm";


type Point = {
  x: number;
  y: number;
};


type DrawBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};


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
    label: "Bacillus cereus (B.c)",
  },
  {
    value: "A.b",
    label: "Acinetobacter baumannii (A.b)",
  },
];


const PIPELINE_STEPS = [
  "이미지 업로드",
  "이미지 검증 및 전처리",
  "Text Prompt 검출",
  "Visual Prompt 보정",
  "Mask-NMS Shadow",
  "결과 생성",
];


const PIPELINE_DESCRIPTIONS = [
  "JPG, PNG 또는 TIFF 이미지를 안전하게 업로드합니다.",
  "파일 내용 검증, RGB 변환 및 분석 크기를 준비합니다.",
  "선택한 미생물 Prompt로 초기 마스크를 생성합니다.",
  "놓친 콜로니에 지정한 Box를 Visual Prompt로 전달합니다.",
  "중복 후보를 분석하되 현재 원본 결과에는 반영하지 않습니다.",
  "검출 마스크와 Box를 결과 이미지로 표시합니다.",
];


const PROCESSING_STEP_LABELS: Record<string, string> = {
  file_read_ms: "업로드 파일 읽기",
  validation_ms: "이미지 보안 검증",
  session_creation_ms: "분석 세션 생성",
  sam3_text_inference_ms: "SAM3 Text Prompt 추론",
  sam3_visual_inference_ms: "SAM3 Visual Prompt 추론",
  mask_nms_shadow_ms: "Mask-NMS Shadow 분석",
  overlay_rendering_ms: "결과 Overlay 생성",
  image_encoding_ms: "결과 이미지 인코딩",
  session_reset_ms: "분석 상태 초기화",
};


function imageDataUrl(image?: string) {
  if (!image) {
    return "";
  }

  if (image.startsWith("data:")) {
    return image;
  }

  return `data:image/png;base64,${image}`;
}


function formatMilliseconds(value?: number) {
  if (value === undefined) {
    return "-";
  }

  return `${value.toLocaleString("ko-KR", {
    maximumFractionDigits: 2,
  })} ms`;
}


function formatBytes(value?: number) {
  if (!value) {
    return "-";
  }

  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }

  return `${(value / 1024 / 1024).toFixed(2)} MB`;
}


function processingStageLabel(
  stage?: ProcessingMetadata["stage"],
) {
  switch (stage) {
    case "upload":
      return "이미지 업로드";

    case "text_prompt":
      return "Text Prompt";

    case "visual_prompt":
      return "Visual Prompt";

    case "reset":
      return "분석 초기화";

    default:
      return "분석 대기";
  }
}


async function apiRequest<T extends BasicResponse>(
  url: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(url, options);
  const responseText = await response.text();

  let data: T;

  try {
    data = JSON.parse(responseText) as T;
  } catch {
    throw new Error(
      "서버가 JSON 대신 HTML을 반환했습니다. 백엔드와 프런트엔드 프록시 연결을 확인하세요.",
    );
  }

  if (!response.ok || !data.success) {
    throw new Error(
      data.error ??
        `요청 처리에 실패했습니다. HTTP ${response.status}`,
    );
  }

  return data;
}


export default function Home() {
  const [activeTab, setActiveTab] =
    useState<TabId>("analysis");

  const [selectedPrompt, setSelectedPrompt] =
    useState("P.g");

  const [sessionId, setSessionId] =
    useState<string | null>(null);

  const [result, setResult] =
    useState<SessionResult | null>(null);

  const [selectedFileName, setSelectedFileName] =
    useState("");

  const [statusMessage, setStatusMessage] =
    useState(
      "분석할 미생물 이미지를 업로드해 주세요.",
    );

  const [errorMessage, setErrorMessage] =
    useState("");

  const [isBusy, setIsBusy] =
    useState(false);

  const [isDraggingFile, setIsDraggingFile] =
    useState(false);

  const [activeStep, setActiveStep] =
    useState(0);

  const [drawStart, setDrawStart] =
    useState<Point | null>(null);

  const [draftBox, setDraftBox] =
    useState<DrawBox | null>(null);

  const [textLayerVisible, setTextLayerVisible] =
    useState(true);

  const [visualLayerVisible, setVisualLayerVisible] =
    useState(true);

  const [boxLayerVisible, setBoxLayerVisible] =
    useState(true);

  const fileInputRef =
    useRef<HTMLInputElement | null>(null);

  const canvasRef =
    useRef<HTMLCanvasElement | null>(null);

  const loadedImageRef =
    useRef<HTMLImageElement | null>(null);

  const loadedLayersRef = useRef<{
    base: HTMLImageElement;
    text: HTMLImageElement;
    visual: HTMLImageElement;
    boxes: HTMLImageElement;
  } | null>(null);

  const layerVisibilityRef = useRef({
    text: true,
    visual: true,
    boxes: true,
  });

  const currentSessionRef =
    useRef<string | null>(null);


  useEffect(() => {
    currentSessionRef.current = sessionId;
  }, [sessionId]);


  useEffect(() => {
    return () => {
      const currentSession =
        currentSessionRef.current;

      if (currentSession) {
        fetch(`/api/sessions/${currentSession}`, {
          method: "DELETE",
          keepalive: true,
        }).catch(() => undefined);
      }
    };
  }, []);


  const rawCandidateCount = useMemo(() => {
    if (!result) {
      return 0;
    }

    return result.text_count + result.visual_count;
  }, [result]);


  const drawCanvas = useCallback(
    (previewBox?: DrawBox | null) => {
      const canvas = canvasRef.current;
      const image = loadedImageRef.current;
      const layers = loadedLayersRef.current;

      if (!canvas || (!image && !layers)) {
        return;
      }

      const context = canvas.getContext("2d");

      if (!context) {
        return;
      }

      context.clearRect(
        0,
        0,
        canvas.width,
        canvas.height,
      );

      if (layers) {
        context.drawImage(
          layers.base,
          0,
          0,
          canvas.width,
          canvas.height,
        );

        if (layerVisibilityRef.current.text) {
          context.drawImage(
            layers.text,
            0,
            0,
            canvas.width,
            canvas.height,
          );
        }

        if (layerVisibilityRef.current.visual) {
          context.drawImage(
            layers.visual,
            0,
            0,
            canvas.width,
            canvas.height,
          );
        }

        if (layerVisibilityRef.current.boxes) {
          context.drawImage(
            layers.boxes,
            0,
            0,
            canvas.width,
            canvas.height,
          );
        }
      } else if (image) {
        context.drawImage(
          image,
          0,
          0,
          canvas.width,
          canvas.height,
        );
      }

      if (previewBox) {
        const left = Math.min(
          previewBox.x1,
          previewBox.x2,
        );

        const top = Math.min(
          previewBox.y1,
          previewBox.y2,
        );

        const width = Math.abs(
          previewBox.x2 - previewBox.x1,
        );

        const height = Math.abs(
          previewBox.y2 - previewBox.y1,
        );

        context.save();
        context.strokeStyle = "#f04438";
        context.fillStyle =
          "rgba(240, 68, 56, 0.10)";
        context.lineWidth = Math.max(
          2,
          canvas.width / 500,
        );
        context.setLineDash([10, 6]);
        context.fillRect(left, top, width, height);
        context.strokeRect(
          left,
          top,
          width,
          height,
        );
        context.restore();
      }
    },
    [],
  );


  useEffect(() => {
    const canvas = canvasRef.current;
    const source = imageDataUrl(result?.image);

    if (!canvas || !source || !result) {
      loadedImageRef.current = null;
      loadedLayersRef.current = null;
      return;
    }

    let cancelled = false;

    const loadResult = async () => {
      try {
        if (result.layers) {
          const [base, text, visual, boxes] = await Promise.all([
            loadBrowserImage(result.layers.base),
            loadBrowserImage(result.layers.text),
            loadBrowserImage(result.layers.visual),
            loadBrowserImage(result.layers.boxes),
          ]);

          if (cancelled) {
            return;
          }

          canvas.width = result.width || base.naturalWidth;
          canvas.height = result.height || base.naturalHeight;
          loadedImageRef.current = null;
          loadedLayersRef.current = {
            base,
            text,
            visual,
            boxes,
          };
          drawCanvas(null);
          return;
        }

        const image = await loadBrowserImage(source);

        if (cancelled) {
          return;
        }

        canvas.width = result.width || image.naturalWidth;
        canvas.height = result.height || image.naturalHeight;
        loadedLayersRef.current = null;
        loadedImageRef.current = image;
        drawCanvas(null);
      } catch (error) {
        if (!cancelled) {
          setErrorMessage(
            error instanceof Error
              ? error.message
              : "결과 이미지를 불러오지 못했습니다.",
          );
        }
      }
    };

    void loadResult();

    return () => {
      cancelled = true;
    };
  }, [
    drawCanvas,
    result?.image,
    result?.layers,
    result?.width,
    result?.height,
  ]);


  useEffect(() => {
    if (loadedImageRef.current || loadedLayersRef.current) {
      drawCanvas(draftBox);
    }
  }, [draftBox, drawCanvas]);


  useEffect(() => {
    layerVisibilityRef.current = {
      text: textLayerVisible,
      visual: visualLayerVisible,
      boxes: boxLayerVisible,
    };

    if (loadedLayersRef.current) {
      drawCanvas(draftBox);
    }
  }, [
    textLayerVisible,
    visualLayerVisible,
    boxLayerVisible,
    draftBox,
    drawCanvas,
  ]);


  const removePreviousSession =
    useCallback(async () => {
      const previousSession =
        currentSessionRef.current;

      if (!previousSession) {
        return;
      }

      try {
        await fetch(
          `/api/sessions/${previousSession}`,
          {
            method: "DELETE",
          },
        );
      } catch {
        // 새 세션 생성은 계속 진행합니다.
      }

      currentSessionRef.current = null;
      setSessionId(null);
    }, []);


  const uploadFile = useCallback(
    async (file: File) => {
      const extension =
        file.name
          .toLowerCase()
          .split(".")
          .pop() ?? "";

      const allowedExtensions = [
        "jpg",
        "jpeg",
        "png",
        "tif",
        "tiff",
      ];

      if (!allowedExtensions.includes(extension)) {
        setErrorMessage(
          "JPG, PNG 또는 TIFF 이미지만 업로드할 수 있습니다.",
        );
        return;
      }

      if (file.size > 30 * 1024 * 1024) {
        setErrorMessage(
          "이미지 파일은 최대 30MB까지 업로드할 수 있습니다.",
        );
        return;
      }

      setIsBusy(true);
      setErrorMessage("");
      setStatusMessage(
        "이미지를 검증하고 분석 세션을 생성하고 있습니다.",
      );
      setDrawStart(null);
      setDraftBox(null);

      try {
        await removePreviousSession();

        const formData = new FormData();
        formData.append("image", file);

        const uploaded =
          await apiRequest<SessionResult>(
            "/api/sessions",
            {
              method: "POST",
              body: formData,
            },
          );

        setSessionId(uploaded.session_id);
        currentSessionRef.current =
          uploaded.session_id;

        setResult(uploaded);
        setSelectedFileName(file.name);
        setActiveStep(2);

        setStatusMessage(
          "업로드 완료. Text Prompt 분석을 실행하세요.",
        );
      } catch (error) {
        setResult(null);
        setSelectedFileName("");
        setActiveStep(0);

        setErrorMessage(
          error instanceof Error
            ? error.message
            : "이미지 업로드에 실패했습니다.",
        );

        setStatusMessage(
          "이미지 업로드에 실패했습니다.",
        );
      } finally {
        setIsBusy(false);
      }
    },
    [removePreviousSession],
  );


  const handleFileInput = (
    event: ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];

    if (file) {
      void uploadFile(file);
    }

    event.currentTarget.value = "";
  };


  const handleDragOver = (
    event: DragEvent<HTMLDivElement>,
  ) => {
    event.preventDefault();
    setIsDraggingFile(true);
  };


  const handleDragLeave = (
    event: DragEvent<HTMLDivElement>,
  ) => {
    event.preventDefault();
    setIsDraggingFile(false);
  };


  const handleDrop = (
    event: DragEvent<HTMLDivElement>,
  ) => {
    event.preventDefault();
    setIsDraggingFile(false);

    const file = event.dataTransfer.files?.[0];

    if (file) {
      void uploadFile(file);
    }
  };


  const runTextInference = async () => {
    if (!sessionId) {
      setErrorMessage(
        "먼저 분석 이미지를 업로드해 주세요.",
      );
      return;
    }

    setIsBusy(true);
    setErrorMessage("");
    setStatusMessage(
      `${selectedPrompt} Text Prompt 분석을 실행하고 있습니다.`,
    );

    try {
      const inferred =
        await apiRequest<SessionResult>(
          `/api/sessions/${sessionId}/text-infer`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              prompt: selectedPrompt,
            }),
          },
        );

      setResult(inferred);
      setActiveStep(3);

      setStatusMessage(
        `Text Prompt 검출 완료: ${inferred.text_count}개가 검출되었습니다. 놓친 객체에 Box를 그리세요.`,
      );
    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : "Text Prompt 분석에 실패했습니다.",
      );

      setStatusMessage(
        "Text Prompt 분석에 실패했습니다.",
      );
    } finally {
      setIsBusy(false);
    }
  };


  const runBoxInference = useCallback(
    async (box: DrawBox) => {
      if (!sessionId || !result?.text_done) {
        setErrorMessage(
          "Text Prompt 분석을 먼저 실행해 주세요.",
        );
        return;
      }

      setIsBusy(true);
      setErrorMessage("");
      setStatusMessage(
        "Visual Box Prompt 분석을 실행하고 있습니다.",
      );

      try {
        const inferred =
          await apiRequest<SessionResult>(
            `/api/sessions/${sessionId}/box-infer`,
            {
              method: "POST",
              headers: {
                "Content-Type":
                  "application/json",
              },
              body: JSON.stringify(box),
            },
          );

        setResult(inferred);
        setActiveStep(6);

        setStatusMessage(
          `Visual Prompt 완료: Text ${inferred.text_count}, Visual ${inferred.visual_count}, Box ${inferred.box_count}`,
        );
      } catch (error) {
        setErrorMessage(
          error instanceof Error
            ? error.message
            : "Visual Prompt 분석에 실패했습니다.",
        );

        setStatusMessage(
          "Visual Prompt 분석에 실패했습니다.",
        );
      } finally {
        setIsBusy(false);
      }
    },
    [result?.text_done, sessionId],
  );


  const resetAnalysis = async () => {
    if (!sessionId) {
      return;
    }

    setIsBusy(true);
    setErrorMessage("");
    setStatusMessage(
      "현재 분석 상태를 초기화하고 있습니다.",
    );

    try {
      const resetResult =
        await apiRequest<SessionResult>(
          `/api/sessions/${sessionId}/reset`,
          {
            method: "POST",
          },
        );

      setResult(resetResult);
      setDrawStart(null);
      setDraftBox(null);
      setActiveStep(2);

      setStatusMessage(
        "분석 상태가 초기화되었습니다. Text Prompt를 다시 실행하세요.",
      );
    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : "분석 초기화에 실패했습니다.",
      );
    } finally {
      setIsBusy(false);
    }
  };


  const getCanvasPoint = (
    event: ReactMouseEvent<HTMLCanvasElement>,
  ): Point => {
    const canvas = event.currentTarget;
    const bounds = canvas.getBoundingClientRect();

    return {
      x:
        (event.clientX - bounds.left) *
        (canvas.width / bounds.width),
      y:
        (event.clientY - bounds.top) *
        (canvas.height / bounds.height),
    };
  };


  const handleCanvasMouseDown = (
    event: ReactMouseEvent<HTMLCanvasElement>,
  ) => {
    if (
      isBusy ||
      !result?.text_done ||
      !sessionId
    ) {
      return;
    }

    const point = getCanvasPoint(event);

    setDrawStart(point);
    setDraftBox({
      x1: point.x,
      y1: point.y,
      x2: point.x,
      y2: point.y,
    });
  };


  const handleCanvasMouseMove = (
    event: ReactMouseEvent<HTMLCanvasElement>,
  ) => {
    if (!drawStart || isBusy) {
      return;
    }

    const point = getCanvasPoint(event);

    setDraftBox({
      x1: drawStart.x,
      y1: drawStart.y,
      x2: point.x,
      y2: point.y,
    });
  };


  const handleCanvasMouseUp = (
    event: ReactMouseEvent<HTMLCanvasElement>,
  ) => {
    if (!drawStart || isBusy) {
      return;
    }

    const end = getCanvasPoint(event);

    const box = {
      x1: Number(
        Math.min(drawStart.x, end.x).toFixed(2),
      ),
      y1: Number(
        Math.min(drawStart.y, end.y).toFixed(2),
      ),
      x2: Number(
        Math.max(drawStart.x, end.x).toFixed(2),
      ),
      y2: Number(
        Math.max(drawStart.y, end.y).toFixed(2),
      ),
    };

    setDrawStart(null);
    setDraftBox(null);

    if (
      box.x2 - box.x1 < 4 ||
      box.y2 - box.y1 < 4
    ) {
      setErrorMessage(
        "Bounding Box를 조금 더 크게 그려주세요.",
      );
      return;
    }

    void runBoxInference(box);
  };


  const cancelBoxDrawing = () => {
    if (!drawStart) {
      return;
    }

    setDrawStart(null);
    setDraftBox(null);
  };


  const downloadResultImage = () => {
    const source = imageDataUrl(result?.image);

    if (!source) {
      return;
    }

    const baseName =
      result?.original_filename
        ?.replace(/\.[^/.]+$/, "")
        .replace(/[^\w가-힣.-]+/g, "_") ||
      "sam3-result";

    const link = document.createElement("a");
    link.href = source;
    link.download = `${baseName}-result.png`;
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const downloadCsvResult = () => {
    if (!result || !result.text_done) {
      setErrorMessage(
        "Text Prompt 분석을 먼저 실행해 주세요.",
      );
      return;
    }

    downloadAnalysisCsv(result);
  };
  return (
    <div className="min-h-screen bg-[#f4f7fb] text-slate-900">
      <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-6 px-5 py-4 lg:px-8">
          <button
            type="button"
            className="flex items-center gap-3 text-left"
            onClick={() => setActiveTab("analysis")}
          >
            <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-blue-600 text-sm font-black text-white shadow-sm">
              S3
            </span>

            <span>
              <strong className="block text-base font-extrabold tracking-tight">
                SAM3 Colony Intelligence
              </strong>

              <small className="text-xs font-medium text-slate-500">
                Microbial Detection Workbench
              </small>
            </span>
          </button>

          <nav className="hidden items-center gap-2 md:flex">
            <button
              type="button"
              className={
                activeTab === "analysis"
                  ? "rounded-lg bg-blue-50 px-4 py-2 text-sm font-bold text-blue-700"
                  : "rounded-lg px-4 py-2 text-sm font-semibold text-slate-600 hover:bg-slate-100"
              }
              onClick={() => setActiveTab("analysis")}
            >
              콜로니 분석
            </button>

            <button
              type="button"
              className={
                activeTab === "pipeline"
                  ? "rounded-lg bg-blue-50 px-4 py-2 text-sm font-bold text-blue-700"
                  : "rounded-lg px-4 py-2 text-sm font-semibold text-slate-600 hover:bg-slate-100"
              }
              onClick={() => setActiveTab("pipeline")}
            >
              처리 과정
            </button>

            <button
              type="button"
              className={
                activeTab === "algorithm"
                  ? "rounded-lg bg-blue-50 px-4 py-2 text-sm font-bold text-blue-700"
                  : "rounded-lg px-4 py-2 text-sm font-semibold text-slate-600 hover:bg-slate-100"
              }
              onClick={() => setActiveTab("algorithm")}
            >
              알고리즘 설명
            </button>
          </nav>

          <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-extrabold text-emerald-700">
            Local Secure Session
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-[1600px] px-5 py-7 lg:px-8">
        <section className="mb-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
            {PIPELINE_STEPS.map((step, index) => {
              const completed = index < activeStep;
              const current = index === activeStep;

              return (
                <div
                  key={step}
                  className={
                    completed
                      ? "rounded-xl border border-emerald-200 bg-emerald-50 p-3"
                      : current
                        ? "rounded-xl border border-blue-300 bg-blue-50 p-3"
                        : "rounded-xl border border-slate-200 bg-slate-50 p-3 opacity-60"
                  }
                >
                  <div className="mb-2 flex items-center justify-between">
                    <span
                      className={
                        completed
                          ? "flex h-7 w-7 items-center justify-center rounded-full bg-emerald-600 text-xs font-black text-white"
                          : current
                            ? "flex h-7 w-7 items-center justify-center rounded-full bg-blue-600 text-xs font-black text-white"
                            : "flex h-7 w-7 items-center justify-center rounded-full bg-slate-300 text-xs font-black text-white"
                      }
                    >
                      {completed ? "✓" : index + 1}
                    </span>

                    <small className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                      STEP
                    </small>
                  </div>

                  <strong className="text-xs leading-5">
                    {step}
                  </strong>
                </div>
              );
            })}
          </div>
        </section>

        {errorMessage && (
          <div className="mb-6 flex items-start justify-between gap-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            <span>{errorMessage}</span>

            <button
              type="button"
              className="font-black"
              onClick={() => setErrorMessage("")}
            >
              ×
            </button>
          </div>
        )}

        {activeTab === "analysis" && (
          <section className="grid gap-5 xl:grid-cols-[300px_minmax(0,1fr)_320px]">
            <aside className="space-y-5">
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="mb-4">
                  <span className="text-xs font-extrabold tracking-wider text-blue-600">
                    INPUT
                  </span>

                  <h2 className="mt-1 text-lg font-extrabold">
                    이미지 업로드
                  </h2>
                </div>

                <div
                  className={
                    isDraggingFile
                      ? "cursor-pointer rounded-xl border-2 border-dashed border-blue-500 bg-blue-50 p-6 text-center"
                      : "cursor-pointer rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 p-6 text-center hover:border-blue-400 hover:bg-blue-50"
                  }
                  role="button"
                  tabIndex={0}
                  onClick={() =>
                    fileInputRef.current?.click()
                  }
                  onKeyDown={(event) => {
                    if (
                      event.key === "Enter" ||
                      event.key === " "
                    ) {
                      fileInputRef.current?.click();
                    }
                  }}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                >
                  <span className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-white text-xl shadow-sm">
                    ↑
                  </span>

                  <strong className="block text-sm">
                    이미지를 끌어놓거나 클릭하세요
                  </strong>

                  <small className="mt-2 block leading-5 text-slate-500">
                    JPG · PNG · TIFF
                    <br />
                    최대 30MB
                  </small>
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".jpg,.jpeg,.png,.tif,.tiff,image/jpeg,image/png,image/tiff"
                  className="hidden"
                  onChange={handleFileInput}
                />

                {selectedFileName && (
                  <div className="mt-3 rounded-lg bg-slate-100 px-3 py-2">
                    <small className="block text-slate-500">
                      선택된 파일
                    </small>

                    <strong className="block truncate text-xs">
                      {selectedFileName}
                    </strong>
                  </div>
                )}
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="mb-4">
                  <span className="text-xs font-extrabold tracking-wider text-blue-600">
                    PROMPT
                  </span>

                  <h2 className="mt-1 text-lg font-extrabold">
                    검출 설정
                  </h2>
                </div>

                <label className="mb-2 block text-xs font-bold text-slate-600">
                  미생물 종류
                </label>

                <select
                  value={selectedPrompt}
                  disabled={isBusy}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm outline-none focus:border-blue-500"
                  onChange={(event) =>
                    setSelectedPrompt(
                      event.target.value,
                    )
                  }
                >
                  {PROMPT_OPTIONS.map((option) => (
                    <option
                      key={option.value}
                      value={option.value}
                    >
                      {option.label}
                    </option>
                  ))}
                </select>

                <button
                  type="button"
                  disabled={
                    !sessionId ||
                    isBusy
                  }
                  className="mt-4 w-full rounded-lg bg-blue-600 px-4 py-3 text-sm font-extrabold text-white shadow-sm hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
                  onClick={() =>
                    void runTextInference()
                  }
                >
                  {isBusy
                    ? "처리 중..."
                    : "Text Prompt 분석"}
                </button>

                <div className="mt-4 grid grid-cols-2 gap-2">
                  <div className="rounded-lg bg-slate-50 p-3">
                    <span className="block text-[11px] font-bold text-slate-500">
                      Confidence
                    </span>

                    <strong className="mt-1 block text-sm">
                      {result?.confidence_threshold ??
                        0.5}
                    </strong>
                  </div>

                  <div className="rounded-lg bg-slate-50 p-3">
                    <span className="block text-[11px] font-bold text-slate-500">
                      Mask Threshold
                    </span>

                    <strong className="mt-1 block text-sm">
                      {result?.mask_threshold ?? "-"}
                    </strong>
                  </div>
                </div>

                <p className="mt-3 text-xs leading-5 text-slate-500">
                  기준값 보존을 위해 Threshold는 읽기 전용으로
                  표시됩니다.
                </p>
              </div>
            </aside>

            <section className="min-w-0 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="mb-4 flex flex-wrap items-start justify-between gap-4">
                <div>
                  <span className="text-xs font-extrabold tracking-wider text-blue-600">
                    CANVAS
                  </span>

                  <h1 className="mt-1 text-xl font-black">
                    SAM3 미생물 콜로니 검출
                  </h1>

                  <p className="mt-1 text-sm text-slate-500">
                    Text 분석 후 놓친 콜로니를 드래그해
                    Bounding Box로 지정하세요.
                  </p>
                </div>

                <div className="flex gap-2">
                  <button
                    type="button"
                    disabled={!sessionId || isBusy}
                    className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-bold text-slate-700 hover:bg-slate-50 disabled:opacity-40"
                    onClick={() =>
                      void resetAnalysis()
                    }
                  >
                    초기화
                  </button>

                  <button
                    type="button"
                    disabled={!result?.image}
                    className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs font-bold text-blue-700 hover:bg-blue-100 disabled:opacity-40"
                    onClick={downloadResultImage}
                  >
                    결과 이미지 저장
                  </button>
                </div>
              </div>

              <div className="relative flex min-h-[560px] items-center justify-center overflow-hidden rounded-xl border border-slate-200 bg-slate-950">
                {result?.image ? (
                  <canvas
                    ref={canvasRef}
                    className={
                      result.text_done
                        ? "block h-auto max-h-[720px] w-full cursor-crosshair object-contain"
                        : "block h-auto max-h-[720px] w-full cursor-default object-contain"
                    }
                    onMouseDown={
                      handleCanvasMouseDown
                    }
                    onMouseMove={
                      handleCanvasMouseMove
                    }
                    onMouseUp={handleCanvasMouseUp}
                    onMouseLeave={cancelBoxDrawing}
                  />
                ) : (
                  <div className="px-6 text-center text-white">
                    <span className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-white/10 text-2xl">
                      ◎
                    </span>

                    <strong className="block">
                      분석 이미지가 없습니다
                    </strong>

                    <p className="mt-2 text-sm text-slate-400">
                      왼쪽에서 이미지를 업로드하세요.
                    </p>
                  </div>
                )}

                {isBusy && (
                  <div className="absolute inset-0 flex items-center justify-center bg-slate-950/65 backdrop-blur-sm">
                    <div className="rounded-xl bg-white px-6 py-4 text-center shadow-xl">
                      <span className="mx-auto mb-3 block h-7 w-7 animate-spin rounded-full border-4 border-blue-100 border-t-blue-600" />

                      <strong className="text-sm">
                        SAM3 처리 중
                      </strong>
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-xl bg-slate-50 px-4 py-3">
                <div>
                  <span className="block text-[11px] font-bold text-slate-500">
                    분석 상태
                  </span>

                  <strong className="text-sm">
                    {statusMessage}
                  </strong>
                </div>

                {result && (
                  <span className="text-xs font-semibold text-slate-500">
                    {result.width} × {result.height}px
                  </span>
                )}
              </div>
            </section>

            <aside className="space-y-5">
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="mb-4">
                  <span className="text-xs font-extrabold tracking-wider text-blue-600">
                    RESULTS
                  </span>

                  <h2 className="mt-1 text-lg font-extrabold">
                    검출 결과
                  </h2>
                </div>

                <div className="grid grid-cols-3 gap-2">
                  <div className="rounded-xl bg-blue-50 p-3 text-center">
                    <span className="block text-[10px] font-bold text-blue-600">
                      TEXT
                    </span>

                    <strong className="mt-1 block text-2xl text-blue-800">
                      {result?.text_count ?? 0}
                    </strong>
                  </div>

                  <div className="rounded-xl bg-violet-50 p-3 text-center">
                    <span className="block text-[10px] font-bold text-violet-600">
                      VISUAL
                    </span>

                    <strong className="mt-1 block text-2xl text-violet-800">
                      {result?.visual_count ?? 0}
                    </strong>
                  </div>

                  <div className="rounded-xl bg-amber-50 p-3 text-center">
                    <span className="block text-[10px] font-bold text-amber-600">
                      BOX
                    </span>

                    <strong className="mt-1 block text-2xl text-amber-800">
                      {result?.box_count ?? 0}
                    </strong>
                  </div>
                </div>

                <div className="mt-3 rounded-xl border border-slate-200 p-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-slate-500">
                      Raw 후보 합계
                    </span>

                    <strong className="text-xl">
                      {rawCandidateCount}
                    </strong>
                  </div>

                  <p className="mt-2 text-xs leading-5 text-slate-500">
                    Text와 Visual 후보의 단순 합계이며 중복 제거된
                    최종 콜로니 수가 아닙니다.
                  </p>
                </div>
              </div>
              <MaskNmsShadowPanel
                data={result?.mask_nms_shadow}
              />
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h3 className="text-sm font-extrabold">
                  결과 레이어
                </h3>

                <div className="mt-4 space-y-3">
                  <label className="flex items-center justify-between gap-4 text-sm text-slate-600">
                    <span>Text Prompt Mask</span>
                    <input
                      type="checkbox"
                      checked={textLayerVisible}
                      disabled={!result?.layers}
                      onChange={(event) =>
                        setTextLayerVisible(event.target.checked)
                      }
                    />
                  </label>

                  <label className="flex items-center justify-between gap-4 text-sm text-slate-600">
                    <span>Visual Prompt Mask</span>
                    <input
                      type="checkbox"
                      checked={visualLayerVisible}
                      disabled={!result?.layers}
                      onChange={(event) =>
                        setVisualLayerVisible(event.target.checked)
                      }
                    />
                  </label>

                  <label className="flex items-center justify-between gap-4 text-sm text-slate-600">
                    <span>Bounding Box</span>
                    <input
                      type="checkbox"
                      checked={boxLayerVisible}
                      disabled={!result?.layers}
                      onChange={(event) =>
                        setBoxLayerVisible(event.target.checked)
                      }
                    />
                  </label>
                </div>

                <p className="mt-3 text-xs leading-5 text-slate-500">
                  각 체크박스로 Text·Visual·Box 표시만 변경합니다.
                  검출 수와 SAM3 결과값은 바뀌지 않습니다.
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h3 className="text-sm font-extrabold">
                  이미지 정보
                </h3>

                <dl className="mt-4 space-y-3 text-xs">
                  <div className="flex justify-between gap-4">
                    <dt className="text-slate-500">
                      원본 크기
                    </dt>

                    <dd className="font-bold">
                      {result
                        ? `${result.original_width} × ${result.original_height}`
                        : "-"}
                    </dd>
                  </div>

                  <div className="flex justify-between gap-4">
                    <dt className="text-slate-500">
                      파일 형식
                    </dt>

                    <dd className="font-bold">
                      {result?.detected_format ?? "-"}
                    </dd>
                  </div>

                  <div className="flex justify-between gap-4">
                    <dt className="text-slate-500">
                      파일 크기
                    </dt>

                    <dd className="font-bold">
                      {formatBytes(result?.file_size)}
                    </dd>
                  </div>

                  <div className="flex justify-between gap-4">
                    <dt className="text-slate-500">
                      최근 처리시간
                    </dt>

                    <dd className="font-bold text-blue-700">
                      {formatMilliseconds(
                        result?.processing?.duration_ms,
                      )}
                    </dd>
                  </div>
                </dl>

                <button
                  type="button"
                  disabled={!result?.text_done}
                  title={
                    result?.text_done
                      ? "현재 분석 결과를 CSV로 저장합니다."
                      : "Text Prompt 분석 후 사용할 수 있습니다."
                  }
                  className="mt-4 w-full rounded-lg bg-blue-600 px-4 py-3 text-xs font-extrabold text-white shadow-sm hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-500 disabled:shadow-none"
                  onClick={downloadCsvResult}
                >
                  결과 CSV 다운로드
                </button>
              </div>
            </aside>
          </section>
        )}

        {activeTab === "pipeline" && (
          <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-6 flex flex-wrap items-start justify-between gap-5">
              <div>
                <span className="text-xs font-extrabold tracking-wider text-blue-600">
                  PROCESSING OBSERVABILITY
                </span>

                <h2 className="mt-1 text-2xl font-black">
                  실시간 분석 파이프라인
                </h2>

                <p className="mt-2 text-sm text-slate-500">
                  SAM3 알고리즘을 변경하지 않고 API 바깥에서
                  측정한 최근 처리 결과입니다.
                </p>
              </div>

              <div className="flex flex-wrap gap-2">
                <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-extrabold text-emerald-700">
                  알고리즘 보존
                </span>

                <span className="rounded-full border border-amber-200 bg-amber-50 px-3 py-2 text-xs font-extrabold text-amber-700">
                  Mask-NMS 미적용
                </span>
              </div>
            </div>

            <div className="mb-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <article className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                <span className="text-xs font-bold text-slate-500">
                  최근 처리 단계
                </span>

                <strong className="mt-2 block text-lg">
                  {processingStageLabel(
                    result?.processing?.stage,
                  )}
                </strong>
              </article>

              <article className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                <span className="text-xs font-bold text-slate-500">
                  전체 처리 시간
                </span>

                <strong className="mt-2 block text-lg text-blue-700">
                  {formatMilliseconds(
                    result?.processing?.duration_ms,
                  )}
                </strong>
              </article>

              <article className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                <span className="text-xs font-bold text-slate-500">
                  Confidence
                </span>

                <strong className="mt-2 block text-lg">
                  {result?.confidence_threshold ?? 0.5}
                </strong>
              </article>

              <article className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                <span className="text-xs font-bold text-slate-500">
                  Mask Threshold
                </span>

                <strong className="mt-2 block text-lg">
                  {result?.mask_threshold ?? "-"}
                </strong>
              </article>
            </div>

            <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
              <div className="space-y-3">
                {PIPELINE_STEPS.map((step, index) => {
                  const completed =
                    index < activeStep;

                  const current =
                    index === activeStep;

                  return (
                    <article
                      key={step}
                      className={
                        completed
                          ? "flex gap-4 rounded-xl border border-emerald-200 bg-emerald-50 p-4"
                          : current
                            ? "flex gap-4 rounded-xl border border-blue-300 bg-blue-50 p-4"
                            : "flex gap-4 rounded-xl border border-slate-200 bg-slate-50 p-4 opacity-60"
                      }
                    >
                      <span
                        className={
                          completed
                            ? "flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-emerald-600 text-sm font-black text-white"
                            : current
                              ? "flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-blue-600 text-sm font-black text-white"
                              : "flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-slate-300 text-sm font-black text-white"
                        }
                      >
                        {completed
                          ? "✓"
                          : index + 1}
                      </span>

                      <div>
                        <strong>{step}</strong>

                        <p className="mt-1 text-sm leading-6 text-slate-500">
                          {
                            PIPELINE_DESCRIPTIONS[
                              index
                            ]
                          }
                        </p>
                      </div>
                    </article>
                  );
                })}
              </div>

              <aside className="rounded-xl border border-slate-200 p-5">
                <div className="flex items-start justify-between gap-4 border-b border-slate-200 pb-4">
                  <div>
                    <span className="text-xs font-bold text-slate-500">
                      최근 요청
                    </span>

                    <h3 className="mt-1 text-lg font-extrabold">
                      단계별 처리 시간
                    </h3>
                  </div>

                  <strong className="text-lg text-blue-700">
                    {formatMilliseconds(
                      result?.processing?.duration_ms,
                    )}
                  </strong>
                </div>

                {result?.processing &&
                Object.keys(result.processing.steps)
                  .length > 0 ? (
                  <div className="mt-5 space-y-5">
                    {Object.entries(
                      result.processing.steps,
                    ).map(([key, value]) => {
                      const total = Math.max(
                        result.processing
                          ?.duration_ms ?? 0,
                        0.01,
                      );

                      const percentage = Math.max(
                        4,
                        Math.min(
                          100,
                          (value / total) * 100,
                        ),
                      );

                      return (
                        <div key={key}>
                          <div className="mb-2 flex items-center justify-between gap-4 text-xs">
                            <span className="font-semibold text-slate-600">
                              {PROCESSING_STEP_LABELS[
                                key
                              ] ?? key}
                            </span>

                            <strong>
                              {formatMilliseconds(
                                value,
                              )}
                            </strong>
                          </div>

                          <div className="h-2 overflow-hidden rounded-full bg-slate-100">
                            <span
                              className="block h-full rounded-full bg-gradient-to-r from-blue-600 to-sky-400"
                              style={{
                                width: `${percentage}%`,
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="py-12 text-center">
                    <strong className="text-sm">
                      처리 기록이 없습니다.
                    </strong>

                    <p className="mt-2 text-xs leading-5 text-slate-500">
                      이미지 업로드나 Prompt 분석을 실행하면
                      실제 시간이 표시됩니다.
                    </p>
                  </div>
                )}
                <MaskNmsShadowPanel
                  data={result?.mask_nms_shadow}
                  compact
                />

                <div className="mt-6 grid grid-cols-3 gap-2">
                  <div className="rounded-lg bg-blue-50 p-3 text-center">
                    <span className="block text-[10px] font-bold text-blue-600">
                      TEXT
                    </span>

                    <strong className="mt-1 block text-xl">
                      {result?.text_count ?? 0}
                    </strong>
                  </div>

                  <div className="rounded-lg bg-violet-50 p-3 text-center">
                    <span className="block text-[10px] font-bold text-violet-600">
                      VISUAL
                    </span>

                    <strong className="mt-1 block text-xl">
                      {result?.visual_count ?? 0}
                    </strong>
                  </div>

                  <div className="rounded-lg bg-amber-50 p-3 text-center">
                    <span className="block text-[10px] font-bold text-amber-600">
                      BOX
                    </span>

                    <strong className="mt-1 block text-xl">
                      {result?.box_count ?? 0}
                    </strong>
                  </div>
                </div>

                <p className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-800">
                  Mask-NMS는 아직 실행되지 않습니다. 현재
                  Text·Visual·Box 값은 검증된 기존 알고리즘의
                  원본 결과입니다.
                </p>
              </aside>
            </div>
          </section>
        )}

        {activeTab === "algorithm" && (
          <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-7">
              <span className="text-xs font-extrabold tracking-wider text-blue-600">
                ALGORITHM
              </span>

              <h2 className="mt-1 text-2xl font-black">
                적용 알고리즘
              </h2>

              <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-500">
                SAM3의 Text Prompt와 Visual Prompt를 사용해
                초기 콜로니를 검출하고 놓친 객체를 보정합니다.
                현재 결과는 기존 SAM3 출력값을 그대로 사용합니다.
              </p>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <article className="rounded-xl border border-slate-200 bg-slate-50 p-5">
                <span className="text-xs font-black text-blue-600">
                  01
                </span>

                <h3 className="mt-2 text-lg font-extrabold">
                  이미지 검증 및 전처리
                </h3>

                <p className="mt-2 text-sm leading-6 text-slate-500">
                  파일 확장자만 확인하지 않고 실제 이미지
                  내용을 검사합니다. 검증된 이미지는 RGB 형식과
                  SAM3 분석 크기로 준비됩니다.
                </p>
              </article>

              <article className="rounded-xl border border-slate-200 bg-slate-50 p-5">
                <span className="text-xs font-black text-blue-600">
                  02
                </span>

                <h3 className="mt-2 text-lg font-extrabold">
                  Text Prompt Detection
                </h3>

                <p className="mt-2 text-sm leading-6 text-slate-500">
                  사용자가 선택한 미생물 종류를 Prompt로
                  전달하여 초기 객체 후보와 마스크 및
                  confidence score를 생성합니다.
                </p>
              </article>

              <article className="rounded-xl border border-slate-200 bg-slate-50 p-5">
                <span className="text-xs font-black text-blue-600">
                  03
                </span>

                <h3 className="mt-2 text-lg font-extrabold">
                  Visual Prompt Refinement
                </h3>

                <p className="mt-2 text-sm leading-6 text-slate-500">
                  Text Prompt가 놓친 콜로니에 사용자가 Bounding
                  Box를 지정하면 Visual Prompt로 추가
                  추론합니다.
                </p>
              </article>

              <article className="rounded-xl border border-amber-200 bg-amber-50 p-5">
                <span className="text-xs font-black text-amber-700">
                  04 · SHADOW
                </span>

                <h3 className="mt-2 text-lg font-extrabold">
                  Mask-NMS
                </h3>

                <p className="mt-2 text-sm leading-6 text-slate-600">
                  Text와 Visual 마스크의 IoU를 비교해 중복
                  후보를 찾는 단계입니다. 아직 결과에는
                  적용하지 않으며 이후 Shadow 검증을 거쳐
                  도입합니다.
                </p>
              </article>
            </div>

            <div className="mt-6 overflow-hidden rounded-xl border border-slate-200">
              <div className="grid grid-cols-[180px_1fr] border-b border-slate-200 bg-slate-50 px-4 py-3 text-xs">
                <strong>항목</strong>
                <strong>현재 정책</strong>
              </div>

              {[
                [
                  "모델",
                  "기존 checkpoint.pt 및 Epoch 58 가중치 유지",
                ],
                [
                  "Confidence",
                  `고정값 ${
                    result?.confidence_threshold ??
                    0.5
                  }`,
                ],
                [
                  "Mask Threshold",
                  `고정값 ${
                    result?.mask_threshold ?? "-"
                  }`,
                ],
                [
                  "Mask-NMS",
                  "미적용 · Shadow 검증 후 도입",
                ],
                [
                  "정밀도 최적화",
                  "FP16·양자화·torch.compile 미적용",
                ],
              ].map(([name, value]) => (
                <div
                  key={name}
                  className="grid grid-cols-[180px_1fr] border-b border-slate-100 px-4 py-3 text-sm last:border-b-0"
                >
                  <strong>{name}</strong>
                  <span className="text-slate-600">
                    {value}
                  </span>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}


function loadBrowserImage(source: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(
      new Error("결과 이미지 레이어를 불러오지 못했습니다."),
    );
    image.src = imageDataUrl(source);
  });
}
