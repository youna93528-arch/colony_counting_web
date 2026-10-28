export type MaskNmsShadowResult = {
  mode: "shadow";
  status: "completed" | "error";
  applied_to_result: boolean;
  iou_threshold: number;
  mask_threshold: number;
  raw_count: number;
  shadow_unique_count: number | null;
  shadow_duplicate_count: number | null;
  kept_candidates: Array<{
    candidate_id: string;
    source: "text" | "visual";
    score: number;
  }>;
  suppressed_candidates: Array<{
    candidate_id: string;
    source: "text" | "visual";
    score: number;
    duplicate_of: string;
    iou: number;
  }>;
  warnings: string[];
};


type MaskNmsShadowPanelProps = {
  data?: MaskNmsShadowResult;
  compact?: boolean;
};


function displayCount(value: number | null) {
  return value === null ? "-" : value;
}


export function MaskNmsShadowPanel({
  data,
  compact = false,
}: MaskNmsShadowPanelProps) {
  if (!data) {
    return (
      <section
        className={
          compact
            ? "mt-5 rounded-xl border border-dashed border-slate-300 bg-slate-50 p-4"
            : "rounded-2xl border border-dashed border-slate-300 bg-white p-5 shadow-sm"
        }
      >
        <div className="flex items-center justify-between gap-3">
          <div>
            <span className="text-[11px] font-extrabold tracking-wider text-amber-700">
              MASK-NMS SHADOW
            </span>

            <h3 className="mt-1 text-sm font-extrabold">
              Shadow 분석 대기
            </h3>
          </div>

          <span className="rounded-full bg-slate-200 px-2.5 py-1 text-[10px] font-extrabold text-slate-600">
            준비 중
          </span>
        </div>

        <p className="mt-3 text-xs leading-5 text-slate-500">
          Text Prompt 분석 후 이미지에 Box를 그리면 중복
          마스크를 분석합니다.
        </p>
      </section>
    );
  }

  if (data.status === "error") {
    return (
      <section
        className={
          compact
            ? "mt-5 rounded-xl border border-red-200 bg-red-50 p-4"
            : "rounded-2xl border border-red-200 bg-red-50 p-5 shadow-sm"
        }
      >
        <span className="text-[11px] font-extrabold tracking-wider text-red-700">
          MASK-NMS SHADOW
        </span>

        <h3 className="mt-1 text-sm font-extrabold text-red-900">
          Shadow 분석 실패
        </h3>

        <p className="mt-2 text-xs leading-5 text-red-700">
          기존 SAM3 결과는 정상적으로 유지됐으며 Shadow
          결과는 적용되지 않았습니다.
        </p>
      </section>
    );
  }

  return (
    <section
      className={
        compact
          ? "mt-5 rounded-xl border border-amber-200 bg-amber-50/60 p-4"
          : "rounded-2xl border border-amber-200 bg-white p-5 shadow-sm"
      }
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <span className="text-[11px] font-extrabold tracking-wider text-amber-700">
            MASK-NMS SHADOW
          </span>

          <h3 className="mt-1 text-sm font-extrabold">
            중복 마스크 관찰 결과
          </h3>
        </div>

        <span className="rounded-full border border-amber-200 bg-amber-50 px-2.5 py-1 text-[10px] font-extrabold text-amber-700">
          결과 미적용
        </span>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2">
        <div className="rounded-lg bg-slate-100 p-3 text-center">
          <span className="block text-[10px] font-bold text-slate-500">
            RAW
          </span>

          <strong className="mt-1 block text-xl">
            {data.raw_count}
          </strong>
        </div>

        <div className="rounded-lg bg-emerald-50 p-3 text-center">
          <span className="block text-[10px] font-bold text-emerald-700">
            SHADOW 고유
          </span>

          <strong className="mt-1 block text-xl text-emerald-800">
            {displayCount(
              data.shadow_unique_count,
            )}
          </strong>
        </div>

        <div className="rounded-lg bg-red-50 p-3 text-center">
          <span className="block text-[10px] font-bold text-red-700">
            중복 후보
          </span>

          <strong className="mt-1 block text-xl text-red-800">
            {displayCount(
              data.shadow_duplicate_count,
            )}
          </strong>
        </div>
      </div>

      <dl className="mt-4 space-y-2 text-xs">
        <div className="flex items-center justify-between gap-4">
          <dt className="text-slate-500">
            Mask IoU Threshold
          </dt>

          <dd className="font-extrabold">
            {data.iou_threshold}
          </dd>
        </div>

        <div className="flex items-center justify-between gap-4">
          <dt className="text-slate-500">
            Mask Threshold
          </dt>

          <dd className="font-extrabold">
            {data.mask_threshold}
          </dd>
        </div>

        <div className="flex items-center justify-between gap-4">
          <dt className="text-slate-500">
            최종 결과 반영
          </dt>

          <dd className="font-extrabold text-amber-700">
            반영 안 함
          </dd>
        </div>
      </dl>

      {data.suppressed_candidates.length > 0 && (
        <details className="mt-4 rounded-lg border border-slate-200 bg-white p-3">
          <summary className="cursor-pointer text-xs font-extrabold">
            중복 후보 상세 보기
          </summary>

          <div className="mt-3 max-h-44 space-y-2 overflow-auto">
            {data.suppressed_candidates.map(
              (candidate) => (
                <div
                  key={candidate.candidate_id}
                  className="rounded-md bg-slate-50 px-3 py-2 text-[11px]"
                >
                  <strong>
                    {candidate.candidate_id}
                  </strong>

                  <span className="mx-2 text-slate-400">
                    →
                  </span>

                  <span>
                    {candidate.duplicate_of}
                  </span>

                  <span className="ml-2 font-bold text-red-600">
                    IoU {candidate.iou.toFixed(3)}
                  </span>
                </div>
              ),
            )}
          </div>
        </details>
      )}

      <p className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-[11px] leading-5 text-amber-800">
        이 값은 비교용 Shadow 결과입니다. 화면의 Text,
        Visual, Box 및 결과 이미지는 기존 SAM3 출력값을
        그대로 사용합니다.
      </p>
    </section>
  );
}