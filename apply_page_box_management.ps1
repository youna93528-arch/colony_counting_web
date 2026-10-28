$ErrorActionPreference = "Stop"

$pagePath = "C:\workspace\colony_counting_web\frontend\app\page.tsx"

if (-not (Test-Path -LiteralPath $pagePath)) {
    throw "page.tsx를 찾을 수 없습니다: $pagePath"
}

$content = [System.IO.File]::ReadAllText($pagePath)

if ($content.Contains("const undoLastBox = async")) {
    Write-Output "Box 관리 함수가 이미 적용되어 있습니다."
    exit 0
}

$oldStage = @'
    case "visual_prompt":
      return "Visual Prompt";

    case "reset":
      return "분석 초기화";
'@

$newStage = @'
    case "visual_prompt":
      return "Visual Prompt";

    case "box_undo":
      return "마지막 Box 취소";

    case "box_clear":
      return "모든 Box 삭제";

    case "reset":
      return "분석 초기화";
'@

if (-not $content.Contains($oldStage)) {
    throw "처리 단계 삽입 위치를 찾지 못했습니다."
}

$content = $content.Replace(
    $oldStage,
    $newStage
)

$resetMarker = @'
  const resetAnalysis = async () => {
'@

$boxFunctions = @'
  const undoLastBox = async () => {
    if (
      !sessionId ||
      !result?.text_done ||
      (result?.box_count ?? 0) === 0
    ) {
      return;
    }

    setIsBusy(true);
    setErrorMessage("");
    setStatusMessage(
      "마지막 Bounding Box를 취소하고 있습니다.",
    );

    try {
      const updated =
        await apiRequest<SessionResult>(
          `/api/sessions/${sessionId}/boxes/undo`,
          {
            method: "POST",
          },
        );

      setResult(updated);
      setDrawStart(null);
      setDraftBox(null);

      setActiveStep(
        updated.box_count > 0 ? 6 : 3,
      );

      setStatusMessage(
        `마지막 Box 취소 완료: Text ${updated.text_count}, Visual ${updated.visual_count}, Box ${updated.box_count}`,
      );
    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : "마지막 Box 취소에 실패했습니다.",
      );

      setStatusMessage(
        "마지막 Box 취소에 실패했습니다.",
      );
    } finally {
      setIsBusy(false);
    }
  };


  const clearAllBoxes = async () => {
    if (
      !sessionId ||
      !result?.text_done ||
      (result?.box_count ?? 0) === 0
    ) {
      return;
    }

    const confirmed = window.confirm(
      "입력한 모든 Bounding Box와 Visual Mask를 삭제할까요?\nText Prompt 검출 결과는 유지됩니다.",
    );

    if (!confirmed) {
      return;
    }

    setIsBusy(true);
    setErrorMessage("");
    setStatusMessage(
      "모든 Bounding Box를 삭제하고 있습니다.",
    );

    try {
      const updated =
        await apiRequest<SessionResult>(
          `/api/sessions/${sessionId}/boxes`,
          {
            method: "DELETE",
          },
        );

      setResult(updated);
      setDrawStart(null);
      setDraftBox(null);
      setActiveStep(3);

      setStatusMessage(
        `모든 Box 삭제 완료: Text ${updated.text_count}, Visual ${updated.visual_count}, Box ${updated.box_count}`,
      );
    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : "모든 Box 삭제에 실패했습니다.",
      );

      setStatusMessage(
        "모든 Box 삭제에 실패했습니다.",
      );
    } finally {
      setIsBusy(false);
    }
  };


'@

if (-not $content.Contains($resetMarker)) {
    throw "resetAnalysis 함수 위치를 찾지 못했습니다."
}

$content = $content.Replace(
    $resetMarker,
    $boxFunctions + $resetMarker
)

$oldButtons = @'
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
'@

$newButtons = @'
                <div className="flex flex-wrap justify-end gap-2">
                  <button
                    type="button"
                    disabled={
                      !sessionId ||
                      !result?.text_done ||
                      (result?.box_count ?? 0) === 0 ||
                      isBusy
                    }
                    className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs font-bold text-amber-800 hover:bg-amber-100 disabled:cursor-not-allowed disabled:opacity-40"
                    onClick={() =>
                      void undoLastBox()
                    }
                  >
                    마지막 Box 취소
                  </button>

                  <button
                    type="button"
                    disabled={
                      !sessionId ||
                      !result?.text_done ||
                      (result?.box_count ?? 0) === 0 ||
                      isBusy
                    }
                    className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs font-bold text-red-700 hover:bg-red-100 disabled:cursor-not-allowed disabled:opacity-40"
                    onClick={() =>
                      void clearAllBoxes()
                    }
                  >
                    모든 Box 지우기
                  </button>

                  <button
                    type="button"
                    disabled={!sessionId || isBusy}
                    className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-bold text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
                    onClick={() =>
                      void resetAnalysis()
                    }
                  >
                    초기화
                  </button>

                  <button
                    type="button"
                    disabled={!result?.image || isBusy}
                    className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs font-bold text-blue-700 hover:bg-blue-100 disabled:cursor-not-allowed disabled:opacity-40"
                    onClick={downloadResultImage}
                  >
                    결과 이미지 저장
                  </button>
                </div>
'@

if (-not $content.Contains($oldButtons)) {
    throw "기존 Canvas 버튼 영역을 찾지 못했습니다."
}

$content = $content.Replace(
    $oldButtons,
    $newButtons
)

$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)

[System.IO.File]::WriteAllText(
    $pagePath,
    $content,
    $utf8WithoutBom
)

Write-Output "page.tsx Box 관리 UI 적용 완료"