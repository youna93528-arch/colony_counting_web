"""
Windows 및 Triton 미지원 GPU를 위한
Euclidean Distance Transform 대체 구현.

기존 Triton GPU 커널 대신 OpenCV CPU 연산을 사용합니다.
정적 이미지 SAM3 추론에서는 이 함수가 사용되지 않거나
아주 제한적으로만 사용됩니다.
"""

import cv2
import numpy as np
import torch


def edt_triton(data: torch.Tensor):
    """
    배치 형태의 이진 마스크에 Euclidean Distance Transform을 수행합니다.

    Args:
        data:
            [B, H, W] 형태의 PyTorch Tensor.
            CUDA 또는 CPU Tensor를 모두 받을 수 있습니다.

    Returns:
        입력과 같은 장치에 위치한
        [B, H, W] 형태의 float32 Tensor.
    """

    if not torch.is_tensor(data):
        raise TypeError(
            "data는 PyTorch Tensor여야 합니다."
        )

    if data.dim() != 3:
        raise ValueError(
            "data의 형태는 [B, H, W]여야 합니다. "
            f"현재 형태: {tuple(data.shape)}"
        )

    original_device = data.device

    # OpenCV 연산을 위해 CPU numpy 배열로 변환
    cpu_data = (
        data
        .detach()
        .to("cpu")
        .numpy()
    )

    distance_results = []

    for mask in cpu_data:
        binary_mask = (
            mask > 0
        ).astype(np.uint8)

        distance = cv2.distanceTransform(
            binary_mask,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE
        )

        distance_results.append(
            distance.astype(np.float32)
        )

    if not distance_results:
        return torch.empty(
            data.shape,
            dtype=torch.float32,
            device=original_device
        )

    result_numpy = np.stack(
        distance_results,
        axis=0
    )

    result_tensor = torch.from_numpy(
        result_numpy
    )

    return result_tensor.to(
        device=original_device,
        dtype=torch.float32
    )