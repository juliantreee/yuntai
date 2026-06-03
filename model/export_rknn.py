"""
Convert YOLO ONNX model to RKNN for LubanCat 4 (RK3588).

Usage on LubanCat 4 (or Linux PC with rknn-toolkit2):
  python export_rknn.py

Prerequisites:
  pip install rknn-toolkit2
  # 如果报错, 参考: https://github.com/airockchip/rknn-toolkit2

Before running:
  1. Copy best.onnx to the same directory as this script
  2. Create a 'dataset.txt' file listing calibration images (200-500 images recommended)
     Each line: /path/to/image.jpg
     Use images from your training set or similar scenes.
"""

import os
import sys
from rknn.api import RKNN

# ============================================================
# 配置 - 按需修改
# ============================================================

ONNX_PATH = "best.onnx"
RKNN_PATH = "best.rknn"
DATASET_TXT = "dataset.txt"  # 量化校准图片列表, 每行一个图片路径

# RKNN 平台配置
TARGET = "rk3588"                     # LubanCat 4 使用 RK3588
DO_QUANT = True                       # True=INT8量化, False=FP16
IMG_SIZE = 640

# YOLO 默认预处理: 图像 uint8 [0,255] → float [0,1]
# RKNN 公式: output = (input - mean) / std
MEAN_VALUES = [[0, 0, 0]]            # 不减均值
STD_VALUES = [[255, 255, 255]]       # 除以255, 映射到[0,1]

# I/O 节点名 (与 best.onnx 一致)
INPUT_NAME = "images"
OUTPUT_NAMES = ["output0"]


def main():
    # ---- 1. 创建 RKNN 对象 ----
    rknn = RKNN(verbose=True)

    # ---- 2. 加载 ONNX ----
    print(f"[1/7] Loading ONNX: {ONNX_PATH}")
    ret = rknn.load_onnx(
        model=ONNX_PATH,
        inputs=[INPUT_NAME],
        input_size_list=[[3, IMG_SIZE, IMG_SIZE]],  # CHW 格式 (onnx 为 NCHW)
        outputs=OUTPUT_NAMES,
    )
    if ret != 0:
        print(f"  load_onnx failed, ret={ret}")
        sys.exit(ret)

    # ---- 3. 配置 ----
    print(f"[2/7] Configuring build: target={TARGET}, quant={'INT8' if DO_QUANT else 'FP16'}")
    build_cfg = {
        "target_platform": TARGET,
        "mean_values": MEAN_VALUES,
        "std_values": STD_VALUES,
        "output_optimize": 0,       # 0=不优化输出 (保留原始输出节点便于调试)
    }

    # ---- 4. 混合量化(可选) ----
    # 如果某些层在INT8下精度损失大, 可指定hybrid量化:
    #   rknn.set_hybrid_quantization(quantized_layers=["/model.0/conv/Conv"])
    # 首次转换建议不开启, 先看精度.

    # ---- 5. 构建 RKNN ----
    print(f"[3/7] Building RKNN model...")
    if DO_QUANT and os.path.exists(DATASET_TXT):
        print(f"  Using calibration dataset: {DATASET_TXT}")
        ret = rknn.build(
            do_quantization=True,
            dataset=DATASET_TXT,
            **build_cfg,
        )
    elif DO_QUANT:
        print(f"  WARNING: {DATASET_TXT} not found, falling back to FP16")
        ret = rknn.build(do_quantization=False, **build_cfg)
    else:
        ret = rknn.build(do_quantization=False, **build_cfg)

    if ret != 0:
        print(f"  build failed, ret={ret}")
        sys.exit(ret)

    # ---- 6. 导出 RKNN ----
    print(f"[4/7] Exporting to {RKNN_PATH}")
    ret = rknn.export_rknn(RKNN_PATH)
    if ret != 0:
        print(f"  export_rknn failed, ret={ret}")
        sys.exit(ret)

    # ---- 7. 验证 ----
    print(f"[5/7] Loading exported model for verification...")
    ret = rknn.load_rknn(RKNN_PATH)
    if ret != 0:
        print(f"  load_rknn failed, ret={ret}")
        sys.exit(ret)

    ret = rknn.init_runtime(target=TARGET)
    if ret != 0:
        print(f"  init_runtime failed, ret={ret}")
        # 如果板子不在手边, 可以用模拟器: target=None
        print("  Trying simulator (target=None)...")
        ret = rknn.init_runtime()
        if ret != 0:
            print(f"  simulator init also failed, ret={ret}")
            sys.exit(ret)

    print(f"[6/7] Running inference test...")
    import numpy as np
    dummy_input = np.random.randint(0, 256, (1, 3, IMG_SIZE, IMG_SIZE)).astype(np.uint8)
    outputs = rknn.inference(inputs=[dummy_input])
    for name, arr in zip(OUTPUT_NAMES, outputs):
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, "
              f"range=[{arr.min():.3f}, {arr.max():.3f}]")

    # ---- 精度分析 (可选) ----
    print(f"[7/7] Evaluating accuracy...")
    print("  To verify accuracy against the ONNX model, run:")
    print(f"    python eval_accuracy.py")
    print(f"  (see inline comment below for instructions)")

    print(f"\n✓ RKNN model saved to: {RKNN_PATH}")
    print(f"  Size: {os.path.getsize(RKNN_PATH) / 1024:.1f} KB")
    rknn.release()


if __name__ == "__main__":
    main()