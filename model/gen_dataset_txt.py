"""
Generate dataset.txt for RKNN calibration from training images.
Run this on Windows to prepare the calibration image list,
then copy dataset.txt + images to LubanCat 4.

Or run directly on LubanCat 4 if the images are accessible there.
"""

import os
import glob


# ============================================================
# 配置
# ============================================================

# 训练集图片目录 (Windows 路径)
IMAGE_DIR = r"F:\YOLO_MY\make_dataset\rectangle\images\train"

# 输出文件
OUTPUT = r"dataset.txt"

# 最多使用多少张图片做校准 (200-500 足够)
MAX_IMAGES = 300


def main():
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(IMAGE_DIR, "**", ext), recursive=True))
        images.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

    if not images:
        print(f"ERROR: No images found in {IMAGE_DIR}")
        print("Please update IMAGE_DIR to point to your training images.")
        return

    images = sorted(set(images))[:MAX_IMAGES]
    print(f"Found {len(images)} images, using first {min(len(images), MAX_IMAGES)}")

    with open(OUTPUT, "w") as f:
        for img in images:
            # 写 Linux 路径 (如果要在鲁班猫4上跑, 需要改成对应的路径)
            f.write(img + "\n")

    print(f"Saved to: {OUTPUT}")

    # 提示: 如果要在鲁班猫4上跑, 路径需要更新
    print(f"\nNOTE: 如果在鲁班猫4上运行 export_rknn.py, 请把图片拷贝过去并更新 dataset.txt 中的路径.")
    print(f"      或者在鲁班猫4上重新运行本脚本 (修改 IMAGE_DIR 为板子上的路径).")


if __name__ == "__main__":
    main()