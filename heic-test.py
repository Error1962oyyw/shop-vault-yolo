import os
import tempfile

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener  # 确保导入注册函数

from ultralytics import YOLO

# 注册 HEIF 插件（这是关键步骤！）
register_heif_opener()

# 源文件夹（修改为您自定义的图片文件夹路径）
source_dir = r"ultralytics/assets/heic-test"  # ← 修改为您要处理的图片文件夹路径

# 创建临时目录用于内存处理
temp_dir = tempfile.mkdtemp()

try:
    # 处理所有HEIC文件（内存转换，不保存磁盘文件）
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(".heic"):
            heic_path = os.path.join(source_dir, filename)

            # 内存中转换HEIC -> RGB数组
            with Image.open(heic_path) as img:
                rgb_img = img.convert("RGB")
                # 转换为OpenCV可用的BGR格式
                img_array = np.array(rgb_img)[:, :, ::-1]  # RGB->BGR

            # 保存到临时目录（仅用于YOLO输入，后续会删除）
            jpg_path = os.path.join(temp_dir, filename.replace(".heic", ".jpg"))
            cv2.imwrite(jpg_path, img_array)

    # 处理非HEIC文件（直接复制到临时目录）
    for filename in os.listdir(source_dir):
        if not filename.lower().endswith(".heic"):
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(temp_dir, filename)
            # 仅复制非HEIC文件（JPG/PNG等）
            if os.path.isfile(src_path):
                with open(src_path, "rb") as f_src, open(dst_path, "wb") as f_dst:
                    f_dst.write(f_src.read())

    # 执行预测（使用临时目录）
    model = YOLO(r"yolo26x.pt")
    model.predict(
        source=temp_dir,
        save=True,
        show=False,
        project="results_heic",
    )

    print("Processing complete. Results saved to: results_heic")

finally:
    # 清理临时目录（无论是否成功都会删除）
    import shutil

    shutil.rmtree(temp_dir)
    print("Temporary files cleaned up.")
