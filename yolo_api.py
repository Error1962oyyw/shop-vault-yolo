import pillow_heif
from flask import Flask, jsonify, request
from PIL import Image

from ultralytics import YOLO

app = Flask(__name__)

# 1. 注册 HEIF 插件，让 PIL 支持直接读取 .heic 文件
pillow_heif.register_heif_opener()

# 2. 加载你的 YOLO 模型
print("正在加载 YOLO 模型...")
model = YOLO("yolo26x.pt")
print("YOLO 模型加载完成！")


@app.route("/predict", methods=["POST"])
def predict():
    # 检查是否有文件上传
    if "file" not in request.files:
        return jsonify({"code": 400, "message": "没有找到文件", "labels": []}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"code": 400, "message": "未选择文件", "labels": []}), 400

    try:
        # 3. 使用 PIL 直接读取上传的文件流 (此时 HEIC 会被自动解析)
        try:
            image = Image.open(file.stream)
        except Exception:
            # 如果解析失败（例如传了 txt 文件），直接拦截！
            return jsonify({"code": 400, "message": "不支持的文件格式或图片已损坏", "labels": []}), 400

        # 4. 核心转换：将图片强制转换为 RGB 模式
        # 这一步相当于把它变成了标准的 .jpg 格式，剔除了 HEIC 特有的通道或透明度
        image = image.convert("RGB")

        # (可选) 如果你想在本地保存转换后的 jpg 看看效果，可以取消下面这行的注释：
        # image.save("converted_test.jpg", "JPEG")

        # 5. 将转换好的标准格式直接传给 YOLO 模型（YOLO 原生支持 PIL Image）
        results = model(image)

        # 6. 解析推理结果
        detected_classes = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                detected_classes.append(class_name)

        # 7. 去重并返回
        unique_classes = list(set(detected_classes))
        return jsonify({"code": 200, "message": "识别成功", "labels": unique_classes})

    except Exception as e:
        print(f"识别发生错误: {e}")
        return jsonify({"code": 500, "message": f"服务器内部处理错误: {e!s}", "labels": []}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
