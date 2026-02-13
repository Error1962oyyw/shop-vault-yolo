from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# 1. 加载你的 YOLOv11 模型
# 注意：这里请替换为你自己训练好的权重文件路径，或者官方的 yolo11n.pt
print("正在加载 YOLO 模型...")
model = YOLO("yolo26x.pt")
print("YOLO 模型加载完成！")


@app.route('/predict', methods=['POST'])
def predict():
    # 2. 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({'code': 400, 'message': '没有找到文件', 'labels': []}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'code': 400, 'message': '未选择文件', 'labels': []}), 400

    try:
        # 3. 读取上传的图片文件进内存（不保存到硬盘，速度更快）
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 4. 使用 YOLO 模型进行推理
        results = model(img)

        # 5. 解析推理结果，提取英文标签 (Labels)
        detected_classes = []
        for r in results:
            for box in r.boxes:
                # 获取类别 ID
                class_id = int(box.cls[0])
                # 根据 ID 获取类别名称 (例如: 'cup', 'bottle', 'backpack')
                class_name = model.names[class_id]
                detected_classes.append(class_name)

        # 6. 去重 (如果你拍了两个杯子，只返回一次 'cup')
        unique_classes = list(set(detected_classes))

        # 7. 返回 JSON 给你的 Java 后端
        return jsonify({
            'code': 200,
            'message': '识别成功',
            'labels': unique_classes
        })

    except Exception as e:
        print(f"识别发生错误: {e}")
        return jsonify({'code': 500, 'message': f'服务器内部错误: {str(e)}', 'labels': []}), 500


if __name__ == '__main__':
    # 启动服务，运行在 5000 端口
    app.run(host='0.0.0.0', port=5000, debug=True)
