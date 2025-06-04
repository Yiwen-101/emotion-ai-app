

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from train_model import EmotionCNN
import numpy as np

print("🚀 启动中...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 摄像头打开失败！")
    exit()

print("✅ 摄像头已打开，开始读取...")

# 设置类别标签
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 图像预处理方式要和训练时保持一致
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN()
model.load_state_dict(torch.load("emotion_model.pt", map_location=device))
model.to(device)
model.eval()

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

print("✅ 摄像头已启动，按 q 键退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 人脸检测（使用OpenCV自带的haar cascade）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face, (48, 48))
        from PIL import Image  # ← 确保这行在文件顶部

        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)  # 灰度 → RGB，才能转换为 PIL Image
        face_pil = Image.fromarray(face_rgb)
        face_tensor = transform(face_pil).unsqueeze(0).to(device)


        with torch.no_grad():
            output = model(face_tensor)
            _, predicted = torch.max(output, 1)
            label = classes[predicted.item()]

        # 绘制预测框
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
