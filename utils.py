import torch
import torch.nn as nn
import cv2
from PIL import Image
import numpy as np
from emotion_cnn import EmotionCNN  # 需要你已有的模型定义

# 类别标签（顺序需和训练时一致）
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 加载模型
def load_model(model_path, device):
    model = EmotionCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 预测函数
def predict_emotion(face_image, model, device, transform):
    try:
        # 保留灰度图像
        face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_pil = Image.fromarray(face_gray)  # 不再转为 RGB
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(face_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)
            label = EMOTION_LABELS[pred.item()]
            return label, confidence.item()
    except Exception as e:
        print("❌ 预测失败：", e)
        return "Unknown", 0.0


