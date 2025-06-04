import cv2
import torch
from torchvision import transforms
from utils import load_model, predict_emotion

# 图像预处理（和训练时保持一致）
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = load_model('emotion_model.pt', device)

# 加载人脸检测器（Haar Cascade）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("✅ 摄像头已开启，开始识别...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        label, confidence = predict_emotion(face_img, model, device, transform)
        text = f'{label} ({confidence*100:.1f}%)'
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
