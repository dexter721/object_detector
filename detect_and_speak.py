import cv2
import torch
from gtts import gTTS
import os
import time
from playsound import playsound
import threading

# โหลดโมเดล (เลือกระหว่าง yolov5n.pt และ yolov5s.pt)
model = torch.hub.load('.', 'custom', path='yolov5n.pt', source='local')
# ถ้าอยากแม่นขึ้น เปลี่ยนเป็น:
# model = torch.hub.load('.', 'custom', path='yolov5s.pt', source='local')

# เปิดกล้อง 720p
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# ป้องกันพูดซ้ำเร็วเกินไป
last_speak_time = 0
cooldown = 2  # พูดได้ทุก 2 วิ

def speak_thread(text):
    tts = gTTS(text=text, lang='th')
    filename = "temp.mp3"
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    labels = results.pandas().xyxy[0]['name'].tolist()

    cv2.imshow("Detecting...", results.render()[0])

    # ถ้ามี object และถึงเวลาให้พูด
    if labels and (time.time() - last_speak_time > cooldown):
        obj = labels[0]
        last_speak_time = time.time()
        threading.Thread(target=speak_thread, args=(obj,)).start()

    # กด q เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
