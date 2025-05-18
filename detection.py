import cv2
import os
import datetime
import threading
import smtplib
from email.mime.text import MIMEText
from ultralytics import YOLO
import numpy as np
import playsound
import requests

# Configuration
EMAIL_ADDRESS = 'ksunidhi866@gmail.com'
EMAIL_PASSWORD = 'sunidhi1221'
RECIPIENTS = ['khushangowda@jnnce.ac.in', 'recipient2@example.com']

VIDEO_OUTPUT_DIR = "fight_videos"
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# Load YOLOv8 pretrained model for person detection
model = YOLO("yolov8n.pt")  # small model, change to yolov8s.pt for better accuracy

def play_alert_sound():
    if os.path.exists("alert.mp3"):
        playsound.playsound("alert.mp3")
    else:
        print("Alert sound file not found.")

def get_location():
    try:
        response = requests.get("https://ipinfo.io")
        data = response.json()
        return data.get("loc", "Unknown")
    except:
        return "Unknown"

def send_email_alert(video_path):
    location = get_location()
    subject = "⚠️ Violence Detected - Women Safety System"
    body = f"Potential violence detected!\nLocation: https://www.google.com/maps?q={location}"
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = ", ".join(RECIPIENTS)
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_ADDRESS, RECIPIENTS, msg.as_string())
        print("✅ Email sent successfully.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

def save_clip(frames, fps):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(VIDEO_OUTPUT_DIR, f"fight_{timestamp}.avi")
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Saved fight clip: {filename}")
    return filename

def bbox_iou(box1, box2):
    # Calculate Intersection Over Union between two bounding boxes
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou

def detect_violent_movement(prev_boxes, curr_boxes, movement_threshold=50, iou_threshold=0.3):
    """
    Simple heuristic:
    - Check if persons move quickly towards each other or have overlapping boxes suddenly.
    - If bounding boxes move more than movement_threshold pixels closer and overlap (IOU > threshold),
      consider potential violence.
    """
    for i, pb in enumerate(prev_boxes):
        for j, cb in enumerate(curr_boxes):
            iou = bbox_iou(pb, cb)
            # Distance between centers
            pb_center = ((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2)
            cb_center = ((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2)
            dist = np.linalg.norm(np.array(pb_center) - np.array(cb_center))
            # If boxes get closer fast and start overlapping
            if i != j and dist < movement_threshold and iou > iou_threshold:
                return True
    return False

def main():
    cap = cv2.VideoCapture("dataset.mp4") # 0 for webcam, or path to video file
    fps = cap.get(cv2.CAP_PROP_FPS) or 20

    pre_fight_frames = []
    fight_frames = []
    fight_recording = False
    fight_timer = 0

    prev_boxes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        curr_boxes = []

        for box in results.boxes.data:
            cls = int(box[5])
            if cls == 0:  # person class
                x1, y1, x2, y2 = map(int, box[:4])
                curr_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Detect violence based on movement heuristic
        fight = False
        if prev_boxes and curr_boxes:
            fight = detect_violent_movement(prev_boxes, curr_boxes)

        if fight:
            cv2.putText(frame, "⚠️ VIOLENCE DETECTED", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        pre_fight_frames.append(frame)
        if len(pre_fight_frames) > int(fps * 5):
            pre_fight_frames.pop(0)

        if fight:
            if not fight_recording:
                fight_recording = True
                fight_frames = pre_fight_frames.copy()
                threading.Thread(target=play_alert_sound, daemon=True).start()
            fight_timer = 0
        elif fight_recording:
            fight_timer += 1
            if fight_timer > int(fps * 3):
                fight_recording = False
                video_path = save_clip(fight_frames, fps)
                threading.Thread(target=send_email_alert, args=(video_path,), daemon=True).start()
            else:
                fight_frames.append(frame)

        prev_boxes = curr_boxes

        cv2.imshow("Violence Detection Cam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
