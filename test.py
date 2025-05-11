from filterpy.kalman import KalmanFilter
import numpy as np
import cv2
from ultralytics import YOLO

# --- Kalman Filter Setup ---
def create_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.R *= 10     # Measurement noise
    kf.P *= 1000   # Initial uncertainty
    kf.Q *= 0.01   # Process noise
    return kf

# --- Load YOLO model ---
model = YOLO("yolov8n.pt")

# --- Initialize video capture ---
cap = cv2.VideoCapture(r"C:\Users\kagad\Kalman_filter\VIRAT_S_010204_09_001285_001336.mp4")

# --- Kalman filter init ---
kf = create_kalman_filter()
initialized = False

# --- Create VideoWriter once (after reading the first frame) ---
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read the first frame from the video.")

frame_height, frame_width = frame.shape[:2]
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

# Put the first frame back (optional)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# --- Main loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = results.boxes.xyxy.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()

    # Filter for 'person' class (class_id = 0)
    person_detections = [
        det for det, cls_id in zip(detections, class_ids) if int(cls_id) == 0
    ]

    if person_detections:
        # Choose largest person bbox
        person = max(person_detections, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        x1, y1, x2, y2 = person
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        if not initialized:
            kf.x[:2] = np.array([[cx], [cy]])
            initialized = True
        else:
            kf.predict()
            kf.update(np.array([[cx], [cy]]))

        # Kalman-filtered center
        fx, fy = kf.x[0], kf.x[1]
        cv2.circle(frame, (int(fx), int(fy)), 5, (0, 255, 0), -1)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # Write frame to output video
    out.write(frame)

# --- Cleanup ---
cap.release()
out.release()
print("Tracking complete. Video saved as output.avi")
