import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

def create_kalman_filter(cx: float, cy: float, frame_idx: int, track_id: int):
    """Initialize a 4D constant-velocity Kalman filter for a new track."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.R *= 10     # measurement noise
    kf.P *= 1000   # initial covariance
    kf.Q *= 0.01   # process noise
    kf.x[:2, 0] = [cx, cy]
    return {
        'kf': kf,
        'id': track_id,
        'last_seen': frame_idx,
        'age': 0,
        'bbox_wh': None,  # width & height of last real detection
    }

def main():
    # --- Setup ---
    model = YOLO("yolov8n.pt")
    cap   = cv2.VideoCapture(r"C:\Users\kagad\Kalman_filter\VIRAT_S_010204_09_001285_001336.mp4")
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    # Prepare output writer
    fps    = cap.get(cv2.CAP_PROP_FPS)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out    = cv2.VideoWriter('tracked_output.avi', fourcc, fps, (w, h))

    # Tracking state
    tracks    = []   # list of dicts: each has 'kf', 'id', 'last_seen', 'age', 'bbox_wh'
    next_id   = 0    # next track identifier
    max_gap   = 40   # keep “dead” tracks for up to 40 frames
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Run person detection
        results      = model(frame)[0]
        dets         = results.boxes.xyxy.cpu().numpy()  # Nx4 array
        cls_ids      = results.boxes.cls.cpu().numpy()   # N array
        # keep only class 0 = person
        person_boxes = [b for b, c in zip(dets, cls_ids) if int(c) == 0]
        centers      = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in person_boxes]

        # 1a) Draw detection centers (blue crosses)
        for cx, cy in centers:
            cv2.drawMarker(
                frame,
                (int(cx), int(cy)),
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
            )

        # Map from track_id to the bbox it was matched with this frame
        detections_for_track = {}

        # 2) Predict all existing tracks
        for tr in tracks:
            tr['kf'].predict()

        # 3) Build cost matrix and solve assignment
        if tracks and centers:
            preds   = np.array([[t['kf'].x[0, 0], t['kf'].x[1, 0]] for t in tracks])  # T×2
            det_arr = np.array(centers)                                             # D×2
            cost    = np.linalg.norm(preds[:, None, :] - det_arr[None, :, :], axis=2)
            t_idx, d_idx = linear_sum_assignment(cost)
        else:
            t_idx = np.array([], dtype=int)
            d_idx = np.array([], dtype=int)

        assigned_tracks = set()
        assigned_dets   = set()

        # 4) Update matched tracks
        for ti, di in zip(t_idx, d_idx):
            if cost[ti, di] < 50:  # threshold in pixels
                tr = tracks[ti]
                tr['kf'].update(np.array(centers[di]))
                tr['age'] = 0
                tr['last_seen'] = frame_idx
                assigned_tracks.add(ti)
                assigned_dets.add(di)
                x1, y1, x2, y2 = person_boxes[di]
                tr['bbox_wh'] = (x2 - x1, y2 - y1)
                detections_for_track[tr['id']] = (x1, y1, x2, y2)

        # 5) Age & prune old tracks
        new_tracks = []
        for i, tr in enumerate(tracks):
            if i not in assigned_tracks:
                tr['age'] += 1
            # only keep if seen within last max_gap frames
            if frame_idx - tr['last_seen'] <= max_gap:
                new_tracks.append(tr)
        tracks = new_tracks

        # 6) Revive or spawn for unmatched detections
        for j, center in enumerate(centers):
            if j in assigned_dets:
                continue

            # look for the best “dead” track to revive
            best_tr, best_d = None, float('inf')
            for tr in tracks:
                if frame_idx - tr['last_seen'] > 0:  # only dead ones
                    px, py = tr['kf'].x[0, 0], tr['kf'].x[1, 0]
                    d = np.hypot(px - center[0], py - center[1])
                    if d < best_d:
                        best_d, best_tr = d, tr

            if best_tr is not None and best_d < 50:
                # revive
                best_tr['kf'].update(np.array(center))
                best_tr['age'] = 0
                best_tr['last_seen'] = frame_idx
                x1, y1, x2, y2 = person_boxes[j]
                best_tr['bbox_wh'] = (x2 - x1, y2 - y1)
                detections_for_track[best_tr['id']] = (x1, y1, x2, y2)
            else:
                # new track
                tid = next_id
                tr  = create_kalman_filter(center[0], center[1], frame_idx, tid)
                x1, y1, x2, y2 = person_boxes[j]
                tr['bbox_wh'] = (x2 - x1, y2 - y1)
                tracks.append(tr)
                detections_for_track[tid] = (x1, y1, x2, y2)
                next_id += 1

        # 7) Draw ghost boxes, KF centers, and IDs for all tracks
        for tr in tracks:
            # always draw the KF-predicted center
            px, py = tr['kf'].x[0, 0], tr['kf'].x[1, 0]
            cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)

            # draw a “ghost” bbox of the same size as last detection
            if tr['bbox_wh'] is not None:
                w, h = tr['bbox_wh']
                x1_pred = int(px - w / 2)
                y1_pred = int(py - h / 2)
                x2_pred = int(px + w / 2)
                y2_pred = int(py + h / 2)
                cv2.rectangle(frame,
                              (x1_pred, y1_pred),
                              (x2_pred, y2_pred),
                              (0, 255, 0), 1)

            # draw the track ID near the predicted box
            cv2.putText(frame,
                        f"id={tr['id']}",
                        (int(px - w / 2), int(py - h / 2) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

        # 8) Write annotated frame to output video
        out.write(frame)
        frame_idx += 1

    # cleanup
    cap.release()
    out.release()
    print("Tracking video saved to tracked_output.avi")

if __name__ == "__main__":
    main()
