# main.py

import os
import cv2
from detector.yolo_detector import YOLODetector
from tracking.deep_sort_wrapper import DeepSortTracker
from tracking.pose_reid import PoseReID
from visualizations.trail_overlay import TrailOverlay
from visualizations.minimap import MiniMap

# Create output directory if not exists
os.makedirs("output_video", exist_ok=True)

# Initializing the  components
detector = YOLODetector("best.pt")
tracker = DeepSortTracker()
reid = PoseReID(similarity_threshold=0.3)
trails = TrailOverlay()
minimap = MiniMap()

# Open video input and output 
cap = cv2.VideoCapture("15sec_input_720p.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_video/output_with_tracking.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Detecting players using YOLOv11
    detections = detector.detect(frame)

    # 2. Tracking players using Deep SORT technique
    tracks = tracker.update(detections, frame)

    # 3. Process each confirmed track
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        bbox = [l, t, r - l, b - t]

        # 4. Pose Re-ID embeddings
        pose_embedding = reid.get_pose_embedding(frame, bbox)
        matched_id = reid.match_pose(pose_embedding)

        if matched_id is not None and matched_id != track_id:
            
            track_id = matched_id

        # here Updating pose memory with latest embeddings
        reid.update_pose_memory(track_id, pose_embedding)

        # 5. Drawing trails and labels
        trails.update_trails(track_id, bbox)
        trails.draw_trails(frame)

        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"Player {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 6. Drawing minimap in left corner
    frame = minimap.draw(frame, tracks)

    # 7. Save and show the output
    out.write(frame)
    cv2.imshow("Player Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
