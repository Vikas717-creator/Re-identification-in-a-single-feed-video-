#main.py
import cv2
from detector.yolo_detector import YOLODetector
from tracking.deep_sort_wrapper import DeepSortTracker
from tracking.pose_reid import PoseReID
from visualizations.trail_overlay import TrailOverlay
from visualizations.minimap import MiniMap

# Initializing the components which are the other files
detector = YOLODetector("best.pt")
tracker = DeepSortTracker()
reid = PoseReID()
trails = TrailOverlay()
minimap = MiniMap()

# Video input and output for the opening and writing of the video
cap = cv2.VideoCapture("15sec_input_720p.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output/output_with_tracking.mp4", fourcc, 30.0, (1280, 720))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Detecting the  players in input video
    detections = detector.detect(frame)

    # 2. Tracking using Deep SORT technique
    tracks = tracker.update(detections, frame)

    # 3. Drawing the tracks through lines
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        bbox = [l, t, r - l, b - t]

        # 4. Pose Re-ID: prevent ID reuse so that no same id assigned to other player
        pose_embedding = reid.get_pose_embedding(frame, bbox)
        matched_id = reid.match_pose(pose_embedding)

        if matched_id is None or matched_id != track_id:
            reid.update_pose_memory(track_id, pose_embedding)

        # 5. Trail update & draw the rectangle and that is with id
        trails.update_trails(track_id, bbox)
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"Player {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    trails.draw_trails(frame)

    # 6. Minimap creation for giving a game like feel
    frame = minimap.draw(frame, tracks)

    # 7. Output shown
    out.write(frame)
    cv2.imshow("Player Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows() 