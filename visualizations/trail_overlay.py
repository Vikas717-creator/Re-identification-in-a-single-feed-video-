#trail_overlay.py

import cv2

class TrailOverlay:
    def __init__(self, max_length=5):
        # Tracks ID to get list of (x, y) trail points
        self.trails = {}
        self.max_length = max_length

    def update_trails(self, track_id, bbox):
        x, y, w, h = bbox
        center = (int(x + w / 2), int(y + h / 2))

        if track_id not in self.trails:
            self.trails[track_id] = []

        self.trails[track_id].append(center)

       
        if len(self.trails[track_id]) > self.max_length:
            self.trails[track_id].pop(0)

    def draw_trails(self, frame):
        for trail in self.trails.values():
            for i in range(1, len(trail)):
                if trail[i - 1] and trail[i]:
                    cv2.line(frame, trail[i - 1], trail[i], (0, 255, 255), 1)
