#minimap.py

import cv2

class MiniMap:
    def __init__(self, width=200, height=200, scale_factor=0.2, offset=(10, 10)):
        self.width = width
        self.height = height
        self.scale_factor = scale_factor
        self.offset = offset  # top-left corner for the minimap

    def draw(self, frame, tracks):
        minimap = frame.copy()

        # Background for minimap that is white and black
        cv2.rectangle(minimap, self.offset, 
                      (self.offset[0] + self.width, self.offset[1] + self.height),
                      (0, 0, 0), -1)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x, y, w, h = map(int, track.to_ltrb())  
            cx = int((x + w) / 2 * self.scale_factor)
            cy = int((y + h) / 2 * self.scale_factor)

            cv2.circle(minimap, 
                       (self.offset[0] + cx, self.offset[1] + cy), 
                       5, (255, 255, 255), -1)
            cv2.putText(minimap, str(track_id),
                        (self.offset[0] + cx + 6, self.offset[1] + cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return minimap
