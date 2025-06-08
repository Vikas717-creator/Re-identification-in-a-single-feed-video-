#deep_sort_wrapper.py
from deep_sort_realtime.deepsort_tracker import DeepSort
#here using the deep sort technique which includes class and confidence score 
class DeepSortTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.4,
            nn_budget=None,
            override_track_class=None
        )

    def update(self, detections, frame):
        formatted_detections = []
        for det in detections:
            x, y, w, h = det["bbox"]
            conf = det["confidence"]
            formatted_detections.append(([x, y, w, h], conf, 'player'))

        tracks = self.tracker.update_tracks(formatted_detections, frame=frame)
        return tracks