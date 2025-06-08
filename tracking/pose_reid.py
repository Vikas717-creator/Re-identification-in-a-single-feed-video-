#pose_reid.py

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import cosine
# here creating the pose embeddings
class PoseReID:
    def __init__(self, similarity_threshold=0.3):
        self.pose = mp.solutions.pose.Pose(static_image_mode=True)
        self.saved_poses = {} 
        self.similarity_threshold = similarity_threshold

    def get_pose_embedding(self, frame, bbox):
        x, y, w, h = bbox
        crop = frame[y:y+h, x:x+w]

        if crop.size == 0:
            return None

        results = self.pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        embedding = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks]).flatten()
        return embedding

    def match_pose(self, new_embedding):
        best_id = None
        best_score = float("inf")

        for track_id, saved_embedding in self.saved_poses.items():
            if saved_embedding is None or new_embedding is None:
                continue
            score = cosine(saved_embedding, new_embedding)
            if score < self.similarity_threshold and score < best_score:
                best_score = score
                best_id = track_id

        return best_id

    def update_pose_memory(self, track_id, embedding):
        if embedding is not None:
            self.saved_poses[track_id] = embedding
