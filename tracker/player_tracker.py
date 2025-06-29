import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchreid.reid.utils import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


class PlayerReIDTracker:
    def __init__(self,
                 det_model_path="models/best.pt",
                 osnet_model_name="osnet_ain_x1_0"):

        self.det_model = YOLO(det_model_path)

        # ReID extractor using OSNet
        self.extractor = FeatureExtractor(
            model_name=osnet_model_name,
            model_path=None,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.memory_embeddings = {}  # {id: appearance_vector}
        self.jersey_colors = {}      # {id: "red" / "blue"}
        self.color_map = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0)
        }

    def get_jersey_color(self, crop):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        mask_red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        mask_red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        mask_blue = cv2.inRange(hsv, (85, 50, 50), (130, 255, 255))

        red_score = (mask_red1 + mask_red2).sum()
        blue_score = mask_blue.sum()

        return "red" if red_score >= blue_score else "blue"

    def get_osnet_embedding(self, img):
        resized = cv2.resize(img, (256, 128))
        embedding = self.extractor([resized])[0]
        return embedding

    def draw_player_id(self, frame, pid, x1, y1, x2, y2, jersey_color):
        center_bottom = (int((x1 + x2) / 2), int(y2))
        player_height = y2 - y1
        ring_radius = max(15, int(player_height * 0.15))
        ring_thickness = max(2, int(ring_radius * 0.2))

        ring_color = self.color_map.get(jersey_color, (255, 255, 255))
        cv2.circle(frame, center_bottom, ring_radius, ring_color, ring_thickness)
        cv2.circle(frame, center_bottom, ring_radius - int(ring_thickness / 2), (255, 255, 255), 1)

        text = str(pid)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = center_bottom[0] - text_size[0] // 2
        text_y = center_bottom[1] + text_size[1] // 2
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def track_video(self, video_path, output_path, save_txt_path="outputs/track.txt"):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(os.path.dirname(save_txt_path), exist_ok=True)

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        frame_id = 0
        tracking_results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            results = self.det_model.track(source=frame, persist=True, save=False)[0]

            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                ids = results.boxes.id.cpu().numpy().astype(int)

                for box, pid in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    w, h = x2 - x1, y2 - y1
                    crop = frame[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    # Get features and jersey color
                    emb = self.get_osnet_embedding(crop)
                    jersey_color = self.get_jersey_color(crop)

                    # Check memory for consistent ID
                    if pid in self.memory_embeddings:
                        sim = cosine_similarity(
                            self.memory_embeddings[pid].reshape(1, -1),
                            emb.reshape(1, -1)
                        )[0][0]

                        if sim < 0.6 or self.jersey_colors.get(pid) != jersey_color:
                            self.memory_embeddings[pid] = emb
                            self.jersey_colors[pid] = jersey_color
                    else:
                        self.memory_embeddings[pid] = emb
                        self.jersey_colors[pid] = jersey_color

                    # Save MOT format
                    out_line = f"{frame_id},{pid},{x1},{y1},{w},{h},1,-1,-1\n"
                    tracking_results.append(out_line)

                    # Visual: circle + ID
                    self.draw_player_id(frame, pid, x1, y1, x2, y2, jersey_color)

            out.write(frame)

        cap.release()
        out.release()

        with open(save_txt_path, "w") as f:
            f.writelines(tracking_results)
