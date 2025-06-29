#  Player Re-ID Tracker

This project delivers an **multiplayer tracking system** that excels in dynamic sports environments, such as soccer, basketball, or hockey. It fuses **YOLOv11** for lightning-fast detection, **BoT-SORT** for reliable multi-object tracking, and **OSNet Re-Identification** to maintain consistent player identities — even through occlusions, re-entries, and frame exits.

Each player is analyzed using appearance-based deep features, matched against memory, and color-tagged based on **HSV-based jersey classification** (e.g., red or sky blue). Instead of messy bounding boxes, the system uses **minimal, elegant rings around players' feet**, colored by team and labeled with persistent IDs. This results in a clean, professional-grade visual output that enhances understanding, storytelling, and analytics.

For evaluation, we use the industry-standard **py-motmetrics**, and since no ground truth was provided, I meticulously annotated the GT file using **CVAT AI** for fair benchmarking.

Whether you're working in computer vision research, building a sports analytics platform, or enhancing video broadcasting, this system can be your foundation for intelligent player tracking.

---

## Overview

This is a **cutting-edge multi-player tracking system** built for sports analytics. It fuses:

- **YOLOv8** for fast and accurate player detection  
- **BoT-SORT** for robust object tracking  
- **OSNet Re-ID** for appearance-based identity recognition  
- **HSV-based color detection** for team classification (red / sky-blue)  
- **Cosine similarity memory matching** to reduce ID switches and preserve identity  

Even if players leave and re-enter the scene, **we remember them.**

---

## Project Structure
```bash
├── assets/                    # Sample video & ground truth
│   ├── 15sec_input_720p.mp4
│   └── gt.txt
├── models/                  # Detection weights
│   └── best.pt
├── outputs/                 # Output videos and logs
│   ├── reid_output.mp4
│   └── reid_track.txt
├── tracker/                 # All Python logic lives here
│   ├── player_tracker.py    # Core tracking + Re-ID logic
│   ├── main.py              # Entry point script
│   └── evaluation.py        # MOT metric computation
|___ requirements.txt
└── README.md                #  You're reading it!
```
---

## Key Features

| Component               | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
|**Multi-Object Tracking** | YOLOv8 + BoT-SORT for real-time detection and tracking                     |
|**Re-ID Embedding Memory** | OSNet generates 512-D appearance embeddings for robust identity handling  |
| **Team Classification**   | HSV-based jersey color classification (supports red and sky-blue)         |
|**Re-Entry Handling**      | Memory match via cosine similarity + jersey color check                    |
|**Clean Visualization**   | Circular ID rings at player feet replace cluttered bounding boxes          |
|**MOT Evaluation**        | Full metric suite using [py-motmetrics](https://github.com/cheind/py-motmetrics) |

---


### Requirements

```bash
pip install ultralytics opencv-python torch torchvision torchaudio
pip install scikit-learn motmetrics torchreid
```
---

## Run Tracking

```bash
python main.py
```

This processes:

- **Input**: `assets/15sec_input_720p.mp4`  
- **Output**: `outputs/reid_output.mp4`  
- **Track Log**: `outputs/reid_track.txt`

---

## Evaluate Tracking

```bash
python evaluation.py
```

It compares your tracks to the provided ground truth:

- `assets/gt.txt`  
- `outputs/reid_track.txt`

> **Note**: As no ground truth was provided, we created `gt.txt` manually using **CVAT AI** annotation platform based on the original input video. py-motmetrics can only be used with **numpy<=1.26.4**

---

## Performance Metrics
![Evaluation Metrics](https://learnopencv.com/wp-content/uploads/2022/06/05-evaluation-measures.png)


| Metric       | Value | 
|--------------|-------|
| IDF1         | 0.463 | 
| MOTA         | 0.376 | 
| Precision    | 0.683 | 
| Recall       | 0.725 | 
| ID Switches  | 61    |


---

## Technical Highlights

### Hybrid Tracking System

Combines:

- **Motion-based tracking** (BoT-SORT)  
- **Appearance-based Re-ID** (OSNet 512-D vectors)  
- Maintains **separate Kalman filters** and **memory pools**

---

### Re-ID Memory Matching

```python
def match_from_db(self, feat, color, frame_id):
    best_match = max(
        (pid for pid, info in self.appearance_db.items() if info['color'] == color),
        key=lambda pid: cosine_similarity(self.appearance_db[pid]['feat'], feat),
        default=None
    )
    return best_match if similarity > threshold else None
```

---

## Visualization

- IDs shown as **color-coded rings under each player's feet**
- Ring size adjusts based on player distance (height)
- Red team, Sky-blue team

---

## Research Contributions

- Built **evaluation-ready tracker** even without official GT (via CVAT)  
- Reduced **ID switches** using appearance memory + jersey logic  
- Replaced messy bounding boxes with **minimal ring-based UI**  
- Clean MOT integration using **py-motmetrics**

---

## References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- [BoT-SORT Tracker](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)  
- [OSNet Re-ID](https://github.com/KaiyangZhou/deep-person-reid)  
- [CVAT Annotation Tool](https://github.com/opencv/cvat)  
- [MOT Metrics Library](https://github.com/cheind/py-motmetrics)

---

## Future Roadmap

- Integrate **pose heatmaps** for better motion understanding  
- Use **transformer-based attention memory** for long-term tracking  
- Launch a **Streamlit dashboard** for match visualization  
- Automatic **team clustering** via KNN + behavior profiles

