# 🏃‍♂️ Player Re-Identification and Tracking System

A Python-based modular system for real-time **player detection**, **tracking**, and **re-identification** using:
- 🧠 YOLOv8 for object detection
- 🎯 Deep SORT for object tracking
- 🔁 Pose-based Re-ID to maintain player identity

---
# Project Structure
player-reid-system/

├── main.py                       # Main runner script

├── best.pt                       # YOLOv8 trained weights

├── 15sec_input_720p.mp4          # Sample input video

├── detector/
│   └── yolo_detector.py          # YOLOv8-based player detection

├── tracking/
│   ├── deep_sort_wrapper.py      # Deep SORT object tracker
│   └── pose_reid.py              # Pose-based Re-Identification

├── visualizations/
│   ├── trail_overlay.py          # Draws trails for tracked players
│   └── minimap.py                # Optional minimap visualization

├── output/
│   └── output_with_tracking.mp4   # Tracked video output (generated after running)

---
## Input Video Link

https://drive.google.com/file/d/1TDcND31fvEDvcnZCaianTxJrmT8q7iIi/view?usp=drive_link

## Trained Ultralytics YOLOv11 model link

https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view

## Output Video Link

https://drive.google.com/file/d/1FwMpBe-9iAuiXcxMtVraldS3OgTXHHm1/view?usp=sharing


## ⚙️ Setup Instructions

### 1. 🔁 Clone the Repository

git clone https://github.com/your-username/player-reid-system.git

cd player-reid-system

### 2. Optional  🧪 Create Virtual Environment
python -m venv venv

- venv\Scripts\activate  # Windows
#OR
- source venv/bin/activate  # macOS/Linux

### 3. 📦 Install Dependencies
- If you have a requirements.txt, run:

pip install -r requirements.txt

- If not, manually install the required packages:

pip install ultralytics opencv-python torch torchvision numpy

✅ Works on CPU — no GPU needed!

### ▶️ How to Run
- Make sure best.pt and 15sec_input_720p.mp4 are present.

- python main.py

- Output video will be saved at:

output/output_with_tracking.mp4

## 🧠 How It Works
🔍 Step 1: Detection (YOLOv8)
Loads best.pt model
Detects players in each frame

🎯 Step 2: Tracking (Deep SORT)
Assigns consistent IDs to detected players

🔁 Step 3: Re-Identification (Pose Re-ID)
Uses pose embeddings to prevent ID switching

🎨 Step 4: Visualization
Draws bounding boxes, player IDs, trails, and a minimap (optional)

## ❗ Troubleshooting
1.Getting .predict error? → Ensure you're using YOLO(model_path) from ultralytics, not torch.load().

2.Slow on CPU? → Resize frames or use a shorter video.

3.No output? → Check your output/ folder and ensure your video codec is supported.

## 📬 Credits
YOLOv8 by Ultralytics

Deep SORT algorithm by nwojke

Pose-based Re-ID adapted for player identity preservation

## 🔐 License
This project is for academic, research, or demo use. For commercial use, please consult the respective licenses of YOLOv8 and Deep SORT.



