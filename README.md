# 🐦 Bird Counting & Weight Estimation System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-purple?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green?style=flat-square&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=flat-square)

> An end-to-end computer vision system for real-world poultry farm management — detects, tracks, counts, and estimates the weight of birds from video footage using YOLOv8 object detection, persistent multi-object tracking, and a production-ready FastAPI backend.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture & Pipeline](#system-architecture--pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Usage](#api-usage)
- [JSON Output Format](#json-output-format)
- [Weight Estimation Logic](#weight-estimation-logic)
- [Performance Optimizations](#performance-optimizations)
- [Accuracy & Stability Analysis](#accuracy--stability-analysis)
- [Results](#results)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## 🔍 Overview

Manual bird counting in large poultry farms is inaccurate, time-consuming, and operationally expensive. A farm with thousands of birds cannot be monitored frame-by-frame by humans.

**This system automates the entire process:**

- Accepts raw poultry farm video as input
- Detects every bird in every frame using YOLOv8
- Assigns persistent unique IDs to each bird and tracks them across frames
- Counts birds per frame and total unique birds across the full video
- Estimates approximate weight per bird using bounding box area analysis
- Generates annotated output videos and structured JSON analytics reports
- Exposes everything via a clean REST API for integration into farm management systems

Built for **real-world conditions** — handles noise (eggs, roofs, cages), long video durations, and variable lighting through intelligent frame cropping and skipping strategies.

---

## ✅ Key Features

| Feature | Description |
|---|---|
| 🎯 YOLOv8 Detection | Accurate bird detection using pretrained YOLOv8n weights |
| 🔁 Persistent Tracking | Unique track IDs maintained across frames — no double counting |
| 🔢 Frame-wise Counting | Bird count logged per frame throughout the video |
| 📊 Total Unique Count | Aggregated count of distinct birds in the entire video |
| ⚖️ Weight Estimation | Bounding-box-area-based weight approximation per bird |
| ✂️ Region Cropping | Top 30% of frame removed to eliminate farm noise |
| ⏩ Frame Skipping | Processes every 5th frame for speed without accuracy loss |
| 🎬 Annotated Video | Output video with bounding boxes, IDs, and confidence scores |
| 📁 JSON Analytics | Structured per-frame and summary data for downstream use |
| 🌐 REST API | FastAPI backend for video upload, processing, and result retrieval |
| 📈 Stability Analysis | Per-video detection coverage and track persistence metrics |

---

## 🏗️ System Architecture & Pipeline

```
┌─────────────────────────────────────────┐
│           Video Upload (FastAPI)         │
│         POST /process-video/             │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         Frame Preprocessing             │
│  • Crop top 30% (remove noise)          │
│  • Skip every 5th frame (performance)   │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      YOLOv8 Detection & Tracking        │  ← detector_tracker.py
│  • COCO class: bird                     │
│  • Persistent tracking (unique IDs)     │
│  • Confidence threshold filtering       │
│  • Bounding box extraction              │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│          Bird Analyzer                  │  ← analyzer.py
│  • Per-frame bird counts                │
│  • Total unique bird aggregation        │
│  • Track ID management                  │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│        Weight Estimation                │  ← weight.py
│  • bbox_area × scale_factor             │
│  • Per-bird weight in kg                │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│     Output Generation                   │  ← utils.py
│  • Annotated video (bboxes + IDs)       │
│  • JSON analytics report                │
└─────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Object Detection** | YOLOv8 (Ultralytics), YOLOv8n pretrained weights |
| **Multi-Object Tracking** | YOLOv8 built-in ByteTrack / BotSort persistent tracker |
| **Image Processing** | OpenCV (cv2), NumPy |
| **Backend API** | FastAPI, Uvicorn |
| **Data Serialization** | JSON, Pydantic |
| **Visualization** | OpenCV annotation (bboxes, IDs, confidence) |
| **Analysis** | Custom accuracy & stability analysis scripts |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
Bird_counting-and-weight_estimation/
│
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app — upload endpoint & routing
│   ├── detector_tracker.py      # YOLOv8 detection + tracking + frame cropping
│   ├── analyzer.py              # Bird counting, unique ID aggregation, analytics
│   ├── weight.py                # Bounding-box-based weight estimation logic
│   └── utils.py                 # Drawing bboxes, IDs, confidence on frames
│
├── data/
│   └── videos/                  # Raw input poultry farm videos
│
├── uploads/                     # API-processed outputs
│   ├── poultry1.mp4
│   ├── annotated_poultry1.mp4
│   ├── poultry1_results.json
│   └── ...
│
├── outputs/
│   ├── detections/              # Offline test JSON results
│   └── experiment/              # Accuracy & stability analysis results
│
├── trials/
│   └── accuracy.py              # Experimental accuracy scripts
│
├── test_detector.py             # Local testing script (no API needed)
├── requirements.txt
├── yolov8n.pt                   # YOLOv8 nano pretrained weights
├── poultry1.json                # Sample output JSON
├── poultry2_results.json        # Sample output JSON
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```
Python 3.10+
pip or conda
```

### 1. Clone the Repository

```bash
git clone https://github.com/Jalagamdolu/Bird_counting-and-weight_estimation.git
cd Bird_counting-and-weight_estimation
```

### 2. Create Virtual Environment (Recommended)

```bash
conda create -n birdcount python=3.10
conda activate birdcount
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI Server

```bash
uvicorn app.main:app --reload
```

Server runs at: `http://127.0.0.1:8000`
Swagger UI: `http://127.0.0.1:8000/docs`

---

## 🌐 API Usage

### Endpoint

```
POST /process-video/
Content-Type: multipart/form-data
```

### Input

| Field | Type | Description |
|---|---|---|
| `file` | `.mp4` video | Raw poultry farm footage |

### Output

| File | Description |
|---|---|
| `annotated_<name>.mp4` | Video with bounding boxes, track IDs, confidence scores |
| `<name>_results.json` | Full JSON analytics — per-frame counts, weights, summary |

Both files are automatically saved to the `uploads/` directory.

### Example (cURL)

```bash
curl -X POST "http://127.0.0.1:8000/process-video/" \
  -H "accept: application/json" \
  -F "file=@data/videos/poultry1.mp4"
```

---

## 📋 JSON Output Format

```json
{
  "summary": {
    "total_unique_birds": 5,
    "total_frames_with_birds": 120,
    "frame_wise_counts": {
      "0": 3,
      "5": 4,
      "10": 5
    }
  },
  "frames": {
    "0": [
      {
        "track_id": 1,
        "bbox": [120, 85, 230, 190],
        "confidence": 0.87,
        "estimated_weight_kg": 1.4
      },
      {
        "track_id": 2,
        "bbox": [310, 100, 420, 210],
        "confidence": 0.91,
        "estimated_weight_kg": 1.6
      }
    ]
  }
}
```

---

## ⚖️ Weight Estimation Logic

Bird weight is estimated from bounding box area using a proportional scale factor:

```python
bbox_area = (x2 - x1) * (y2 - y1)
estimated_weight_kg = bbox_area * scale_factor
```

**Rationale:** In a fixed-camera farm environment, a bird's apparent size in frame is proportional to its physical size. Larger bounding boxes indicate heavier birds. The scale factor is calibrated to the specific camera setup and bird breed.

> This provides a **consistent relative estimate** suitable for flock analytics, sorting, and farm management — not a precision scale replacement.

---

## ⚡ Performance Optimizations

### 1. Frame Cropping (Noise Reduction)
```
Original frame height:  720px
Cropped frame height:   504px  (top 30% removed)
```
Removes distracting elements at the top of the frame:
- Eggs on conveyor belts
- Roof structures
- Hatchery equipment
- Lighting fixtures

### 2. Frame Skipping
- Processes **every 5th frame** instead of every frame
- Reduces total processing time by ~80%
- Tracking IDs persist across skipped frames — no accuracy loss on total counts

### 3. Persistent Multi-Object Tracking
- YOLOv8's built-in tracker assigns stable IDs across frames
- Prevents the same bird from being counted multiple times
- Handles partial occlusion and re-entry into frame

---

## 📊 Accuracy & Stability Analysis

Run the analysis script:

```bash
python trials/accuracy.py
```

**What it computes:**

| Metric | Description |
|---|---|
| Detection Coverage | `frames_with_birds / total_frames` |
| Track Persistence | Average frames per unique track ID |
| Avg Birds Per Frame | Mean bird count across all frames |
| Frame Height Verification | Original vs. cropped height confirmation |

**Example output:**

```json
{
  "video": "poultry3_results.json",
  "total_frames": 18612,
  "frames_with_birds": 13200,
  "detection_coverage": 0.71,
  "track_lengths": {
    "1": 5000,
    "2": 4800
  },
  "average_birds_per_frame": 4.5,
  "original_frame_height": 720,
  "cropped_frame_height": 504
}
```

Results saved to `outputs/experiment/` as per-video JSON files.

---

## 🏁 Local Testing (Without API)

```bash
python test_detector.py
```

Useful for:
- Debugging YOLOv8 detection on a single video
- Verifying tracking IDs are persistent
- Testing frame cropping and skipping parameters
- Offline experimentation without spinning up the API server

---

## 📈 Results

| Metric | Value |
|---|---|
| Detection Model | YOLOv8n (nano — optimized for speed) |
| Detection Coverage | ~71% of frames contain detected birds |
| Average Birds Per Frame | 4.5 birds |
| Track Persistence | ~5,000 frames per unique bird |
| Frame Processing Speed | Every 5th frame (~5x speed boost) |
| Noise Reduction | Top 30% crop eliminates false positives |
| Output Formats | Annotated MP4 + structured JSON |

---

## 💡 Key Learnings

- **Custom preprocessing beats generic pipelines** — the top-30% crop was the single biggest improvement to reduce false detections in cluttered farm environments. Domain-specific preprocessing is more valuable than model complexity.
- **Frame skipping + persistent tracking** is more efficient than processing every frame — tracking IDs survive across skipped frames, so count accuracy is maintained at a fraction of the compute cost.
- **Bounding box area as a proxy metric** works well for relative estimation in fixed-camera setups — the same principle applies to cell size estimation in microscopy imaging.
- **Modular architecture** (detector, analyzer, weight, utils as separate modules) made debugging and iteration dramatically faster.

---

## 🔮 Future Improvements

- [ ] Real-world calibrated weight models using known reference objects in frame
- [ ] Bird re-identification across camera cuts and multi-camera setups
- [ ] GPU batch inference for real-time farm deployment
- [ ] Frontend dashboard with live video feed and analytics charts
- [ ] Replace YOLOv8n with YOLOv8m/l for higher accuracy on dense flocks
- [ ] Export weight reports to CSV for farm management software integration
- [ ] Docker containerization for one-command deployment

---

## 👨‍💻 Author

**Jalagam Dolender**
AI Engineer | Computer Vision | Deep Learning | Python Developer

- 🔗 [GitHub](https://github.com/Jalagamdolu)
- 💼 [LinkedIn](https://www.linkedin.com/in/jalagam-dolender-8a45b7236)
- 📧 dolujalagam@gmail.com

---

## 📄 License

This project is licensed under the MIT License.

---

> *"The same pipeline that counts birds in a farm counts cells under a microscope — the only difference is the domain."*
