# 🤟 Real-Time Sign Language Recognition System

A real-time sign language recognition system that detects and classifies 30 hand gestures using a webcam and machine learning.

---

## 📌 Project Overview

This system recognizes **30 hand gestures** (A–Z alphabets + Hello, Yes, No, Thanks) in real-time using a webcam. It uses MediaPipe for hand landmark detection and a Random Forest classifier for gesture classification.

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.11 | Core programming language |
| OpenCV | Webcam feed & image processing |
| MediaPipe | Hand landmark detection |
| Scikit-learn | Random Forest classifier |
| NumPy | Numerical computing |
| Pandas | Data handling |

---

## 📁 Project Structure

```
signlang/
│
├── data/                  ← Training data (gestures.csv)
├── models/                ← Saved trained model
└── src/
    ├── utils.py           ← Hand landmark extractor
    ├── 1_collect_data.py  ← Data collection script
    ├── 2_train_model.py   ← Model training script
    └── 3_recognize.py     ← Live recognition script
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/harikeshmishra74-ship-it/signlanguage_detection.git
cd signlanguage_detection
```

**2. Create conda environment**
```bash
conda create -n signlang python=3.11
conda activate signlang
```

**3. Install dependencies**
```bash
pip install opencv-python mediapipe==0.10.9 scikit-learn numpy pandas
```

---

## 🚀 How to Run

**Step 1 — Collect training data**
```bash
cd src
python 1_collect_data.py
```

**Step 2 — Train the model**
```bash
python 2_train_model.py
```

**Step 3 — Run live recognition**
```bash
python 3_recognize.py
```

---

## 🎮 Controls

| Key | Action |
|---|---|
| `SPACE` | Add space to sentence |
| `C` | Clear sentence |
| `Q` | Quit |

---

## 📊 How It Works

1. MediaPipe detects **21 hand landmarks** per frame
2. Extracts **63 x,y,z coordinates** as features
3. Random Forest classifier predicts the gesture
4. **12-frame buffer** smooths predictions to avoid flickering
5. Hold a gesture for **1.5 seconds** to add it to the sentence

---

## 👨‍💻 Developer

**Harikesh Mishra**
Computer Science Student — NIST University, Berhampur

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).