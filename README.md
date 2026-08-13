# 📍 Visual Localization System

**GPS-free positioning using computer vision, deep learning, and unsupervised clustering.**

A machine learning pipeline that estimates a user's location from camera images alone — built for environments where GPS fails: indoors, underground, low-light conditions, and dense urban canyons.

---

## 🎯 Overview

Traditional GPS breaks down in many real-world scenarios — inside buildings, underground transit systems, parking garages, and dense city blocks. This project solves that problem by using **visual features from the environment itself** to determine position, combining a CNN, an autoencoder, and a Self-Organizing Map (SOM) into a complete localization pipeline, served through a lightweight web application.

---

## ✨ Key Features

- 🖼️ **Camera-based localization** — no GPS hardware required
- 🌙 **Low-light robustness** — works with nighttime and poorly lit images
- 🧠 **CNN feature extraction** — learns environmental patterns (buildings, roads, landmarks)
- 📉 **Autoencoder compression** — reduces high-dimensional features into a compact latent space
- 🗺️ **SOM-based clustering** — groups similar environments for location matching
- 🌐 **Web interface** — upload an image and get an instant location estimate on an interactive map

---

## 🏗️ How It Works

```
Camera Image
     │
     ▼
┌─────────────────────┐
│  1. Preprocessing    │  OpenCV — cleans & prepares raw images
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  2. CNN Extraction   │  TensorFlow — extracts visual features
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  3. Autoencoder      │  Compresses features into latent space
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  4. SOM Clustering   │  MiniSom — groups similar environments
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  5. Location Match   │  Compares clusters against known map data
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  6. Web Visualization│  Flask — displays estimated position
└─────────────────────┘
```

| Stage | Description |
|---|---|
| **Image Collection** | Captures images from a camera or smartphone, including low-light conditions |
| **Feature Extraction (CNN)** | Identifies key visual patterns — buildings, roads, unique landmarks |
| **Dimensionality Reduction (Autoencoder)** | Compresses features into a compact latent representation |
| **Clustering (SOM)** | Groups visually similar environments on a 2D grid |
| **Localization** | Matches clustered features to a predefined location database |
| **Visualization** | Displays the estimated position on an interactive map |

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python |
| **Image Processing** | OpenCV |
| **Deep Learning** | TensorFlow (CNN + Autoencoder) |
| **Clustering** | MiniSom (Self-Organizing Maps) |
| **Backend** | Flask |
| **Frontend** | HTML, CSS, JavaScript |

---

## 📂 Project Structure

```
visual-localization-system/
│
├── app/
│   ├── index.py                # Flask entry point
│   ├── static/
│   │   ├── images/             # Reference images
│   │   ├── uploads/             # User-uploaded images
│   │   ├── css/
│   │   └── js/
│   └── templates/
│       ├── index.html
│       └── result.html
│
├── models/                      # CNN, autoencoder & SOM logic
├── notebooks/                   # Experiments & training notebooks
│   ├── afterSOM.ipynb
│   └── CompleteSom.ipynb
├── data/                        # Raw & processed datasets
├── saved_models/                # Trained model weights
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash

# Install dependencies
pip install -r requirements.txt

# Run the app
python app/index.py
```

Then open `http://localhost:5000` in your browser and upload an image to see your estimated location.

---

## 📈 Impact

- ✅ **Reliable** — works where GPS signals are weak or absent
- ✅ **Versatile** — handles indoor, outdoor, and low-light conditions
- ✅ **Cost-effective** — no specialized GPS hardware needed
- ✅ **Scalable** — suited for smart cities, malls, and underground transit systems
- ✅ **Real-time capable** — supports live camera feed localization

---

## 🔮 Future Enhancements

- 🔗 Sensor fusion with accelerometer/gyroscope data for improved accuracy
- 📊 Larger training datasets for better generalization across environments
- ⏱️ Real-time location updates as the user moves

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

**Tauqeer Ahmad Qureshi**
AI/ML Engineer — Agentic Systems, RAG & Computer Vision
📧 tauqeerqureshi112@gmail.com