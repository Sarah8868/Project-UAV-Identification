# UAV Identification System  
Final Project – Software Engineering

## 📌 Description
This project focuses on real-time detection and identification of Unmanned Aerial Vehicles (UAVs) using computer vision and deep learning techniques.  
The system is designed to detect UAVs in images and video streams, track them over time, and provide reliable alerts while reducing false positives caused by objects such as birds or background noise.

The project was developed as a final graduation project in Software Engineering.

---

## 🎯 Project Motivation
With the rapid increase in UAV usage for civilian and military purposes, accurate and early UAV detection has become a significant technological challenge.  
Small UAVs are difficult to detect due to their size, speed, and ability to blend into complex visual environments.  
This project aims to address these challenges using a deep learning–based approach optimized for real-time performance.

---

## 🛠 Technologies & Tools
- **Programming Language:** Python  
- **Deep Learning Frameworks:** PyTorch  
- **Model:** YOLOv8 (You Only Look Once)  
- **Computer Vision:** OpenCV  
- **Tracking Algorithm:** Kalman Filter  
- **Data Handling & Visualization:** Matplotlib  
- **Development Environments:** VS Code, Jupyter Notebook  
- **Hardware (optional / edge deployment):** ESP32, Camera module, Jetson Nano / Raspberry Pi  

---

## ✨ Key Features
- Real-time UAV detection in images and video streams
- Custom-trained YOLOv8 model for UAV recognition
- Kalman Filter integration for stable tracking across video frames
- Reduction of false positives caused by birds or moving background objects
- Support for both image-based and video-based detection
- Modular and extensible system architecture

---

## 🧠 What I Implemented
- Research and comparison of multiple UAV detection approaches
- Selection and justification of YOLOv8 for real-time detection
- Creation and organization of a custom UAV dataset (train / validation / test)
- Data cleaning and augmentation (rotation, lighting changes, noise)
- Training and evaluation of the deep learning model
- Implementation of Kalman Filter–based tracking
- Video frame processing and result visualization
- Performance evaluation using accuracy, precision, and tracking metrics

---

## ▶️ How It Works (High-Level Flow)
1. Input image or video stream is received
2. Frames are processed using YOLOv8 for UAV detection
3. Detected UAVs are tracked using a Kalman Filter
4. Detection results are visualized and logged
5. Alerts can be triggered when a UAV is identified


