# End-to-End-Hand_Gesture-Detection-with-OAK-D-From-Annotation-to-Deployment

🚀 Overview

This project implements real-time hand gesture detection using the YOLOv11n model, trained on a custom dataset from Roboflow, and deployed using an OAK-D (OpenCV AI Kit with Depth) device.

📌 Workflow

# Data Collection & Annotation

Hand gesture dataset from https://universe.roboflow.com/lebanese-university-grkoz/hand-gesture-recognition-y5827/dataset/6.

Data annotation and preprocessing performed via Roboflow.

# Model Training

YOLOv11n trained using PyTorch and the coding file name is **OD_Yolov8.ipynb**.

# Model Conversion

PyTorch weights (best.pt) converted to OpenVINO-compatible .blob format by uing this link (https://tools.luxonis.com/).

# Deployment

.blob file deployed on OAK-D device for real-time detection and coresponding code is named as **Optimized_yolov11.py** .

🛠️ Usage

## 📥 Installation

Clone the repository:
```bash
git clone https://github.com/AdnanMunir338/End-to-End-Hand-Gesture-Detection-with-OAK-D-From-Annotation-to-Deployment.git
cd End-to-End-Hand_Gesture-Detection-with-OAK-D-From-Annotation-to-Deployment
```

install dependencies:
```bash
pip install -r Installed_package-list.txt
```
# Results
| Metric           | Value   |
|------------------|---------|
| mAP@0.5          | 89.7 %|
| Precision        |  78.1 % |
| Recall           |  84.3 %|
| FPS (OAK-D)      | 11     |

![image](https://github.com/user-attachments/assets/57dbf46d-f7e9-4d47-bfcb-ddb8bf51c23d)

![image](https://github.com/user-attachments/assets/2cece457-f29c-436c-990c-fa7edc29aaa4)





