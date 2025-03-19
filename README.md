# End-to-End-Hand_Gesture-Detection-with-OAK-D-From-Annotation-to-Deployment

🚀 Overview

This project implements real-time hand gesture detection using the YOLOv11n model, trained on a custom dataset from Roboflow, and deployed using an OAK-D (OpenCV AI Kit with Depth) device.

📌 Workflow

Data Collection & Annotation

Hand gesture dataset from Roboflow.

Data annotation and preprocessing performed via Roboflow.

Model Training

YOLOv11n trained using PyTorch.

Model Conversion

PyTorch weights (best.pt) converted to OpenVINO-compatible .blob format.

Deployment

.blob file deployed on OAK-D device for real-time detection.

⚙️ Requirements

Python 3.8+

PyTorch

Ultralytics YOLOv11

OpenVINO Toolkit

DepthAI API

Roboflow

📥 Installation

Clone the repository:

Install dependencies:

🛠️ Usage
