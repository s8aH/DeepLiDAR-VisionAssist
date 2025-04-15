# LiDAR-Based 3D Object Detection with PointPillars

### Real-Time Environmental Awareness for the Visually Impaired

This project presents a LiDAR-based 3D object detection system designed to assist visually impaired individuals in real-time navigation. By leveraging the **PointPillars** architecture, we built a fast and efficient detection model that converts LiDAR point clouds into 3D bounding boxes. The system integrates **audio feedback**, enabling users to receive spoken descriptions of nearby obstacles, including distance, direction, and object class.

---

## 📷 Sample Outputs

🎥 [Click here to view the demo video](https://drive.google.com/file/d/1kJScnVlS8XourOvelULTHqV6_wJT1BV7/view?usp=sharing)

---

## 🚀 Features

- Real-time 3D object detection using PointPillars
- Trained on **KITTI**, fine-tuned on **Lyft (converted to KITTI format)**
- Optimized for **mobility applications on iOS** with LiDAR (e.g., iPhone Pro series)
- Integrated **gTTS-based audio feedback** for enhanced accessibility
- Evaluation includes 2D, BEV, and 3D metrics with orientation accuracy (AOS)

---

## 🙏 Acknowledgements

- This project builds on the [PointPillars implementation by zhulf0804](https://github.com/zhulf0804/PointPillars).

---

## 🛠 Setup

### Prerequisites
- Python 3.8+
- PyTorch
- OpenPCDet (or custom PointPillars implementation)
- NumPy, Matplotlib
- gTTS (for audio feedback)
- CUDA-compatible GPU

### Installation
```bash
git clone https://github.com/your-repo/pointpillars-lidar-assistive.git
cd pointpillars-lidar-assistive
pip install -r requirements.txt
```

---

## 📦 Datasets

- **KITTI**: [Link](https://www.cvlibs.net/datasets/kitti/)
- **Lyft (nuScenes format)**: [Kaggle](https://www.kaggle.com/competitions/3d-object-detection-for-autonomous-vehicles)

> Lyft dataset was converted to KITTI format with relabeled classes:
```
car, truck, bus → Car  
bicycle, motorcycle → Cyclist  
pedestrian → Pedestrian
```

---

## 📈 Training & Fine-Tuning

### Training on KITTI:
```bash
python train.py --cfg configs/pointpillars_kitti.yaml
```

### Fine-tuning on Lyft (converted):
```bash
python train.py --cfg configs/pointpillars_lyft.yaml --pretrained_model checkpoints/kitti.pth
```

- Optimizer: AdamW
- LR: 2.5e-4 with OneCycleLR
- Epochs: 160
- Batch size: 6
- Mixed precision + gradient clipping

---

## 🎯 Evaluation

Metrics:  
- **2D BBox**, **BEV**, **3D BBox**, and **AOS** (Average Orientation Similarity)  
- AP@0.5 for pedestrian/cyclist, AP@0.7 for cars

| Metric         | Gain from Fine-Tuning |
|----------------|------------------------|
| BEV BBox (Easy) | +0.12 AP |
| 3D BBox (Moderate) | +0.07 AP |
| AOS (Easy) | +0.05 AP |

---

## 🔊 Audio Feedback Module

Each detected object is translated into natural language via `gTTS`:
> “Pedestrian, 8.3 meters, right, 87% confidence”

```python
from audio_feedback import speak_detection
speak_detection(class_name="Car", distance=10.5, direction="left", confidence=0.91)
```

---

## 🧠 Future Work

- Add more diverse datasets for broader generalization
- Extend to additional object categories
- Explore domain adaptation or transfer learning
- Include object tracking and trajectory prediction

---

## 📜 Citation

If you find this work useful, please cite:

```
@misc{han2025pointpillarsassistive,
  title={LiDAR and Deep Learning for Object Detection and Environment Description for the Visually Impaired},
  author={Sooa Han, Alex Cao},
  year={2025},
  note={Course Project}
}
```
