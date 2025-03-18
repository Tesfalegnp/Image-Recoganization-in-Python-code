 # YOLOv8 Custom Object Detection

This repository provides a step-by-step guide to training YOLOv8 on a custom dataset for object detection.

## **1. Install Dependencies**
Ensure you have Python and the necessary libraries installed. Run the following command:

```bash
pip install ultralytics opencv-python matplotlib tqdm
```

## **2. Prepare Your Dataset**
The dataset should be in YOLO format:

### **Folder Structure**
```
dataset/
│── images/
│   ├── train/  (training images)
│   ├── val/    (validation images)
│   ├── test/   (test images)
│── labels/
│   ├── train/  (training labels)
│   ├── val/    (validation labels)
│   ├── test/   (test labels)
│── data.yaml  (dataset configuration file)
```

### **Label Format (YOLO format)**
Each `.txt` file inside `labels/` contains:
```
<class_id> <x_center> <y_center> <width> <height>
```
Example:
```
0 0.5 0.5 0.4 0.6
```

### **Create `data.yaml`**
This file tells YOLO where to find the dataset.

```yaml
path: ./dataset
train: images/train
val: images/val
test: images/test

names:
  0: "cat"
  1: "dog"
  2: "bird"
```

## **3. Train YOLOv8**
Create a new Python script `train.py` and add the following code:

```python
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use yolov8s.pt or yolov8m.pt for larger models

# Train the model
model.train(data="dataset/data.yaml", epochs=50, imgsz=640)
```

Run the script:
```bash
python train.py
```

## **4. Test the Trained Model**
Once training is complete, test it on new images:

```python
# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run detection on a test image
results = model("test.jpg")

# Show results
results.show()
```

## **5. Next Steps**
- Fine-tune hyperparameters like learning rate and batch size.
- Convert the model for real-time applications.
- Deploy the model in a mobile or web application.

---
### **References**
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [YOLO Dataset Preparation](https://roboflow.com/)

**Happy Coding! 🚀**


