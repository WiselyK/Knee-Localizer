# Knee Joint Localization Interface (YOLOv8)

This project provides a web-based interface for uploading knee X-ray images, using a trained YOLOv8 model to automatically detect and crop knee joints. It is part of an ongoing research effort on automating knee OA (osteoarthritis) analysis.

## 🚀 Features

- ✅ Upload knee X-ray images (JPG, PNG, etc.)
- ✅ Localize knee joints using a YOLOv8 object detection model
- ✅ Automatically crop the localized knee regions
- ✅ View and download the cropped images
- ✅ Lightweight Flask-based backend

## 🧠 Model

This project uses a fine-tuned [YOLOv8](https://github.com/ultralytics/ultralytics) model trained on annotated knee X-ray datasets. The model predicts bounding boxes around the knee joints with high accuracy and speed.

## 🖼️ Example Workflow

1. Upload an X-ray image.
2. The YOLOv8 model detects the knee joint(s).
3. Bounding boxes are drawn and knees are cropped.
4. Cropped outputs are displayed and made downloadable.

## 🛠️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/WiselyK/Knee-Localizer.git
cd Knee-Localizer
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```

## 📝 License
This project is for academic and research use. Contact the author if you'd like to collaborate or reuse the code/model.

## 👤 Author
Wisely Koay
Graduate Researcher, Malaysia
wisely.framer.website