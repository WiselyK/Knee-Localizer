from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import os
import cv2
import numpy as np
from ultralytics import YOLO
import pydicom
import uuid
import zipfile
import io
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['CROPPED_FOLDER'] = 'static/cropped'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'dcm'}
app.config['MODEL_PATH'] = 'models/best.pt'  # Path to your YOLOv8 model

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['CROPPED_FOLDER'], exist_ok=True)

# Load the YOLOv8 model
model = YOLO(app.config['MODEL_PATH'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_to_jpg(file_path):
    """Convert any supported file format to JPG"""
    filename = os.path.basename(file_path)
    extension = filename.rsplit('.', 1)[1].lower()
    
    if extension == 'dcm':
        # Convert DICOM to JPG
        dicom = pydicom.dcmread(file_path)
        img = dicom.pixel_array.astype(float)
        
        # Normalize and convert to uint8
        img = (np.maximum(img, 0) / img.max()) * 255.0
        img = np.uint8(img)
        
        # If it's grayscale, convert to 3-channel
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        # Read regular image format
        img = cv2.imread(file_path)
    
    # Generate JPG path
    jpg_filename = f"{os.path.splitext(filename)[0]}.jpg"
    jpg_path = os.path.join(app.config['UPLOAD_FOLDER'], jpg_filename)
    
    # Save as JPG
    cv2.imwrite(jpg_path, img)
    
    return jpg_path, jpg_filename

def process_image(file_path):
    """Run YOLOv8 model on the image and return results"""
    image = cv2.imread(file_path)
    results = model(image)[0]
    return results, image

def draw_bounding_boxes(image, results):
    """Draw bounding boxes on the image"""
    annotated_image = image.copy()
    
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        if score > 0.5:  # Confidence threshold
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw red rectangle
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    return annotated_image

def crop_knee_joints(image, results, original_filename):
    """Crop knee joints from the image and save them"""
    crops = []
    image_height, image_width = image.shape[:2]
    
    # Sort detected boxes from left to right based on x-coordinate
    boxes = sorted(results.boxes.data.tolist(), key=lambda x: x[0])
    
    for i, result in enumerate(boxes):
        if i >= 2:  # Only process first two detections (left and right knees)
            break
            
        x1, y1, x2, y2, score, class_id = result
        
        if score > 0.5:  # Confidence threshold
            # Calculate square ROI coordinates around the knee joint's center
            side_length = max(x2 - x1, y2 - y1)
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            
            x1_roi = int(x_center - side_length / 2)
            y1_roi = int(y_center - side_length / 2)
            x2_roi = int(x_center + side_length / 2)
            y2_roi = int(y_center + side_length / 2)
            
            # Ensure the ROI coordinates are within the image bounds
            x1_roi = max(0, x1_roi)
            y1_roi = max(0, y1_roi)
            x2_roi = min(image_width, x2_roi)
            y2_roi = min(image_height, y2_roi)
            
            # Crop the ROI
            roi = image[y1_roi:y2_roi, x1_roi:x2_roi]
            
            # Determine if left or right knee based on position in the image
            # For bilateral knee X-rays, usually the left knee is on the right side of the image
            side = "right" if x_center < image_width / 2 else "left"
            
            # Generate unique filename for the crop
            crop_filename = f"{os.path.splitext(original_filename)[0]}_{side}.jpg"
            crop_path = os.path.join(app.config['CROPPED_FOLDER'], crop_filename)
            
            # Save the cropped image
            cv2.imwrite(crop_path, roi)
            
            # Add crop info to the list
            crops.append({
                'path': os.path.join('cropped', crop_filename),
                'filename': crop_filename,
                'side': side,
                'original_filename': original_filename
            })
    
    return crops

@app.route('/')
def index():
    # Clear previous uploads when visiting the home page
    for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'], app.config['CROPPED_FOLDER']]:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No file part'})
    
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected file'})
    
    uploaded_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            # Secure the filename and save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Convert to JPG if necessary
            if not filename.lower().endswith('.jpg'):
                jpg_path, jpg_filename = convert_to_jpg(file_path)
                uploaded_files.append({
                    'original': filename,
                    'converted': jpg_filename,
                    'path': os.path.join('uploads', jpg_filename)
                })
                
                # Remove the original file if it's not a JPG
                if os.path.exists(file_path) and file_path != jpg_path:
                    os.remove(file_path)
            else:
                uploaded_files.append({
                    'original': filename,
                    'converted': filename,
                    'path': os.path.join('uploads', filename)
                })
    
    return jsonify({'files': uploaded_files})

@app.route('/process', methods=['POST'])
def process_images():
    data = request.json
    files = data.get('files', [])
    
    processed_files = []
    
    for file_info in files:
        file_path = os.path.join('static', file_info['path'])
        
        # Process the image with YOLOv8
        results, image = process_image(file_path)
        
        # Draw bounding boxes
        annotated_image = draw_bounding_boxes(image, results)
        
        # Save the annotated image
        annotated_filename = f"annotated_{file_info['converted']}"
        annotated_path = os.path.join(app.config['PROCESSED_FOLDER'], annotated_filename)
        cv2.imwrite(annotated_path, annotated_image)
        
        # Crop knee joints
        crops = crop_knee_joints(image, results, file_info['converted'])
        
        processed_files.append({
            'original': file_info['original'],
            'converted': file_info['converted'],
            'annotated': annotated_filename,
            'annotated_path': os.path.join('processed', annotated_filename),
            'crops': crops
        })
    
    return jsonify({'processed': processed_files})

@app.route('/download', methods=['POST'])
def download_selected():
    data = request.json
    selected_crops = data.get('selectedCrops', [])
    format_type = data.get('format', 'jpg')
    renamed_files = data.get('renamedFiles', {})
    
    # Create a temporary directory for renamed files
    temp_dir = os.path.join(app.config['CROPPED_FOLDER'], 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a memory file for the zip
    memory_file = io.BytesIO()
    
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for crop_path in selected_crops:
            # Get the original crop path and filename
            original_filename = os.path.basename(crop_path)
            full_path = os.path.join('static', crop_path)
            
            # Check if this file has been renamed
            if crop_path in renamed_files:
                new_name = renamed_files[crop_path]
                # Ensure it has the correct extension
                if not new_name.lower().endswith(f'.{format_type}'):
                    new_name = f"{os.path.splitext(new_name)[0]}.{format_type}"
            else:
                # Use the original name but change extension if needed
                new_name = f"{os.path.splitext(original_filename)[0]}.{format_type}"
            
            # If format is different from jpg, convert the image
            if format_type != 'jpg' and os.path.splitext(full_path)[1].lower() != f'.{format_type}':
                img = cv2.imread(full_path)
                new_path = os.path.join(temp_dir, new_name)
                cv2.imwrite(new_path, img)
                zf.write(new_path, new_name)
            else:
                zf.write(full_path, new_name)
    
    # Clean up temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Seek to the beginning of the file
    memory_file.seek(0)
    
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='knee_crops.zip'
    )

@app.route('/results')
def results():
    return render_template('results.html')

# Run the Flask application
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use 5000 locally
    app.run(host='0.0.0.0', port=port, debug=True)
