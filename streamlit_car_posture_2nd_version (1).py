import streamlit as st
import torch
from ultralytics import YOLO
import torchvision
import cv2
from PIL import Image
import numpy as np
import json
import tempfile
import os

class YOLOv8Ensemble:
    def __init__(self, model_paths):
        self.models = [YOLO(path) for path in model_paths]
        # Assume all models have the same class names
        self.class_names = self.models[0].names
    
    def predict(self, image_path, conf_threshold=0.25, iou_threshold=0.40):
        results = []
        for i, model in enumerate(self.models):
            result = model(image_path, conf=conf_threshold, iou=iou_threshold)[0]
            
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            labels = [self.class_names[int(class_id)] for class_id in class_ids]
            
            # Calculate areas of bounding boxes
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            
            # Find the index of the largest (nearest) object with the highest confidence
            if len(areas) > 0:
                nearest_highest_conf_index = max(range(len(areas)), key=lambda i: (areas[i], confidences[i]))
                
                # Prepare data for the nearest object with highest confidence
                box = boxes[nearest_highest_conf_index]
                confidence = float(confidences[nearest_highest_conf_index])
                label = labels[nearest_highest_conf_index]
                class_id = class_ids[nearest_highest_conf_index]
                
                results.append((box, confidence, class_id))
            
        return self._ensemble_results(results)
    
    def _ensemble_results(self, results):
        if not results:
            return []
        
        # Convert results to tensors
        boxes = torch.tensor([r[0] for r in results])
        scores = torch.tensor([r[1] for r in results])
        classes = torch.tensor([r[2] for r in results])
        
        # Perform Non-Maximum Suppression (NMS)
        keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
        
        final_boxes = boxes[keep]
        final_scores = scores[keep]
        final_classes = classes[keep]
        
        # Group detections by class and find highest confidence for each
        class_detections = {}
        for box, score, cls in zip(final_boxes, final_scores, final_classes):
            cls_id = int(cls.item())
            if cls_id not in class_detections or score > class_detections[cls_id][1]:
                class_detections[cls_id] = (box, score, cls)
        
        return list(class_detections.values())

    def get_class_name(self, class_id):
        return self.class_names[int(class_id)]

    def plot_results(self, image_path, detections):
        image = cv2.imread(image_path)
        for box, confidence, class_id in detections:
            label = self.get_class_name(class_id)
            box = box.int().tolist()
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {confidence:.2f}", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Initialize the ensemble model
@st.cache_resource
def load_model():
    model_paths = ['./best_discussed_with_team_then_trained_14thoct_2024.pt', 
                   './best_TRIAL6.pt', 
                   './best_TRIAL8.pt']
    return YOLOv8Ensemble(model_paths)

ensemble = load_model()

def process_image(image_path):
    detections = ensemble.predict(image_path)
    plotted_image = ensemble.plot_results(image_path, detections)
    
    result_json = []
    for box, score, cls in detections:
        class_name = ensemble.get_class_name(cls.item())
        result_json.append({
            "class": class_name,
            "confidence": float(score),
            "box": box.tolist()
        })
    
    return plotted_image, json.dumps(result_json, indent=2)

# Streamlit app
st.title('Truck Posture Detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        image_path = tmp_file.name

    if st.button('Process Image'):
        # Show progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Processing steps
        steps = ['Loading image', 'Applying ensemble model', 'Generating results']
        for i, step in enumerate(steps):
            status_text.text(f"Step {i+1}/{len(steps)}: {step}")
            progress_bar.progress((i + 1) / len(steps))
            
            if i == 1:  # This is where the actual processing happens
                plotted_image, result_json = process_image(image_path)
        
        status_text.text("Processing complete!")
        progress_bar.empty()

        # Display JSON result
        st.json(result_json)

        # Display the plotted image
        st.image(plotted_image, caption='Detected Truck Part', use_column_width=True)
    
    # Clean up the temporary file
    os.unlink(image_path)
