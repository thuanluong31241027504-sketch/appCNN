import cv2
import numpy as np
from PIL import Image
import streamlit as st

def load_image(image_file):
    """Đọc ảnh từ file upload"""
    img = Image.open(image_file)
    return np.array(img)

def preprocess_image(image, target_size=(224, 224)):
    """Tiền xử lý ảnh cho model CNN"""
    # Chuyển đổi sang RGB nếu cần
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize ảnh
    img_resized = cv2.resize(image, target_size)
    
    # Chuẩn hóa
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Thêm batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def detect_food_boxes(image):
    """
    Phát hiện các khay thức ăn trong ảnh
    Trả về danh sách các vùng ảnh cắt được
    """
    # Chuyển sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Làm mờ để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Tìm edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Tìm contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc các contours
    food_boxes = []
    min_area = 5000  # Diện tích tối thiểu
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            # Cắt vùng ảnh
            food_region = image[y:y+h, x:x+w]
            food_boxes.append({
                "bbox": (x, y, w, h),
                "image": food_region,
                "area": area
            })
    
    # Sắp xếp theo vị trí từ trái sang phải
    food_boxes.sort(key=lambda box: box["bbox"][0])
    
    return food_boxes

def crop_food_items(image):
    """Cắt từng món ăn từ ảnh gốc"""
    # Phát hiện các khay thức ăn
    boxes = detect_food_boxes(image)
    
    # Trả về danh sách ảnh đã cắt
    cropped_images = [box["image"] for box in boxes]
    
    return cropped_images, boxes
