import cv2
import numpy as np
from PIL import Image

def load_image(image_file):
    """Đọc ảnh từ file upload"""
    img = Image.open(image_file)
    return np.array(img)

def preprocess_image(image, target_size=(224, 224)):
    """Tiền xử lý ảnh cho model ONNX"""
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

def detect_food_boxes_advanced(image):
    """
    Phát hiện các khay thức ăn trong ảnh - CẢI TIẾN
    Sử dụng nhiều phương pháp để tìm khay thức ăn
    """
    # Chuyển sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Áp dụng các phương pháp khác nhau
    boxes = []
    
    # PHƯƠNG PHÁP 1: Canny Edge Detection
    edges = cv2.Canny(gray, 30, 150)
    contours1, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # PHƯƠNG PHÁP 2: Threshold
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours2, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # PHƯƠNG PHÁP 3: Adaptive Threshold
    thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
    contours3, _ = cv2.findContours(thresh_adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # PHƯƠNG PHÁP 4: Phát hiện hình chữ nhật
    # Tìm các hình chữ nhật trong ảnh
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, rect_kernel)
    _, thresh_rect = cv2.threshold(morph, 50, 255, cv2.THRESH_BINARY)
    contours4, _ = cv2.findContours(thresh_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Gộp tất cả contours
    all_contours = []
    all_contours.extend(contours1)
    all_contours.extend(contours2)
    all_contours.extend(contours3)
    all_contours.extend(contours4)
    
    # Lọc và hợp nhất các contours
    min_area = 2000  # Giảm ngưỡng để bắt được nhiều hơn
    max_area = image.shape[0] * image.shape[1] * 0.5  # Không quá 50% ảnh
    
    # Lọc contours theo diện tích
    filtered_contours = []
    for contour in all_contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Kiểm tra tỉ lệ khung hình (gần hình chữ nhật)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 0.3 < aspect_ratio < 3.0:  # Tỉ lệ hợp lý
                filtered_contours.append(contour)
    
    # Hợp nhất các contours gần nhau
    merged_boxes = []
    used = [False] * len(filtered_contours)
    
    for i, contour1 in enumerate(filtered_contours):
        if used[i]:
            continue
        
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        merged_x, merged_y, merged_w, merged_h = x1, y1, w1, h1
        
        for j, contour2 in enumerate(filtered_contours[i+1:], i+1):
            if used[j]:
                continue
            
            x2, y2, w2, h2 = cv2.boundingRect(contour2)
            
            # Kiểm tra khoảng cách giữa 2 box
            if (abs(x1 - x2) < 50 and abs(y1 - y2) < 50) or \
               (abs((x1 + w1) - (x2 + w2)) < 50 and abs((y1 + h1) - (y2 + h2)) < 50):
                # Hợp nhất
                merged_x = min(merged_x, x2)
                merged_y = min(merged_y, y2)
                merged_w = max(merged_x + merged_w, x2 + w2) - merged_x
                merged_h = max(merged_y + merged_h, y2 + h2) - merged_y
                used[j] = True
        
        used[i] = True
        merged_boxes.append((merged_x, merged_y, merged_w, merged_h))
    
    # Tạo danh sách các vùng ảnh đã cắt
    food_boxes = []
    for (x, y, w, h) in merged_boxes:
        # Thêm padding để không bị cắt mất viền
        pad = 5
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image.shape[1] - x, w + 2*pad)
        h = min(image.shape[0] - y, h + 2*pad)
        
        # Cắt vùng ảnh
        food_region = image[y:y+h, x:x+w]
        
        # Kiểm tra vùng ảnh có đủ lớn không
        if food_region.shape[0] > 20 and food_region.shape[1] > 20:
            food_boxes.append({
                "bbox": (x, y, w, h),
                "image": food_region,
                "area": w * h
            })
    
    # Sắp xếp theo vị trí từ trái sang phải
    food_boxes.sort(key=lambda box: box["bbox"][0])
    
    return food_boxes

def crop_food_items(image):
    """Cắt từng món ăn từ ảnh gốc"""
    boxes = detect_food_boxes_advanced(image)
    cropped_images = [box["image"] for box in boxes]
    return cropped_images, boxes
