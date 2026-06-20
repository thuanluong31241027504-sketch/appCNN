import cv2
import numpy as np
from PIL import Image

def load_image(image_file):
    """Đọc ảnh từ file upload"""
    img = Image.open(image_file)
    return np.array(img)

def preprocess_image(image, target_size=(224, 224)):
    """Tiền xử lý ảnh cho model ONNX"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    img_resized = cv2.resize(image, target_size)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def crop_food_items_fixed(image):
    """
    Cắt ảnh theo tọa độ cố định đã điều chỉnh
    Đầu tiên resize ảnh về 1400x1300
    """
    # RESIZE ẢNH VỀ 1400x1300
    img_resized = cv2.resize(image, (1400, 1300))
    
    # Định nghĩa các vùng cắt theo tọa độ đã điều chỉnh
    # Format: (y1, y2, x1, x2)
    regions = [
        {
            "id": 1,
            "name": "Khay 1",
            "y1": 0, "y2": 715,
            "x1": 64, "x2": 687
        },
        {
            "id": 2,
            "name": "Khay 2",
            "y1": 52, "y2": 723,
            "x1": 808, "x2": 1307
        },
        {
            "id": 3,
            "name": "Khay 3",
            "y1": 760, "y2": 1206,
            "x1": 30, "x2": 461
        },
        {
            "id": 4,
            "name": "Khay 4",
            "y1": 760, "y2": 1229,
            "x1": 472, "x2": 897
        },
        {
            "id": 5,
            "name": "Khay 5",
            "y1": 749, "y2": 1247,
            "x1": 892, "x2": 1315
        }
    ]
    
    cropped_results = []
    
    for region in regions:
        # Cắt ảnh theo tọa độ
        y1, y2 = region["y1"], region["y2"]
        x1, x2 = region["x1"], region["x2"]
        
        # Kiểm tra tọa độ hợp lệ
        if y1 < img_resized.shape[0] and y2 <= img_resized.shape[0] and \
           x1 < img_resized.shape[1] and x2 <= img_resized.shape[1]:
            
            cropped_img = img_resized[y1:y2, x1:x2]
            
            # Kiểm tra ảnh cắt có rỗng không
            if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                cropped_results.append({
                    "id": region["id"],
                    "name": region["name"],
                    "image": cropped_img,
                    "bbox": (x1, y1, x2 - x1, y2 - y1)
                })
    
    return cropped_results, img_resized

def crop_food_items(image):
    """Wrapper function cho tương thích với code cũ"""
    return crop_food_items_fixed(image)

def draw_boxes_fixed(image, cropped_results):
    """Vẽ bounding boxes lên ảnh theo tọa độ cố định"""
    img_copy = image.copy()
    
    # Màu sắc cho từng khay
    colors = [
        (0, 255, 0),    # Xanh lá - Khay 1
        (255, 0, 0),    # Đỏ - Khay 2
        (0, 0, 255),    # Xanh dương - Khay 3
        (255, 255, 0),  # Vàng - Khay 4
        (255, 0, 255)   # Tím - Khay 5
    ]
    
    for idx, result in enumerate(cropped_results):
        x1, y1, w, h = result["bbox"]
        color = colors[idx % len(colors)]
        
        # Vẽ box
        cv2.rectangle(img_copy, (x1, y1), (x1 + w, y1 + h), color, 3)
        
        # Thêm số thứ tự và tên
        label = f"Khay {result['id']}"
        cv2.putText(img_copy, label, (x1 + 5, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return img_copy
