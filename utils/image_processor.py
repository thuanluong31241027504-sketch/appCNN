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
    Cắt ảnh theo tọa độ cố định cho từng vị trí khay
    Đầu tiên resize ảnh về 1400x1300
    """
    # RESIZE ẢNH VỀ 1400x1300
    img_resized = cv2.resize(image, (1400, 1300))
    
    # Lấy kích thước
    h, w = img_resized.shape[:2]
    
    # Định nghĩa các vùng cắt theo tọa độ [y1:y2, x1:x2]
    # Dựa trên thông số từ ảnh mẫu đã resize
    regions = [
        {
            "id": 1,
            "name": "Khay 1 (Canh rau)",
            "y1": 40, "y2": 700,
            "x1": 40, "x2": 760
        },
        {
            "id": 2,
            "name": "Khay 2 (Com trang)",
            "y1": 40, "y2": 700,
            "x1": 820, "x2": 1380
        },
        {
            "id": 3,
            "name": "Khay 3 (Rau song)",
            "y1": 760, "y2": 1280,
            "x1": 30, "x2": 500
        },
        {
            "id": 4,
            "name": "Khay 4 (Ca kho)",
            "y1": 760, "y2": 1280,
            "x1": 520, "x2": 920
        },
        {
            "id": 5,
            "name": "Khay 5 (Thit kho)",
            "y1": 760, "y2": 1280,
            "x1": 950, "x2": 1380
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
    
    for result in cropped_results:
        x1, y1, w, h = result["bbox"]
        # Vẽ box màu xanh
        cv2.rectangle(img_copy, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 3)
        # Thêm số thứ tự và tên
        label = f"Khay {result['id']}"
        cv2.putText(img_copy, label, (x1 + 5, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return img_copy
