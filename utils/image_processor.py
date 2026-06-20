import numpy as np
import cv2

def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def preprocess_image(img, target_size=(224, 224)):
    img_resized = cv2.resize(img, target_size)
    img_array = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
    return img_array

def crop_food_items(image):
    img = cv2.resize(image, (1400, 1300))
    
    regions = {
        "canhrau": img[40:700, 40:760],
        "comtrang": img[40:700, 820:1380],
        "rausong": img[760:1280, 30:500],
        "cakho": img[760:1280, 520:920],
        "thitkho": img[760:1280, 950:1380]
    }
    
    cropped_results = []
    for i, (name, crop) in enumerate(regions.items()):
        if crop.size > 0:
            cropped_results.append({
                "id": i + 1,
                "name": name,
                "image": crop
            })
    
    return cropped_results, img

def draw_boxes_fixed(img, results):
    img_copy = img.copy()
    
    positions = {
        1: (40, 700, 40, 760),
        2: (40, 700, 820, 1380),
        3: (760, 1280, 30, 500),
        4: (760, 1280, 520, 920),
        5: (760, 1280, 950, 1380)
    }
    
    for result in results:
        idx = result["id"]
        if idx in positions:
            y1, y2, x1, x2 = positions[idx]
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img_copy, f"Tray {idx}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return img_copy
