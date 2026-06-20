import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import sys
import onnxruntime as ort

# Thêm đường dẫn để import module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.menu import MENU, get_food_name, get_food_price, calculate_total
from utils.image_processor import load_image, preprocess_image, crop_food_items, draw_boxes_fixed

# ============================================
# KIỂM TRA MODEL
# ============================================

def check_model_files():
    model_files = []
    possible_names = ['model.onnx', 'trainaicuoiky1.onnx']
    
    for name in possible_names:
        if os.path.exists(name):
            size = os.path.getsize(name) / (1024*1024)
            model_files.append({
                'name': name,
                'size': size,
                'path': os.path.abspath(name)
            })
    
    return model_files

@st.cache_resource
def load_model():
    model_files = check_model_files()
    if not model_files:
        return None
    
    onnx_files = [f for f in model_files if f['name'].endswith('.onnx')]
    if onnx_files:
        chosen = onnx_files[0]
        try:
            session = ort.InferenceSession(chosen['path'])
            return session
        except Exception as e:
            st.error(f"Loi load ONNX: {str(e)}")
            return None
    
    return None

# ============================================
# HÀM CẮT ẢNH THEO TỌA ĐỘ NHẬP
# ============================================

def crop_with_custom_coords(image, coords):
    """
    Cắt ảnh theo tọa độ tùy chỉnh
    coords: dict với key là tên khay, value là (y1, y2, x1, x2)
    """
    # Resize ảnh về 1400x1300
    img_resized = cv2.resize(image, (1400, 1300))
    
    cropped_results = []
    
    for khay_id, (y1, y2, x1, x2) in coords.items():
        # Kiểm tra tọa độ hợp lệ
        if y1 < img_resized.shape[0] and y2 <= img_resized.shape[0] and \
           x1 < img_resized.shape[1] and x2 <= img_resized.shape[1]:
            
            cropped_img = img_resized[y1:y2, x1:x2]
            
            if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                cropped_results.append({
                    "id": khay_id,
                    "name": f"Khay {khay_id}",
                    "image": cropped_img,
                    "bbox": (x1, y1, x2 - x1, y2 - y1)
                })
    
    return cropped_results, img_resized

def draw_boxes_custom(image, cropped_results):
    """Vẽ bounding boxes lên ảnh"""
    img_copy = image.copy()
    
    for result in cropped_results:
        x1, y1, w, h = result["bbox"]
        cv2.rectangle(img_copy, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        cv2.putText(img_copy, f"Khay {result['id']}", (x1 + 5, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return img_copy

# ============================================
# GIAO DIỆN STREAMLIT
# ============================================

st.set_page_config(
    page_title="Food Detection System",
    page_icon="",
    layout="wide"
)

st.title("Food Detection System - Dieu Chinh Vi Tri Cat")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Menu")
    for item in MENU:
        note = f" ({item['note']})" if item['note'] else ""
        st.text(f"{item['id']}. {item['name']} - {item['price']:,} VND{note}")
    
    st.markdown("---")
    
    model_files = check_model_files()
    if model_files:
        st.success("Model da san sang")
        for mf in model_files:
            st.text(f"{mf['name']} ({mf['size']:.1f}MB)")
    else:
        st.error("Chua co model.onnx")
    
    st.markdown("---")
    st.caption("Dieu chinh toa do cat cho tung khay")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Tai anh len")
    
    uploaded_file = st.file_uploader(
        "Chon anh mon an",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(image, caption="Anh goc", use_column_width=True)
        
        # Nút xử lý
        if st.button("Cat anh va dieu chinh", type="primary"):
            st.session_state['image'] = image
            st.session_state['step'] = 'adjust'
            st.rerun()

with col2:
    st.subheader("Dieu chinh toa do cat")
    
    # BƯỚC 1: ĐIỀU CHỈNH TỌA ĐỘ
    if 'step' in st.session_state and st.session_state['step'] == 'adjust':
        image = st.session_state['image']
        
        # Resize ảnh để hiển thị
        img_resized = cv2.resize(image, (1400, 1300))
        h, w = img_resized.shape[:2]
        
        st.info(f"Kich thuoc anh sau resize: {w} x {h}")
        
        # Tạo các thanh trượt cho từng khay
        st.subheader("Điều chỉnh tọa độ cho 5 khay")
        
        # Khởi tạo giá trị mặc định
        default_coords = {
            1: (40, 700, 40, 760),    # (y1, y2, x1, x2)
            2: (40, 700, 820, 1380),
            3: (760, 1280, 30, 500),
            4: (760, 1280, 520, 920),
            5: (760, 1280, 950, 1380)
        }
        
        # Lấy giá trị từ session state hoặc dùng mặc định
        if 'coords' not in st.session_state:
            st.session_state['coords'] = default_coords.copy()
        
        coords = st.session_state['coords']
        
        # Tạo tab cho từng khay
        tabs = st.tabs(["Khay 1", "Khay 2", "Khay 3", "Khay 4", "Khay 5"])
        
        for idx, tab in enumerate(tabs, 1):
            with tab:
                st.markdown(f"**Điều chỉnh khay {idx}**")
                
                col_y, col_x = st.columns(2)
                
                with col_y:
                    y1 = st.slider(
                        f"Y1 (trên)",
                        min_value=0,
                        max_value=h-100,
                        value=coords[idx][0],
                        key=f"y1_{idx}"
                    )
                    y2 = st.slider(
                        f"Y2 (dưới)",
                        min_value=y1+50,
                        max_value=h,
                        value=coords[idx][1],
                        key=f"y2_{idx}"
                    )
                
                with col_x:
                    x1 = st.slider(
                        f"X1 (trái)",
                        min_value=0,
                        max_value=w-100,
                        value=coords[idx][2],
                        key=f"x1_{idx}"
                    )
                    x2 = st.slider(
                        f"X2 (phải)",
                        min_value=x1+50,
                        max_value=w,
                        value=coords[idx][3],
                        key=f"x2_{idx}"
                    )
                
                # Cập nhật tọa độ
                coords[idx] = (y1, y2, x1, x2)
                
                # Hiển thị preview
                preview = img_resized[y1:y2, x1:x2]
                if preview.shape[0] > 0 and preview.shape[1] > 0:
                    st.image(preview, caption=f"Preview Khay {idx}", use_column_width=True)
        
        # Lưu tọa độ
        st.session_state['coords'] = coords
        
        # Nút preview tất cả
        if st.button("Preview tat ca khay", type="primary"):
            cropped_results, img_preview = crop_with_custom_coords(image, coords)
            
            if cropped_results:
                # Hiển thị ảnh có bounding boxes
                img_with_boxes = draw_boxes_custom(img_preview, cropped_results)
                st.image(img_with_boxes, caption="Cac khay da cat", use_column_width=True)
                
                st.session_state['cropped_results'] = cropped_results
                st.session_state['coords'] = coords
                st.session_state['step'] = 'predict'
                st.rerun()
            else:
                st.error("Khong cat duoc anh. Vui long kiem tra toa do!")
        
        if st.button("Dat lai mac dinh"):
            st.session_state['coords'] = default_coords.copy()
            st.rerun()
    
    # BƯỚC 2: DỰ ĐOÁN
    elif 'step' in st.session_state and st.session_state['step'] == 'predict':
        cropped_results = st.session_state['cropped_results']
        
        st.success(f"Da cat thanh {len(cropped_results)} khay")
        
        session = load_model()
        if session is None:
            st.error("Khong tim thay model.onnx!")
        else:
            detected_foods = []
            
            for result in cropped_results:
                cropped_img = result["image"]
                khay_id = result["id"]
                
                st.markdown(f"**Khay {khay_id}:**")
                
                col_img, col_info = st.columns([1, 2])
                
                with col_img:
                    try:
                        h, w = cropped_img.shape[:2]
                        scale = min(150/h, 150/w)
                        new_h, new_w = int(h*scale), int(w*scale)
                        img_display = cv2.resize(cropped_img, (new_w, new_h))
                        st.image(img_display, use_column_width=True)
                    except:
                        st.image(cropped_img, use_column_width=True)
                
                with col_info:
                    try:
                        preprocessed = preprocess_image(cropped_img, target_size=(224, 224))
                        
                        input_name = session.get_inputs()[0].name
                        output_name = session.get_outputs()[0].name
                        
                        result_onnx = session.run([output_name], {input_name: preprocessed})
                        predictions = result_onnx[0]
                        
                        food_id = np.argmax(predictions[0])
                        confidence = np.max(predictions[0])
                        
                        food_name = get_food_name(food_id)
                        food_price = get_food_price(food_id)
                        
                        st.text(f"Du doan: {food_name}")
                        st.text(f"Gia: {food_price:,} VND")
                        st.text(f"Do tin cay: {confidence*100:.1f}%")
                        
                        if confidence > 0.8:
                            st.success("Tin cay cao")
                        elif confidence > 0.5:
                            st.warning("Tin cay trung binh")
                        else:
                            st.error("Tin cay thap")
                        
                        detected_foods.append(food_id)
                        
                    except Exception as e:
                        st.error(f"Loi: {str(e)}")
                        detected_foods.append(0)
                
                st.markdown("---")
            
            # TÍNH TỔNG TIỀN
            if detected_foods:
                total_price = 0
                detail_text = []
                
                for idx, food_id in enumerate(detected_foods):
                    name = get_food_name(food_id)
                    price = get_food_price(food_id)
                    total_price += price
                    detail_text.append(f"Khay {idx+1}: {name} - {price:,} VND")
                
                st.markdown("### Tong tien")
                st.markdown(f"**{total_price:,} VND**")
                
                with st.expander("Xem chi tiet"):
                    for detail in detail_text:
                        st.text(detail)
        
        if st.button("Quay lai dieu chinh"):
            st.session_state['step'] = 'adjust'
            st.rerun()
    
    else:
        st.info("Tai anh len va bam 'Cat anh va dieu chinh'")

# Footer
st.markdown("---")
st.caption("Food Detection System v1.0 - Dieu chinh toa do cat")
