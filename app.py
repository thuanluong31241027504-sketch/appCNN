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
from utils.image_processor import load_image, preprocess_image, crop_food_items

# ============================================
# KIỂM TRA MODEL
# ============================================

def check_model_files():
    """Kiểm tra tất cả file model có thể có"""
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
    """Load model ONNX Runtime"""
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
# HÀM HIỂN THỊ ẢNH CÓ BOUNDING BOX
# ============================================

def draw_boxes(image, boxes):
    """Vẽ bounding boxes lên ảnh để debug"""
    img_copy = image.copy()
    for box in boxes:
        x, y, w, h = box["bbox"]
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Thêm số thứ tự
        cv2.putText(img_copy, str(boxes.index(box) + 1), (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img_copy

# ============================================
# GIAO DIỆN STREAMLIT
# ============================================

st.set_page_config(
    page_title="Food Detection System",
    page_icon="",
    layout="wide"
)

st.title("Food Detection System")
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
    st.caption("He thong nhan dien mon an tu anh")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Tai anh len de nhan dien")
    
    uploaded_file = st.file_uploader(
        "Chon anh mon an",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(image, caption="Anh da tai len", use_column_width=True)
        
        # Nút xử lý
        if st.button("Nhan dien mon an", type="primary"):
            with st.spinner("Dang xu ly anh..."):
                session = load_model()
                
                if session is None:
                    st.error("Khong tim thay model.onnx!")
                else:
                    st.session_state['image'] = image
                    st.session_state['session'] = session
                    st.session_state['processed'] = True
                    st.rerun()

with col2:
    st.subheader("Ket qua nhan dien")
    
    if 'processed' in st.session_state and st.session_state['processed']:
        image = st.session_state['image']
        session = st.session_state['session']
        
        # CẮT ẢNH
        with st.spinner("Dang cat anh thanh cac mon..."):
            cropped_images, boxes = crop_food_items(image)
        
        # DEBUG: HIỂN THỊ ẢNH CÓ BOX
        st.write(f"**Tim thay {len(boxes)} vung anh**")
        
        if len(boxes) > 0:
            # Hiển thị ảnh có bounding boxes
            img_with_boxes = draw_boxes(image, boxes)
            st.image(img_with_boxes, caption="Da phat hien cac mon an", use_column_width=True)
            
            st.success(f"Da phat hien {len(cropped_images)} mon an")
            
            detected_foods = []
            
            # DUYỆT TỪNG ẢNH ĐÃ CẮT
            for idx, cropped_img in enumerate(cropped_images):
                st.markdown(f"**Mon {idx + 1}:**")
                
                col_img, col_info = st.columns([1, 2])
                
                with col_img:
                    try:
                        # Hiển thị ảnh đã cắt
                        if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                            img_display = cv2.resize(cropped_img, (150, 150))
                            st.image(img_display, use_column_width=True)
                        else:
                            st.warning("Anh rong")
                    except Exception as e:
                        st.warning(f"Khong the hien thi anh: {str(e)}")
                        st.image(cropped_img, use_column_width=True)
                
                with col_info:
                    try:
                        # Tiền xử lý
                        preprocessed = preprocess_image(cropped_img, target_size=(224, 224))
                        
                        # Dự đoán
                        input_name = session.get_inputs()[0].name
                        output_name = session.get_outputs()[0].name
                        
                        result = session.run([output_name], {input_name: preprocessed})
                        predictions = result[0]
                        
                        food_id = np.argmax(predictions[0])
                        confidence = np.max(predictions[0])
                        
                        food_name = get_food_name(food_id)
                        food_price = get_food_price(food_id)
                        
                        st.text(f"Ten: {food_name}")
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
                        st.error(f"Loi du doan: {str(e)}")
                        detected_foods.append(0)
                
                st.markdown("---")
            
            # TÍNH TỔNG TIỀN
            if detected_foods:
                total_price = 0
                detail_text = []
                
                for food_id in detected_foods:
                    name = get_food_name(food_id)
                    price = get_food_price(food_id)
                    total_price += price
                    detail_text.append(f"{name}: {price:,} VND")
                
                st.markdown("### Tong tien")
                st.markdown(f"**{total_price:,} VND**")
                
                with st.expander("Xem chi tiet"):
                    for detail in detail_text:
                        st.text(detail)
        else:
            st.warning("Khong tim thay mon an trong anh")
            
            # HIỂN THỊ THÔNG TIN DEBUG
            with st.expander("Debug: Thong tin anh"):
                st.write(f"Kich thuoc anh: {image.shape}")
                st.write(f"So kenh mau: {image.shape[2] if len(image.shape) > 2 else 'Grayscale'}")
                st.write("Thu nguoi dung anh khac hoac dieu chinh anh sang hon.")
        
        if st.button("Lam moi"):
            st.session_state['processed'] = False
            st.rerun()
    
    else:
        st.info("Tai anh len va bam 'Nhan dien mon an'")

# Footer
st.markdown("---")
st.caption("Food Detection System v1.0 - ONNX Runtime")
