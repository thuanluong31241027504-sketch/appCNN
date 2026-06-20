import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import sys
import tensorflow as tf

# Thêm đường dẫn để import module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.menu import MENU, get_food_name, get_food_price, calculate_total
from utils.image_processor import load_image, preprocess_image, crop_food_items

# Cấu hình trang
st.set_page_config(
    page_title="Food Detection System",
    page_icon="",
    layout="wide"
)

# Tiêu đề chính
st.title("Food Detection System")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    """Load model CNN đã train"""
    # Tìm file model
    model_paths = [
        'trainaicuoiky1.keras',
        'model/trainaicuoiky1.keras',
        'trainaicuoiky.keras',
        'model.keras'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                st.success(f"Da load model thanh cong: {path}")
                return model
            except Exception as e:
                st.error(f"Loi load model {path}: {str(e)}")
                continue
    
    # Nếu không tìm thấy model
    st.error("Khong tim thay file model (trainaicuoiky1.keras)")
    return None

# Sidebar
with st.sidebar:
    st.header("Menu")
    st.markdown("**Danh sach mon an:**")
    
    # Hiển thị menu
    for item in MENU:
        note = f" ({item['note']})" if item['note'] else ""
        st.text(f"{item['id']}. {item['name']} - {item['price']:,} VND{note}")
    
    st.markdown("---")
    st.caption("He thong nhan dien mon an tu anh")
    
    # Thông tin model
    if os.path.exists('trainaicuoiky1.keras'):
        size = os.path.getsize('trainaicuoiky1.keras') / (1024*1024)
        st.success(f"Model: trainaicuoiky1.keras ({size:.1f} MB)")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Tai anh len de nhan dien")
    
    uploaded_file = st.file_uploader(
        "Chon anh mon an",
        type=['jpg', 'jpeg', 'png'],
        help="Tai anh chua khay thuc an de he thong nhan dien"
    )
    
    if uploaded_file is not None:
        # Đọc và hiển thị ảnh
        image = load_image(uploaded_file)
        st.image(image, caption="Anh da tai len", use_column_width=True)
        
        # Nút xử lý
        if st.button("Nhan dien mon an", type="primary"):
            with st.spinner("Dang xu ly anh..."):
                model = load_model()
                if model is None:
                    st.error("Khong the load model. Vui long kiem tra file trainaicuoiky1.keras")
                else:
                    st.session_state['image'] = image
                    st.session_state['model'] = model
                    st.session_state['processed'] = True
                    st.rerun()

with col2:
    st.subheader("Ket qua nhan dien")
    
    if 'processed' in st.session_state and st.session_state['processed']:
        image = st.session_state['image']
        model = st.session_state['model']
        
        # Cắt các món ăn
        cropped_images, boxes = crop_food_items(image)
        
        if len(cropped_images) > 0:
            st.success(f"Da phat hien {len(cropped_images)} mon an")
            
            detected_foods = []
            
            for idx, cropped_img in enumerate(cropped_images):
                st.markdown(f"**Mon {idx + 1}:**")
                
                col_img, col_info = st.columns([1, 2])
                
                with col_img:
                    try:
                        img_display = cv2.resize(cropped_img, (150, 150))
                        st.image(img_display, use_column_width=True)
                    except:
                        st.image(cropped_img, use_column_width=True)
                
                with col_info:
                    try:
                        # Tiền xử lý
                        preprocessed = preprocess_image(cropped_img, target_size=(224, 224))
                        
                        # Dự đoán
                        predictions = model.predict(preprocessed, verbose=0)
                        food_id = np.argmax(predictions[0])
                        confidence = np.max(predictions[0])
                        
                        # Lấy thông tin
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
                        st.error(f"Loi khi du doan: {str(e)}")
                        detected_foods.append(0)
                
                st.markdown("---")
            
            # Tính tổng tiền
            if detected_foods:
                total_price, details = calculate_total(detected_foods)
                
                st.markdown("### Tong tien")
                st.markdown(f"**{total_price:,} VND**")
                
                with st.expander("Xem chi tiet"):
                    for detail in details:
                        st.text(f"{detail['name']}: {detail['price']:,} VND")
        else:
            st.warning("Khong tim thay mon an trong anh")
            st.info("Vui long tai anh khac hoac kiem tra lai")
        
        if st.button("Lam moi"):
            st.session_state['processed'] = False
            st.rerun()
    
    else:
        st.info("Tai anh len va bam 'Nhan dien mon an' de bat dau")
        st.caption("He thong se tu dong cat va nhan dien cac mon an")

# Footer
st.markdown("---")
st.caption("Food Detection System v1.0")
