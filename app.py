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

# Load model CNN
@st.cache_resource
def load_model():
    """Load model CNN đã train"""
    # Model ở thư mục gốc
    model_path = 'trainaicuoiky1.keras'
    
    # Kiểm tra các đường dẫn khác nếu không tìm thấy
    if not os.path.exists(model_path):
        # Thử tìm trong thư mục model
        alt_path = 'model/trainaicuoiky1.keras'
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            st.error(f"Khong tim thay file model. Da tim tai: {model_path} va {alt_path}")
            return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        st.success(f"Da load model thanh cong!")
        return model
    except Exception as e:
        st.error(f"Loi khi load model: {str(e)}")
        return None

# Tiêu đề chính
st.title("Food Detection System")
st.markdown("---")

# Hiển thị menu ở sidebar
with st.sidebar:
    st.header("Menu")
    st.markdown("**Danh sach mon an:**")
    
    # Tạo bảng menu
    menu_data = []
    for item in MENU:
        note = f" ({item['note']})" if item['note'] else ""
        menu_data.append(f"{item['id']}. {item['name']} - {item['price']:,} VND{note}")
    
    for line in menu_data:
        st.text(line)
    
    st.markdown("---")
    
    # Hiển thị thông tin model
    if os.path.exists('trainaicuoiky1.keras'):
        st.success("Model: trainaicuoiky1.keras")
        st.caption(f"Kich thuoc: {os.path.getsize('trainaicuoiky1.keras') / (1024*1024):.2f} MB")
    else:
        st.warning("Chua tim thay file model")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Tai anh len de nhan dien")
    
    # Upload ảnh
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
                # Load model
                model = load_model()
                if model is None:
                    st.error("Khong the load model. Vui long kiem tra file trainaicuoiky1.keras")
                else:
                    # Xử lý ảnh
                    st.session_state['image'] = image
                    st.session_state['model'] = model
                    st.session_state['processed'] = True
                    st.rerun()

with col2:
    st.subheader("Ket qua nhan dien")
    
    # Kiểm tra trạng thái xử lý
    if 'processed' in st.session_state and st.session_state['processed']:
        image = st.session_state['image']
        model = st.session_state['model']
        
        # Cắt các món ăn
        cropped_images, boxes = crop_food_items(image)
        
        if len(cropped_images) > 0:
            st.success(f"Da phat hien {len(cropped_images)} mon an")
            
            # Hiển thị từng món đã cắt và dự đoán
            detected_foods = []
            results_container = st.container()
            
            with results_container:
                for idx, cropped_img in enumerate(cropped_images):
                    st.markdown(f"**Mon {idx + 1}:**")
                    
                    # Hiển thị ảnh đã cắt
                    col_img, col_info = st.columns([1, 2])
                    
                    with col_img:
                        # Resize ảnh để hiển thị
                        img_display = cv2.resize(cropped_img, (150, 150))
                        st.image(img_display, use_column_width=True)
                    
                    with col_info:
                        # Tiền xử lý cho model
                        preprocessed = preprocess_image(cropped_img, target_size=(224, 224))
                        
                        # Dự đoán
                        predictions = model.predict(preprocessed)
                        food_id = np.argmax(predictions[0])
                        confidence = np.max(predictions[0])
                        
                        # Lấy thông tin món ăn
                        food_name = get_food_name(food_id)
                        food_price = get_food_price(food_id)
                        
                        st.text(f"Ten: {food_name}")
                        st.text(f"Gia: {food_price:,} VND")
                        st.text(f"Do tin cay: {confidence*100:.1f}%")
                        
                        # Màu sắc theo độ tin cậy
                        if confidence > 0.8:
                            st.success("Tin cay cao")
                        elif confidence > 0.5:
                            st.warning("Tin cay trung binh")
                        else:
                            st.error("Tin cay thap")
                        
                        detected_foods.append(food_id)
                    
                    st.markdown("---")
            
            # Tính tổng tiền
            total_price, details = calculate_total(detected_foods)
            
            # Hiển thị tổng tiền
            st.markdown("### Tong tien")
            st.markdown(f"**{total_price:,} VND**")
            
            # Hiển thị chi tiết
            with st.expander("Xem chi tiet"):
                for detail in details:
                    st.text(f"{detail['name']}: {detail['price']:,} VND")
        
        else:
            st.warning("Khong tim thay mon an trong anh")
            st.info("Vui long tai anh khac hoac kiem tra lai")
        
        # Nút reset
        if st.button("Lam moi"):
            st.session_state['processed'] = False
            st.rerun()
    
    else:
        st.info("Tai anh len va bam 'Nhan dien mon an' de bat dau")
        st.caption("He thong se tu dong cat va nhan dien cac mon an")

# Footer
st.markdown("---")
st.caption("Food Detection System v1.0 - Su dung CNN de nhan dien mon an")
