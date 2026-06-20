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

# ============================================
# CÁC HÀM XỬ LÝ ẢNH
# ============================================

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
    boxes = detect_food_boxes(image)
    cropped_images = [box["image"] for box in boxes]
    return cropped_images, boxes

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
    
    # Ưu tiên file .onnx
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
# GIAO DIỆN STREAMLIT
# ============================================

# Cấu hình trang
st.set_page_config(
    page_title="Food Detection System",
    page_icon="",
    layout="wide"
)

# Tiêu đề
st.title("Food Detection System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Menu")
    st.markdown("**Danh sach mon an:**")
    
    for item in MENU:
        note = f" ({item['note']})" if item['note'] else ""
        st.text(f"{item['id']}. {item['name']} - {item['price']:,} VND{note}")
    
    st.markdown("---")
    
    # Kiểm tra model
    model_files = check_model_files()
    if model_files:
        st.success("Model da san sang")
        for mf in model_files:
            st.text(f"📁 {mf['name']} ({mf['size']:.1f}MB)")
    else:
        st.error("Chua co model.onnx")
        st.info("Tai file model.onnx len thu muc goc")
    
    st.markdown("---")
    st.caption("He thong nhan dien mon an tu anh")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Tai anh len de nhan dien")
    
    # Upload ảnh
    uploaded_file = st.file_uploader(
        "Chon anh mon an",
        type=['jpg', 'jpeg', 'png'],
        help="Tai anh chua khay thuc an"
    )
    
    if uploaded_file is not None:
        # Đọc và hiển thị ảnh gốc
        image = load_image(uploaded_file)
        st.image(image, caption="Anh da tai len", use_column_width=True)
        
        # Nút xử lý
        if st.button("Nhan dien mon an", type="primary"):
            with st.spinner("Dang xu ly anh..."):
                # Load model
                session = load_model()
                
                if session is None:
                    st.error("Khong tim thay model.onnx!")
                    st.info("Vui long upload file model.onnx vao thu muc goc")
                else:
                    # Lưu vào session state
                    st.session_state['image'] = image
                    st.session_state['session'] = session
                    st.session_state['processed'] = True
                    st.rerun()

with col2:
    st.subheader("Ket qua nhan dien")
    
    if 'processed' in st.session_state and st.session_state['processed']:
        image = st.session_state['image']
        session = st.session_state['session']
        
        # BƯỚC 1: CẮT ẢNH THÀNH CÁC MÓN RIÊNG
        with st.spinner("Dang cat anh thanh cac mon..."):
            cropped_images, boxes = crop_food_items(image)
        
        if len(cropped_images) > 0:
            st.success(f"Da phat hien {len(cropped_images)} mon an")
            
            detected_foods = []
            results_container = st.container()
            
            # BƯỚC 2: DUYỆT TỪNG ẢNH ĐÃ CẮT
            with results_container:
                for idx, cropped_img in enumerate(cropped_images):
                    st.markdown(f"**Mon {idx + 1}:**")
                    
                    # Tạo 2 cột: ảnh cắt và thông tin
                    col_img, col_info = st.columns([1, 2])
                    
                    # Cột 1: Hiển thị ảnh đã cắt
                    with col_img:
                        try:
                            # Resize ảnh để hiển thị đẹp
                            img_display = cv2.resize(cropped_img, (150, 150))
                            st.image(img_display, use_column_width=True)
                        except:
                            st.image(cropped_img, use_column_width=True)
                    
                    # Cột 2: Hiển thị thông tin
                    with col_info:
                        try:
                            # Tiền xử lý cho model
                            preprocessed = preprocess_image(cropped_img, target_size=(224, 224))
                            
                            # Dự đoán với ONNX
                            input_name = session.get_inputs()[0].name
                            output_name = session.get_outputs()[0].name
                            
                            result = session.run([output_name], {input_name: preprocessed})
                            predictions = result[0]
                            
                            # Lấy kết quả
                            food_id = np.argmax(predictions[0])
                            confidence = np.max(predictions[0])
                            
                            # Lấy thông tin món ăn
                            food_name = get_food_name(food_id)
                            food_price = get_food_price(food_id)
                            
                            # Hiển thị
                            st.text(f"Ten: {food_name}")
                            st.text(f"Gia: {food_price:,} VND")
                            st.text(f"Do tin cay: {confidence*100:.1f}%")
                            
                            # Đánh giá độ tin cậy
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
            
            # BƯỚC 3: TÍNH TỔNG TIỀN
            if detected_foods:
                total_price = 0
                detail_text = []
                
                for food_id in detected_foods:
                    name = get_food_name(food_id)
                    price = get_food_price(food_id)
                    total_price += price
                    detail_text.append(f"{name}: {price:,} VND")
                
                # Hiển thị tổng tiền
                st.markdown("### Tong tien")
                st.markdown(f"**{total_price:,} VND**")
                
                # Hiển thị chi tiết
                with st.expander("Xem chi tiet"):
                    for detail in detail_text:
                        st.text(detail)
        else:
            st.warning("Khong tim thay mon an trong anh")
            st.info("Vui long tai anh khac hoac kiem tra lai")
        
        # Nút reset
        if st.button("Lam moi"):
            st.session_state['processed'] = False
            st.rerun()
    
    else:
        st.info("Tai anh len va bam 'Nhan dien mon an'")
        st.caption("He thong se cat va nhan dien tung mon")

# Footer
st.markdown("---")
st.caption("Food Detection System v1.0 - ONNX Runtime")

# Debug: Hiển thị thông tin hệ thống
with st.expander("Thong tin he thong"):
    st.write("**Cac file trong thu muc:**")
    for f in os.listdir('.'):
        if not f.startswith('.'):
            try:
                size = os.path.getsize(f) / 1024
                if size > 1024:
                    st.write(f"- {f} ({size/1024:.1f} MB)")
                else:
                    st.write(f"- {f} ({size:.1f} KB)")
            except:
                st.write(f"- {f}")
