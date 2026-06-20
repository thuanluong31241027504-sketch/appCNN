import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import sys
import onnxruntime as ort

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.menu import MENU, get_food_name, get_food_price, calculate_total
from utils.image_processor import load_image, crop_food_items, draw_boxes_fixed

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

def preprocess_image(image, target_size=(224, 224)):
    """Tiền xử lý ảnh cho model ONNX - ĐÚNG CHO NHWC (batch, h, w, c)"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize ảnh
    img_resized = cv2.resize(image, target_size)
    
    # Chuẩn hóa về [0,1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Thêm batch dimension - KEEP NHWC (batch, height, width, channels)
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

st.set_page_config(
    page_title="Food Detection System",
    page_icon="",
    layout="wide"
)

st.title("Food Detection System")
st.markdown("---")

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

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Tai anh len de nhan dien")
    
    uploaded_file = st.file_uploader(
        "Chon anh mon an",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(image, caption="Anh goc", use_column_width=True)
        
        if st.button("Nhan dien mon an", type="primary"):
            with st.spinner("Dang xu ly..."):
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
        
        with st.spinner("Dang cat anh..."):
            cropped_results, img_resized = crop_food_items(image)
        
        if cropped_results:
            img_with_boxes = draw_boxes_fixed(img_resized, cropped_results)
            st.image(img_with_boxes, caption="Da cat cac khay", use_column_width=True)
            
            st.success(f"Da cat thanh {len(cropped_results)} khay")
            
            # DEBUG: Kiểm tra model
            with st.expander("🔧 Thong tin model"):
                st.write("**Input:**")
                for inp in session.get_inputs():
                    st.write(f"- {inp.name}: Shape: {inp.shape} - Type: {inp.type}")
                st.write("**Output:**")
                for out in session.get_outputs():
                    st.write(f"- {out.name}: Shape: {out.shape} - Type: {out.type}")
            
            detected_foods = []
            all_predictions = []
            
            for result in cropped_results:
                cropped_img = result["image"]
                khay_id = result["id"]
                khay_name = result["name"]
                
                st.markdown(f"**{khay_name} {khay_id}:**")
                
                col_img, col_info = st.columns([1, 2])
                
                with col_img:
                    try:
                        h, w = cropped_img.shape[:2]
                        scale = min(200/h, 200/w)
                        new_h, new_w = int(h*scale), int(w*scale)
                        img_display = cv2.resize(cropped_img, (new_w, new_h))
                        st.image(img_display, use_column_width=True)
                        st.caption(f"Kich thuoc: {w}x{h}")
                    except:
                        st.image(cropped_img, use_column_width=True)
                
                with col_info:
                    try:
                        # Tiền xử lý
                        preprocessed = preprocess_image(cropped_img, target_size=(224, 224))
                        
                        # Lấy tên input/output
                        input_name = session.get_inputs()[0].name
                        output_name = session.get_outputs()[0].name
                        
                        # Dự đoán
                        result_onnx = session.run([output_name], {input_name: preprocessed})
                        predictions = result_onnx[0][0]
                        
                        # Lưu để kiểm tra
                        all_predictions.append(predictions)
                        
                        # HIỂN THỊ TOP 5
                        st.text("=== TOP 5 DU DOAN ===")
                        top5_idx = np.argsort(predictions)[-5:][::-1]
                        top5_conf = predictions[top5_idx]
                        
                        for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
                            name = get_food_name(idx)
                            if i == 0:
                                st.markdown(f"**{i+1}. {name}: {conf*100:.2f}%**")
                            else:
                                st.text(f"{i+1}. {name}: {conf*100:.2f}%")
                        
                        # Lấy dự đoán tốt nhất
                        food_id = top5_idx[0]
                        confidence = top5_conf[0]
                        food_name = get_food_name(food_id)
                        food_price = get_food_price(food_id)
                        
                        st.markdown("---")
                        st.markdown(f"**Ket qua: {food_name}**")
                        st.text(f"Gia: {food_price:,} VND")
                        st.text(f"Do tin cay: {confidence*100:.2f}%")
                        
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
            
            # KIỂM TRA TẤT CẢ DỰ ĐOÁN CÓ GIỐNG NHAU KHÔNG
            if all_predictions:
                with st.expander("🔍 Kiem tra model"):
                    # Kiểm tra xem tất cả predictions có giống nhau không
                    first_pred = all_predictions[0]
                    all_same = all(np.array_equal(pred, first_pred) for pred in all_predictions)
                    
                    if all_same:
                        st.error("⚠️ TAT CA CAC KHAY CHO CUNG KET QUA!")
                        st.info("Model co the bi loi hoac chua train tot.")
                    
                    # Hiển thị phân phối xác suất
                    st.write("**Phan phoi xac suat trung binh:**")
                    avg_pred = np.mean(all_predictions, axis=0)
                    for i, prob in enumerate(avg_pred):
                        name = get_food_name(i)
                        st.text(f"{name}: {prob*100:.2f}%")
            
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
        else:
            st.warning("Khong cat duoc anh!")
        
        if st.button("Lam moi"):
            st.session_state.clear()
            st.rerun()
    
    else:
        st.info("Tai anh len va bam 'Nhan dien mon an'")

st.markdown("---")
st.caption("Food Detection System v1.0")
