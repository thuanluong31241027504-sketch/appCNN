import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import sys
import onnxruntime as ort
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.menu import MENU, get_food_name, get_food_price, get_food_category, get_food_note, calculate_total
from utils.image_processor import load_image, preprocess_image, crop_food_items, draw_boxes_fixed

# ============================================
# CẤU HÌNH TRANG
# ============================================

st.set_page_config(
    page_title="Food Detection System - Nhận diện món ăn",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS TÙY CHỈNH
st.markdown("""
<style>
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    
    /* Total Card */
    .total-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1.5rem 0;
    }
    .total-card h2 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
    }
    .total-card p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    
    /* Badge */
    .badge-high { background: #d4edda; color: #155724; padding: 0.25rem 0.75rem; border-radius: 50px; font-size: 0.75rem; font-weight: 600; }
    .badge-medium { background: #fff3cd; color: #856404; padding: 0.25rem 0.75rem; border-radius: 50px; font-size: 0.75rem; font-weight: 600; }
    .badge-low { background: #f8d7da; color: #721c24; padding: 0.25rem 0.75rem; border-radius: 50px; font-size: 0.75rem; font-weight: 600; }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #6c757d;
        font-size: 0.9rem;
        border-top: 1px solid #e9ecef;
        margin-top: 2rem;
    }
    
    /* Menu item */
    .menu-item {
        padding: 0.25rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    .menu-price {
        float: right;
        font-weight: 600;
        color: #28a745;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================

st.markdown("""
<div class="main-header">
    <h1>🍽️ Food Detection System</h1>
    <p>Nhận diện món ăn từ ảnh khay cơm • Tính tiền tự động</p>
</div>
""", unsafe_allow_html=True)

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
            st.error(f"Lỗi load ONNX: {str(e)}")
            return None
    
    return None

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("### 📋 Menu món ăn")
    st.markdown("---")
    
    # Hiển thị menu theo danh mục
    categories = {}
    for item in MENU:
        cat = item["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item)
    
    for cat, items in categories.items():
        st.markdown(f"**{cat}:**")
        for item in items:
            note = f" ({item['note']})" if item['note'] else ""
            st.markdown(f"<div class='menu-item'>• {item['name']} <span class='menu-price'>{item['price']:,} VND</span>{note}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Trạng thái model
    st.markdown("### 📊 Trạng thái")
    model_files = check_model_files()
    
    if model_files:
        st.success("✅ Model sẵn sàng")
        for mf in model_files:
            st.info(f"📁 {mf['name']} ({mf['size']:.1f} MB)")
    else:
        st.error("❌ Chưa có model.onnx")
        st.info("📤 Upload file model.onnx")
    
    st.markdown("---")
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}")

# ============================================
# MAIN CONTENT
# ============================================

col_left, col_right = st.columns([2.5, 1.5])

with col_left:
    st.markdown("### 📤 Tải ảnh lên")
    
    uploaded_file = st.file_uploader(
        "Chọn ảnh khay cơm",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(image, caption="📷 Ảnh đã tải lên", use_column_width=True)
        
        if st.button("🚀 Nhận diện món ăn", type="primary", use_container_width=True):
            with st.spinner("🔄 Đang xử lý..."):
                session = load_model()
                
                if session is None:
                    st.error("❌ Không tìm thấy model.onnx!")
                else:
                    st.session_state['image'] = image
                    st.session_state['session'] = session
                    st.session_state['processed'] = True
                    st.rerun()

with col_right:
    st.markdown("### 📊 Kết quả")
    
    if 'processed' in st.session_state and st.session_state['processed']:
        image = st.session_state['image']
        session = st.session_state['session']
        
        with st.spinner("✂️ Đang cắt ảnh..."):
            cropped_results, img_resized = crop_food_items(image)
        
        if cropped_results:
            # Hiển thị ảnh đã cắt
            img_with_boxes = draw_boxes_fixed(img_resized, cropped_results)
            st.image(img_with_boxes, caption="✅ Đã phát hiện các khay", use_column_width=True)
            
            st.success(f"🎯 {len(cropped_results)} khay")
            
            detected_foods = []
            food_details = []
            
            # Nhận diện từng khay
            for idx, result in enumerate(cropped_results):
                cropped_img = result["image"]
                khay_id = result["id"]
                khay_name = result["name"]
                
                with st.expander(f"🍽️ {khay_name}", expanded=(idx == 0)):
                    col_img, col_info = st.columns([1, 1.5])
                    
                    with col_img:
                        try:
                            h, w = cropped_img.shape[:2]
                            scale = min(180/h, 180/w)
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
                            predictions = result_onnx[0][0]
                            
                            # Top 3
                            top3_idx = np.argsort(predictions)[-3:][::-1]
                            top3_conf = predictions[top3_idx]
                            
                            st.markdown("**🔍 Top 3 dự đoán:**")
                            for i, (fid, conf) in enumerate(zip(top3_idx, top3_conf)):
                                name = get_food_name(fid)
                                price = get_food_price(fid)
                                if i == 0:
                                    st.markdown(f"**🏆 {i+1}. {name}** - {conf*100:.1f}%")
                                    st.markdown(f"💰 Giá: **{price:,} VND**")
                                    detected_foods.append(fid)
                                    food_details.append({
                                        "khay": khay_id,
                                        "name": name,
                                        "price": price,
                                        "confidence": conf
                                    })
                                else:
                                    st.text(f"   {i+1}. {name} - {conf*100:.1f}%")
                            
                            # Đánh giá độ tin cậy
                            if top3_conf[0] > 0.8:
                                st.markdown('<span class="badge-high">✅ Tin cậy cao</span>', unsafe_allow_html=True)
                            elif top3_conf[0] > 0.5:
                                st.markdown('<span class="badge-medium">⚠️ Trung bình</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="badge-low">❌ Tin cậy thấp</span>', unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Lỗi: {str(e)}")
            
            # ============================================
            # TỔNG TIỀN
            # ============================================
            
            if detected_foods:
                total_price, details = calculate_total(detected_foods)
                
                # Tổng hóa đơn
                st.markdown("---")
                st.markdown(f"""
                <div class="total-card">
                    <p>🧾 TỔNG HÓA ĐƠN</p>
                    <h2>{total_price:,} VND</h2>
                    <p>{len(detected_foods)} món • {len(set(detected_foods))} loại</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Chi tiết
                with st.expander("📋 Chi tiết hóa đơn", expanded=False):
                    for i, detail in enumerate(details):
                        khay_num = i + 1
                        st.markdown(f"**Khay {khay_num}:** {detail['name']} - {detail['price']:,} VND")
                    
                    st.markdown("---")
                    st.markdown(f"**Tổng cộng: {total_price:,} VND**")
                    st.caption(f"Số món: {len(detected_foods)} | Số loại: {len(set(detected_foods))}")
                    
                    # Tải hóa đơn
                    invoice_text = f"HÓA ĐƠN\n{'-'*30}\n"
                    for i, d in enumerate(details):
                        invoice_text += f"Khay {i+1}: {d['name']} - {d['price']:,} VND\n"
                    invoice_text += f"{'-'*30}\nTỔNG CỘNG: {total_price:,} VND"
                    
                    st.download_button(
                        label="📥 Tải hóa đơn",
                        data=invoice_text,
                        file_name=f"hoadon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                # Thống kê
                st.markdown("### 📈 Thống kê")
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Tổng món", len(detected_foods))
                with col_stat2:
                    st.metric("Số loại", len(set(detected_foods)))
                with col_stat3:
                    avg_price = total_price // len(detected_foods) if len(detected_foods) > 0 else 0
                    st.metric("Giá TB", f"{avg_price:,} VND")
            
        else:
            st.warning("⚠️ Không phát hiện khay nào")
            st.info("💡 Thử ảnh khác")
        
        if st.button("🔄 Làm mới", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    else:
        st.info("📌 Tải ảnh và bấm 'Nhận diện'")
        st.caption("Tự động cắt và nhận diện từng khay")

# ============================================
# FOOTER
# ============================================

st.markdown("""
<div class="footer">
    <p>🍽️ Food Detection System v2.0 • Nhận diện 10 loại món ăn • ONNX Runtime</p>
    <p style="font-size: 0.8rem;">© 2026 - Đồ án Nhận diện món ăn</p>
</div>
""", unsafe_allow_html=True)
