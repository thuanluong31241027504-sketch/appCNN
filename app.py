import streamlit as st
import numpy as np
import cv2
import os
import sys
import onnxruntime as ort
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.menu import MENU, get_food_name, get_food_price, calculate_total
from utils.image_processor import load_image, preprocess_image, crop_food_items, draw_boxes_fixed


st.set_page_config(
    page_title="Food Detection System",
    page_icon="",
    layout="wide"
)


st.markdown("""
<style>
    .header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: #ffffff;
    }
    .header h1 {
        font-size: 2.2rem;
        font-weight: 300;
        letter-spacing: 2px;
        margin: 0;
    }
    .header p {
        font-size: 1rem;
        opacity: 0.7;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
        letter-spacing: 1px;
    }
    .total-card {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: #ffffff;
        margin: 1.5rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .total-card h2 {
        font-size: 2.8rem;
        font-weight: 300;
        letter-spacing: 2px;
        margin: 0;
    }
    .total-card p {
        font-size: 1rem;
        opacity: 0.7;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    .menu-item {
        padding: 0.3rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 0.9rem;
    }
    .menu-price {
        float: right;
        font-weight: 400;
        color: #4fc3f7;
    }
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #6c757d;
        font-size: 0.8rem;
        border-top: 1px solid #2d2d2d;
        margin-top: 2rem;
        letter-spacing: 0.5px;
    }
    .badge-high {
        background: #1b5e20;
        color: #a5d6a7;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 400;
    }
    .badge-medium {
        background: #e65100;
        color: #ffcc80;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 400;
    }
    .badge-low {
        background: #b71c1c;
        color: #ef9a9a;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 400;
    }
    .khay-container {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #2d2d2d;
    }
    .stButton button {
        background: #0f3460;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 300;
        letter-spacing: 1px;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background: #1a1a2e;
        border: 1px solid #0f3460;
    }
</style>
""", unsafe_allow_html=True)


def check_model_files():
    model_files = []
    possible_names = ['model.onnx', 'trainaicuoiky1.onnx']
    
    for name in possible_names:
        if os.path.exists(name):
            size = os.path.getsize(name) / (1024 * 1024)
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
        except Exception:
            return None
    
    return None


st.markdown("""
<div class="header">
    <h1>FOOD DETECTION SYSTEM</h1>
    <p>Automated meal recognition and billing</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### Menu")
    st.markdown("---")
    
    categories = {}
    for item in MENU:
        cat = item.get("category", "Other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item)
    
    for cat, items in categories.items():
        st.markdown(f"**{cat.upper()}**")
        for item in items:
            note = f" ({item['note']})" if item.get('note') else ""
            st.markdown(
                f"<div class='menu-item'>{item['name']} {note} "
                f"<span class='menu-price'>{item['price']:,} VND</span></div>",
                unsafe_allow_html=True
            )
    
    st.markdown("---")
    
    model_files = check_model_files()
    if model_files:
        st.success("Model ready")
        for mf in model_files:
            st.caption(f"{mf['name']} ({mf['size']:.1f} MB)")
    else:
        st.error("Model not found")
    
    st.markdown("---")
    st.caption(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


col_left, col_right = st.columns([2.5, 1.5])

with col_left:
    st.markdown("### Upload Image")
    
    uploaded_file = st.file_uploader(
        "Select meal tray image",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)
        
        if st.button("Recognize", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                session = load_model()
                
                if session is None:
                    st.error("Model not found")
                else:
                    st.session_state['image'] = image
                    st.session_state['session'] = session
                    st.session_state['processed'] = True
                    st.rerun()


with col_right:
    st.markdown("### Results")
    
    if 'processed' in st.session_state and st.session_state['processed']:
        image = st.session_state['image']
        session = st.session_state['session']
        
        with st.spinner("Segmenting image..."):
            cropped_results, img_resized = crop_food_items(image)
        
        if cropped_results:
            img_with_boxes = draw_boxes_fixed(img_resized, cropped_results)
            st.image(img_with_boxes, caption="Detected trays", use_column_width=True)
            
            st.info(f"{len(cropped_results)} trays detected")
            
            detected_foods = []
            food_details = []
            
            for idx, result in enumerate(cropped_results):
                cropped_img = result["image"]
                khay_id = result["id"]
                
                with st.expander(f"Tray {khay_id}", expanded=(idx == 0)):
                    col_img, col_info = st.columns([1, 1.5])
                    
                    with col_img:
                        try:
                            h, w = cropped_img.shape[:2]
                            scale = min(160 / h, 160 / w)
                            new_h, new_w = int(h * scale), int(w * scale)
                            img_display = cv2.resize(cropped_img, (new_w, new_h))
                            st.image(img_display, use_column_width=True)
                        except Exception:
                            st.image(cropped_img, use_column_width=True)
                    
                    with col_info:
                        try:
                            preprocessed = preprocess_image(cropped_img, target_size=(224, 224))
                            
                            input_name = session.get_inputs()[0].name
                            output_name = session.get_outputs()[0].name
                            
                            result_onnx = session.run([output_name], {input_name: preprocessed})
                            predictions = result_onnx[0][0]
                            
                            top3_idx = np.argsort(predictions)[-3:][::-1]
                            top3_conf = predictions[top3_idx]
                            
                            st.caption("Top predictions:")
                            for i, (fid, conf) in enumerate(zip(top3_idx, top3_conf)):
                                name = get_food_name(fid)
                                price = get_food_price(fid)
                                if i == 0:
                                    st.markdown(f"**{name}** - {conf*100:.1f}%")
                                    st.caption(f"Price: {price:,} VND")
                                    detected_foods.append(fid)
                                    food_details.append({
                                        "tray": khay_id,
                                        "name": name,
                                        "price": price,
                                        "confidence": conf
                                    })
                                else:
                                    st.text(f"{name} - {conf*100:.1f}%")
                            
                            if top3_conf[0] > 0.8:
                                st.markdown('<span class="badge-high">High confidence</span>', unsafe_allow_html=True)
                            elif top3_conf[0] > 0.5:
                                st.markdown('<span class="badge-medium">Medium confidence</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="badge-low">Low confidence</span>', unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            if detected_foods:
                total_price, details = calculate_total(detected_foods)
                
                st.markdown("---")
                st.markdown(f"""
                <div class="total-card">
                    <p>TOTAL</p>
                    <h2>{total_price:,} VND</h2>
                    <p>{len(detected_foods)} items · {len(set(detected_foods))} types</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("Invoice details", expanded=False):
                    for i, detail in enumerate(details):
                        st.text(f"Tray {i+1}: {detail['name']} - {detail['price']:,} VND")
                    
                    st.markdown("---")
                    st.markdown(f"**Total: {total_price:,} VND**")
                    st.caption(f"{len(detected_foods)} items · {len(set(detected_foods))} types")
                    
                    invoice_text = f"INVOICE\n{'-'*30}\n"
                    for i, d in enumerate(details):
                        invoice_text += f"Tray {i+1}: {d['name']} - {d['price']:,} VND\n"
                    invoice_text += f"{'-'*30}\nTOTAL: {total_price:,} VND"
                    
                    st.download_button(
                        label="Download invoice",
                        data=invoice_text,
                        file_name=f"invoice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Items", len(detected_foods))
                with col_stat2:
                    st.metric("Types", len(set(detected_foods)))
                with col_stat3:
                    avg_price = total_price // len(detected_foods) if len(detected_foods) > 0 else 0
                    st.metric("Avg price", f"{avg_price:,} VND")
            
        else:
            st.warning("No trays detected")
        
        if st.button("Reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    else:
        st.info("Upload image and click Recognize")
        st.caption("System will segment and classify each tray")


st.markdown("""
<div class="footer">
    <p>FOOD DETECTION SYSTEM v2.0 &middot; ONNX Runtime &middot; 10 food classes</p>
</div>
""", unsafe_allow_html=True)
