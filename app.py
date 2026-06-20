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
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }
    
    .main {
        background: #0d0d0d;
    }
    
    .header {
        background: #1a1a1a;
        padding: 2.5rem 2rem;
        border-radius: 0px;
        margin-bottom: 2rem;
        text-align: center;
        border-bottom: 1px solid #2a2a2a;
    }
    .header h1 {
        font-size: 1.8rem;
        font-weight: 300;
        letter-spacing: 4px;
        color: #00ff41;
        margin: 0;
        text-shadow: 0 0 20px rgba(0, 255, 65, 0.1);
    }
    .header p {
        font-size: 0.8rem;
        color: #666666;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
        letter-spacing: 2px;
    }
    
    /* Terminal style */
    .terminal {
        background: #0d0d0d;
        border: 1px solid #1a1a1a;
        padding: 1.5rem;
        border-radius: 0px;
        margin: 1rem 0;
    }
    .terminal-text {
        color: #00ff41;
        font-size: 0.8rem;
        font-weight: 300;
    }
    .terminal-text-gray {
        color: #666666;
        font-size: 0.8rem;
        font-weight: 300;
    }
    
    /* Menu */
    .menu-item {
        padding: 0.4rem 0;
        border-bottom: 1px solid #1a1a1a;
        font-size: 0.75rem;
        color: #cccccc;
        font-weight: 300;
    }
    .menu-item:hover {
        color: #00ff41;
        border-bottom: 1px solid #00ff41;
        transition: all 0.3s;
    }
    .menu-price {
        float: right;
        color: #00ff41;
        font-weight: 300;
    }
    
    /* Total Card */
    .total-card {
        background: #0d0d0d;
        border: 1px solid #00ff41;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }
    .total-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 255, 65, 0.03), transparent);
        animation: scan 4s linear infinite;
    }
    @keyframes scan {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    .total-card p {
        color: #666666;
        font-size: 0.7rem;
        font-weight: 300;
        letter-spacing: 3px;
        margin: 0;
        text-transform: uppercase;
    }
    .total-card h2 {
        color: #00ff41;
        font-size: 2.8rem;
        font-weight: 300;
        letter-spacing: 2px;
        margin: 0.5rem 0;
        text-shadow: 0 0 30px rgba(0, 255, 65, 0.05);
    }
    .total-card .sub {
        color: #666666;
        font-size: 0.7rem;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    /* 3D Button */
    .stButton button {
        background: #1a1a1a;
        color: #00ff41;
        border: 1px solid #2a2a2a;
        border-radius: 0px;
        padding: 0.7rem 2rem;
        font-size: 0.8rem;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
        width: 100%;
        position: relative;
        transition: all 0.1s ease;
        box-shadow: 
            0 4px 0 #0a0a0a,
            0 8px 0 #0a0a0a,
            0 12px 0 #0a0a0a;
        transform: translateY(-8px);
        cursor: pointer;
        text-shadow: 0 0 10px rgba(0, 255, 65, 0.1);
    }
    .stButton button:hover {
        background: #00ff41;
        color: #0d0d0d;
        border-color: #00ff41;
        box-shadow: 
            0 2px 0 #0a0a0a,
            0 4px 0 #0a0a0a,
            0 6px 0 #0a0a0a;
        transform: translateY(-4px);
        text-shadow: none;
    }
    .stButton button:active {
        box-shadow: 
            0 0px 0 #0a0a0a,
            0 0px 0 #0a0a0a,
            0 0px 0 #0a0a0a;
        transform: translateY(0px);
    }
    .stButton button:disabled {
        opacity: 0.3;
        box-shadow: 
            0 2px 0 #0a0a0a,
            0 4px 0 #0a0a0a;
        transform: translateY(-4px);
        cursor: not-allowed;
    }
    
    /* Secondary button */
    .stButton button[kind="secondary"] {
        background: transparent;
        color: #666666;
        border: 1px solid #2a2a2a;
        box-shadow: 
            0 2px 0 #0a0a0a,
            0 4px 0 #0a0a0a;
        transform: translateY(-4px);
    }
    .stButton button[kind="secondary"]:hover {
        background: #1a1a1a;
        color: #00ff41;
        border-color: #00ff41;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'JetBrains Mono', monospace !important;
        background: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 0px !important;
        color: #cccccc !important;
        font-size: 0.8rem !important;
        font-weight: 300 !important;
        letter-spacing: 1px !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: #00ff41 !important;
        color: #00ff41 !important;
    }
    .streamlit-expanderContent {
        background: #0d0d0d !important;
        border: 1px solid #1a1a1a !important;
        border-top: none !important;
        border-radius: 0px !important;
        padding: 1rem !important;
    }
    
    /* Metrics */
    .stMetric {
        background: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 0px !important;
        padding: 1rem !important;
    }
    .stMetric label {
        color: #666666 !important;
        font-size: 0.65rem !important;
        font-weight: 300 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
    }
    .stMetric .stMetricValue {
        color: #00ff41 !important;
        font-size: 1.5rem !important;
        font-weight: 300 !important;
    }
    
    /* Image caption */
    .stImage figcaption {
        color: #666666 !important;
        font-size: 0.7rem !important;
        font-weight: 300 !important;
        letter-spacing: 1px !important;
        text-align: center !important;
    }
    
    /* File uploader */
    .stFileUploader {
        background: #1a1a1a !important;
        border: 1px dashed #2a2a2a !important;
        border-radius: 0px !important;
        padding: 1rem !important;
    }
    .stFileUploader:hover {
        border-color: #00ff41 !important;
    }
    
    /* Info, warning, success, error */
    .stAlert {
        border-radius: 0px !important;
        background: #1a1a1a !important;
        border-left: 3px solid #00ff41 !important;
        color: #cccccc !important;
        font-weight: 300 !important;
    }
    .stAlert svg {
        fill: #00ff41 !important;
    }
    
    .stAlert.warning {
        border-left-color: #ff6b35 !important;
    }
    .stAlert.error {
        border-left-color: #ff1744 !important;
    }
    .stAlert.success {
        border-left-color: #00e676 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #2a2a2a;
        font-size: 0.65rem;
        border-top: 1px solid #1a1a1a;
        margin-top: 2rem;
        letter-spacing: 2px;
    }
    
    /* Spinner */
    .stSpinner {
        border-color: #00ff41 !important;
    }
    
    /* Badge */
    .badge-high {
        color: #00e676;
        font-size: 0.7rem;
        font-weight: 300;
        letter-spacing: 1px;
        border: 1px solid #00e676;
        padding: 0.2rem 0.8rem;
        display: inline-block;
    }
    .badge-medium {
        color: #ffab00;
        font-size: 0.7rem;
        font-weight: 300;
        letter-spacing: 1px;
        border: 1px solid #ffab00;
        padding: 0.2rem 0.8rem;
        display: inline-block;
    }
    .badge-low {
        color: #ff1744;
        font-size: 0.7rem;
        font-weight: 300;
        letter-spacing: 1px;
        border: 1px solid #ff1744;
        padding: 0.2rem 0.8rem;
        display: inline-block;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #0d0d0d !important;
    }
    .css-1d391kg .stMarkdown {
        color: #cccccc !important;
    }
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #00ff41 !important;
        font-weight: 300 !important;
        letter-spacing: 2px !important;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #1a1a1a;
        margin: 1.5rem 0;
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
    <p>▸ AUTOMATED MEAL RECOGNITION &amp; BILLING ▸ v2.0</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### MENU")
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
        st.success("MODEL READY")
        for mf in model_files:
            st.caption(f"▸ {mf['name']} ({mf['size']:.1f} MB)")
    else:
        st.error("MODEL NOT FOUND")
    
    st.markdown("---")
    st.caption(f"▸ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


col_left, col_right = st.columns([2.5, 1.5])

with col_left:
    st.markdown("### UPLOAD")
    
    uploaded_file = st.file_uploader(
        "Select image",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(image, caption="INPUT", use_column_width=True)
        
        if st.button("Recognize", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                session = load_model()
                
                if session is None:
                    st.error("MODEL NOT FOUND")
                else:
                    st.session_state['image'] = image
                    st.session_state['session'] = session
                    st.session_state['processed'] = True
                    st.rerun()


with col_right:
    st.markdown("### OUTPUT")
    
    if 'processed' in st.session_state and st.session_state['processed']:
        image = st.session_state['image']
        session = st.session_state['session']
        
        with st.spinner("Segmenting..."):
            cropped_results, img_resized = crop_food_items(image)
        
        if cropped_results:
            img_with_boxes = draw_boxes_fixed(img_resized, cropped_results)
            st.image(img_with_boxes, caption="DETECTED", use_column_width=True)
            
            st.info(f"{len(cropped_results)} trays detected")
            
            detected_foods = []
            food_details = []
            
            for idx, result in enumerate(cropped_results):
                cropped_img = result["image"]
                khay_id = result["id"]
                
                with st.expander(f"TRAY {khay_id}", expanded=(idx == 0)):
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
                            
                            st.caption("PREDICTIONS:")
                            for i, (fid, conf) in enumerate(zip(top3_idx, top3_conf)):
                                name = get_food_name(fid)
                                price = get_food_price(fid)
                                if i == 0:
                                    st.markdown(f"**{name}**  {conf*100:.1f}%")
                                    st.caption(f"Price: {price:,} VND")
                                    detected_foods.append(fid)
                                    food_details.append({
                                        "tray": khay_id,
                                        "name": name,
                                        "price": price,
                                        "confidence": conf
                                    })
                                else:
                                    st.text(f"{name}  {conf*100:.1f}%")
                            
                            if top3_conf[0] > 0.8:
                                st.markdown('<span class="badge-high">HIGH CONFIDENCE</span>', unsafe_allow_html=True)
                            elif top3_conf[0] > 0.5:
                                st.markdown('<span class="badge-medium">MEDIUM CONFIDENCE</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="badge-low">LOW CONFIDENCE</span>', unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            if detected_foods:
                total_price, details = calculate_total(detected_foods)
                
                st.markdown("---")
                st.markdown(f"""
                <div class="total-card">
                    <p>TOTAL</p>
                    <h2>{total_price:,} VND</h2>
                    <p class="sub">{len(detected_foods)} items · {len(set(detected_foods))} types</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("INVOICE DETAILS", expanded=False):
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
                        label="DOWNLOAD INVOICE",
                        data=invoice_text,
                        file_name=f"invoice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("ITEMS", len(detected_foods))
                with col_stat2:
                    st.metric("TYPES", len(set(detected_foods)))
                with col_stat3:
                    avg_price = total_price // len(detected_foods) if len(detected_foods) > 0 else 0
                    st.metric("AVG PRICE", f"{avg_price:,} VND")
            
        else:
            st.warning("No trays detected")
        
        if st.button("RESET", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    else:
        st.info("Upload image and click Recognize")
        st.caption("▸ automatic segmentation ▸ real-time inference")


st.markdown("""
<div class="footer">
    <p>FOOD DETECTION SYSTEM v2.0 &middot; ONNX RUNTIME &middot; 10 FOOD CLASSES</p>
    <p style="color: #1a1a1a;">▸ ▸ ▸</p>
</div>
""", unsafe_allow_html=True)
