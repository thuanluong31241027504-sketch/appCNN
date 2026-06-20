import streamlit as st
import numpy as np
import cv2
import os
import sys
import onnxruntime as ort
from datetime import datetime
import qrcode
from io import BytesIO
import base64

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.menu import MENU, get_food_name, get_food_price, get_food_category, has_extra_option, calculate_total
from utils.image_processor import load_image, preprocess_image, crop_food_items, draw_boxes_fixed


st.set_page_config(
    page_title="Food Image Recognizing",
    page_icon="",
    layout="wide"
)


def generate_qr_code(amount, bank_id="0393167129", bank_name="MB", account_name="LUONG NGOC THUAN"):
    """Generate QR code for bank transfer"""
    qr_data = f"https://img.vietqr.io/image/{bank_name}-{bank_id}-compact.png?amount={amount}&addInfo=THANHTOAN"
    
    qr = qrcode.QRCode(version=1, box_size=4, border=2)
    qr.add_data(qr_data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'JetBrains Mono', 'Menlo', 'Monaco', 'Courier New', monospace;
    }
    
    .header {
        background: #ffffff;
        padding: 1.5rem 2rem;
        border: 1px solid #1a1a1a;
        margin-bottom: 2rem;
        text-align: center;
    }
    .header h1 {
        font-size: 1.4rem;
        font-weight: 300;
        letter-spacing: 6px;
        color: #1a1a1a;
        margin: 0;
        text-transform: uppercase;
    }
    .header p {
        font-size: 0.6rem;
        color: #888888;
        margin: 0.4rem 0 0 0;
        font-weight: 300;
        letter-spacing: 1px;
        text-transform: none;
    }
    
    .sidebar-title {
        font-size: 0.7rem;
        font-weight: 400;
        letter-spacing: 4px;
        color: #1a1a1a;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    
    .menu-item {
        padding: 0.3rem 0;
        border-bottom: 1px solid #eeeeee;
        font-size: 0.65rem;
        color: #1a1a1a;
        font-weight: 300;
        transition: all 0.2s;
    }
    .menu-item:hover {
        border-bottom: 1px solid #1a1a1a;
        padding-left: 0.5rem;
    }
    .menu-price {
        float: right;
        color: #22c55e;
        font-weight: 400;
    }
    .menu-category {
        font-size: 0.55rem;
        color: #999999;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 0.8rem 0 0.2rem 0;
        border-bottom: 1px solid #eeeeee;
        padding-bottom: 0.2rem;
    }
    
    .total-card {
        background: #ffffff;
        border: 1px solid #1a1a1a;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .total-card .label {
        color: #999999;
        font-size: 0.5rem;
        font-weight: 300;
        letter-spacing: 4px;
        text-transform: uppercase;
    }
    .total-card .amount {
        color: #22c55e;
        font-size: 2.2rem;
        font-weight: 300;
        letter-spacing: 3px;
        margin: 0.3rem 0;
    }
    .total-card .summary {
        color: #999999;
        font-size: 0.55rem;
        font-weight: 300;
        letter-spacing: 2px;
    }
    
    .invoice-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #f0f0f0;
        font-size: 0.65rem;
        color: #333333;
    }
    .invoice-row:hover {
        background: #fafafa;
    }
    .invoice-row .price {
        color: #22c55e;
    }
    .invoice-total {
        display: flex;
        justify-content: space-between;
        padding: 0.6rem 0;
        border-top: 2px solid #1a1a1a;
        font-size: 0.75rem;
        font-weight: 500;
        color: #1a1a1a;
        margin-top: 0.5rem;
    }
    .invoice-total .total-price {
        color: #22c55e;
    }
    
    .stButton button {
        background: #ffffff;
        color: #1a1a1a;
        border: 1px solid #1a1a1a;
        border-radius: 0px;
        padding: 0.6rem 2rem;
        font-size: 0.65rem;
        font-weight: 300;
        letter-spacing: 4px;
        text-transform: uppercase;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        background: #1a1a1a;
        color: #ffffff;
    }
    
    .streamlit-expanderHeader {
        background: #ffffff !important;
        border: 1px solid #1a1a1a !important;
        border-radius: 0px !important;
        color: #1a1a1a !important;
        font-size: 0.65rem !important;
        font-weight: 300 !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
    }
    .streamlit-expanderHeader:hover {
        background: #f7f7f7 !important;
    }
    .streamlit-expanderContent {
        background: #ffffff !important;
        border: 1px solid #1a1a1a !important;
        border-top: none !important;
        border-radius: 0px !important;
        padding: 1rem !important;
    }
    
    .stAlert {
        border-radius: 0px !important;
        background: #f7f7f7 !important;
        border-left: 3px solid #1a1a1a !important;
        color: #333333 !important;
        font-weight: 300 !important;
        font-size: 0.65rem !important;
    }
    
    .stImage figcaption {
        color: #999999 !important;
        font-size: 0.5rem !important;
        font-weight: 300 !important;
        letter-spacing: 2px !important;
        text-align: center !important;
        text-transform: uppercase !important;
    }
    
    .footer {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        color: #999999;
        font-size: 0.55rem;
        border-top: 1px solid #eeeeee;
        margin-top: 1.5rem;
        letter-spacing: 2px;
    }
    .footer .copyright {
        color: #999999;
        font-weight: 300;
    }
    
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-dot.green {
        background: #22c55e;
    }
    .status-dot.red {
        background: #cccccc;
    }
    
    .model-info {
        font-size: 0.55rem;
        color: #888888;
        letter-spacing: 1px;
        padding: 0.3rem 0;
    }
    
    .extra-input {
        margin: 0.5rem 0;
        padding: 0.5rem;
        border: 1px solid #eeeeee;
        background: #fafafa;
    }
    
    .prediction-item {
        padding: 0.2rem 0;
    }
    .prediction-item .main {
        font-weight: 400;
        font-size: 0.8rem;
        color: #1a1a1a;
    }
    .prediction-item .main .conf {
        color: #888888;
        font-size: 0.6rem;
        font-weight: 300;
    }
    .prediction-item .sub {
        font-size: 0.55rem;
        color: #22c55e;
        font-weight: 300;
    }
    .prediction-item .alt {
        color: #bbbbbb;
        font-size: 0.6rem;
    }
    .prediction-item .alt .conf-alt {
        color: #cccccc;
        font-size: 0.5rem;
    }
    
    .qr-container {
        border: 1px solid #1a1a1a;
        padding: 1rem;
        text-align: center;
        background: #ffffff;
        margin: 1rem 0;
    }
    .qr-container img {
        max-width: 180px;
        height: auto;
    }
    .qr-container .qr-amount {
        font-size: 1.2rem;
        font-weight: 300;
        color: #22c55e;
        margin: 0.3rem 0;
    }
    .qr-container .qr-bank {
        font-size: 0.55rem;
        color: #666666;
        font-weight: 300;
        letter-spacing: 1px;
    }
    .qr-container .qr-label {
        font-size: 0.5rem;
        color: #999999;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 0.2rem;
    }
    
    .camera-container {
        border: 1px solid #1a1a1a;
        padding: 1rem;
        text-align: center;
        background: #fafafa;
        margin: 1rem 0;
    }
    .camera-container .camera-label {
        font-size: 0.55rem;
        color: #666666;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    ::-webkit-scrollbar {
        width: 4px;
        background: #f7f7f7;
    }
    ::-webkit-scrollbar-thumb {
        background: #1a1a1a;
    }
    
    .css-1d391kg {
        background-color: #ffffff !important;
        border-right: 1px solid #eeeeee !important;
    }
    .css-1d391kg .stMarkdown {
        color: #333333 !important;
    }
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #1a1a1a !important;
        font-weight: 300 !important;
        letter-spacing: 3px !important;
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
    }
    
    hr {
        border: none;
        border-top: 1px solid #eeeeee;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def check_model_files():
    model_files = []
    possible_names = ['food_model.onnx', 'model.onnx', 'trainaicuoiky1.onnx']
    
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
    
    onnx_files = [f for f in model_files if f['name'] == 'food_model.onnx']
    if not onnx_files:
        onnx_files = [f for f in model_files if f['name'].endswith('.onnx')]
    
    if onnx_files:
        chosen = onnx_files[0]
        try:
            session = ort.InferenceSession(chosen['path'])
            return session
        except Exception:
            return None
    
    return None


def render_status_dot(ready):
    color = "green" if ready else "red"
    return f'<span class="status-dot {color}"></span>'


st.markdown("""
<div class="header">
    <h1>FOOD IMAGE RECOGNIZING</h1>
    <p>Mo hinh CNN trong nhan dien mon an va tinh tien tu dong</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="sidebar-title">Menu</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    categories = {}
    for item in MENU:
        cat = item.get("category", "Other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item)
    
    for cat, items in categories.items():
        st.markdown(f'<div class="menu-category">{cat}</div>', unsafe_allow_html=True)
        for item in items:
            note = f" ({item['note']})" if item.get('note') else ""
            st.markdown(
                f"<div class='menu-item'>{item['name']}{note} "
                f"<span class='menu-price'>{item['price']:,} VND</span></div>",
                unsafe_allow_html=True
            )
    
    st.markdown("---")
    
    model_files = check_model_files()
    model_ready = len(model_files) > 0
    
    status_html = render_status_dot(model_ready)
    if model_ready:
        st.markdown(f'{status_html} **MODEL READY**', unsafe_allow_html=True)
        for mf in model_files:
            st.markdown(f'<div class="model-info">▸ {mf["name"]} ({mf["size"]:.1f} MB)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'{status_html} **MODEL NOT FOUND**', unsafe_allow_html=True)
        st.markdown('<div class="model-info">Upload food_model.onnx</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(f'<div class="model-info">{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>', unsafe_allow_html=True)


col_left, col_right = st.columns([2.5, 1.5])

with col_left:
    st.markdown("### Camera")
    st.markdown("---")
    
    camera_image = st.camera_input("Take a photo", key="camera")
    
    if camera_image is not None:
        image = load_image(camera_image)
        st.image(image, caption="Captured Image", use_column_width=True)
        
        if st.button("Recognize", use_container_width=True):
            with st.spinner("Processing..."):
                session = load_model()
                
                if session is None:
                    st.error("Model not found")
                else:
                    st.session_state['image'] = image
                    st.session_state['session'] = session
                    st.session_state['processed'] = True
                    st.rerun()
    else:
        st.info("Click the camera button above to take a photo")
        st.markdown('<div class="camera-container"><div class="camera-label">Ready to capture</div></div>', unsafe_allow_html=True)


with col_right:
    st.markdown("### Results")
    st.markdown("---")
    
    if 'processed' in st.session_state and st.session_state['processed']:
        image = st.session_state['image']
        session = st.session_state['session']
        
        with st.spinner("Segmenting..."):
            cropped_results, img_resized = crop_food_items(image)
        
        if cropped_results:
            img_with_boxes = draw_boxes_fixed(img_resized, cropped_results)
            st.image(img_with_boxes, caption="Detected Trays", use_column_width=True)
            
            st.success(f"{len(cropped_results)} tray(s) detected")
            
            detected_foods = []
            food_details = []
            extras = {}
            
            for idx, result in enumerate(cropped_results):
                cropped_img = result["image"]
                khay_id = result["id"]
                
                with st.expander(f"TRAY {khay_id}", expanded=(idx == 0)):
                    col_img, col_info = st.columns([1, 1.5])
                    
                    with col_img:
                        try:
                            h, w = cropped_img.shape[:2]
                            scale = min(150 / h, 150 / w)
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
                            
                            st.markdown('<div style="font-size:0.5rem;color:#999;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.3rem;">Predictions</div>', unsafe_allow_html=True)
                            
                            for i, (fid, conf) in enumerate(zip(top3_idx, top3_conf)):
                                name = get_food_name(fid)
                                price = get_food_price(fid)
                                if i == 0:
                                    st.markdown(f'<div class="prediction-item"><span class="main">▸ {name} <span class="conf">{conf*100:.1f}%</span></span><br><span class="sub">{price:,} VND</span></div>', unsafe_allow_html=True)
                                    detected_foods.append(fid)
                                    
                                    if has_extra_option(fid):
                                        extra_key = len(detected_foods) - 1
                                        egg_count = st.number_input(
                                            "Extra eggs",
                                            min_value=0,
                                            max_value=10,
                                            value=0,
                                            step=1,
                                            key=f"egg_{khay_id}",
                                            help="Add extra eggs (+6,000 VND each)"
                                        )
                                        if egg_count > 0:
                                            extras[extra_key] = egg_count
                                    
                                    food_details.append({
                                        "tray": khay_id,
                                        "name": name,
                                        "price": price,
                                        "confidence": conf
                                    })
                                else:
                                    st.markdown(f'<div class="prediction-item"><span class="alt">{name} <span class="conf-alt">{conf*100:.1f}%</span></span></div>', unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            if detected_foods:
                total_price, details = calculate_total(detected_foods, extras)
                
                st.markdown("---")
                st.markdown(f"""
                <div class="total-card">
                    <div class="label">Total</div>
                    <div class="amount">{total_price:,} VND</div>
                    <div class="summary">{len(detected_foods)} items · {len(set(detected_foods))} types</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### Payment")
                st.markdown("---")
                
                qr_img = generate_qr_code(total_price)
                st.markdown(f"""
                <div class="qr-container">
                    <img src="{qr_img}" alt="QR Code">
                    <div class="qr-amount">{total_price:,} VND</div>
                    <div class="qr-bank">MB BANK · 0393167129</div>
                    <div class="qr-label">LUONG NGOC THUAN</div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("Invoice Details", expanded=False):
                    for i, detail in enumerate(details):
                        extra_text = detail.get('extra_text', '')
                        st.markdown(
                            f"<div class='invoice-row'>"
                            f"<span>#{i+1}</span>"
                            f"<span>{detail['name']}{extra_text}</span>"
                            f"<span class='price'>{detail['price']:,} VND</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    
                    st.markdown(
                        f"<div class='invoice-total'>"
                        f"<span>TOTAL</span>"
                        f"<span class='total-price'>{total_price:,} VND</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    invoice_text = f"INVOICE\n{'-'*30}\n"
                    for i, d in enumerate(details):
                        extra_text = d.get('extra_text', '')
                        invoice_text += f"Tray {i+1}: {d['name']}{extra_text} - {d['price']:,} VND\n"
                    invoice_text += f"{'-'*30}\nTOTAL: {total_price:,} VND"
                    
                    st.download_button(
                        label="Download Invoice",
                        data=invoice_text,
                        file_name=f"invoice_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain"
                    )
            
        else:
            st.warning("No trays detected")
        
        if st.button("Reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    else:
        st.info("Take a photo and click RECOGNIZE")
        st.markdown('<div style="font-size:0.5rem;color:#ccc;letter-spacing:2px;">▸ Camera capture</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.5rem;color:#ccc;letter-spacing:2px;">▸ Automatic segmentation</div>', unsafe_allow_html=True)


st.markdown("""
<div class="footer">
    <p>FOOD IMAGE RECOGNIZING <span class="copyright">2026 &copy;</span></p>
</div>
""", unsafe_allow_html=True)
