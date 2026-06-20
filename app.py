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

from data.menu import MENU, get_food_name, get_food_price, calculate_total
from utils.image_processor import load_image, preprocess_image, crop_food_items, draw_boxes_fixed

st.set_page_config(
    page_title="Food Detection System",
    page_icon="",
    layout="wide"
)

def generate_qr_code(amount, bank_id="0393167129", bank_name="MB", account_name="LUONG NGOC THUAN"):
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
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&display=swap');
    * {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }
    .header {
        background: #ffffff;
        padding: 1.2rem 2rem;
        border: 1px solid #000000;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .header h1 {
        font-size: 1.2rem;
        font-weight: 300;
        letter-spacing: 4px;
        color: #000000;
        margin: 0;
    }
    .header p {
        font-size: 0.6rem;
        color: #666666;
        margin: 0.2rem 0 0 0;
        font-weight: 300;
        letter-spacing: 2px;
    }
    .menu-item {
        padding: 0.25rem 0;
        border-bottom: 1px solid #e0e0e0;
        font-size: 0.65rem;
        color: #333333;
        font-weight: 300;
    }
    .menu-item:hover {
        border-bottom: 1px solid #000000;
        padding-left: 0.5rem;
        transition: all 0.3s;
    }
    .menu-price {
        float: right;
        color: #000000;
        font-weight: 400;
    }
    .pushable {
        background: transparent;
        border: none;
        padding: 0;
        cursor: pointer;
        outline-offset: 4px;
        outline: none;
        width: 100%;
    }
    .pushable:focus {
        outline: none;
    }
    .front {
        display: block;
        padding: 0.6rem 1.5rem;
        font-size: 0.6rem;
        font-weight: 400;
        letter-spacing: 3px;
        text-transform: uppercase;
        background: #ffffff;
        color: #000000;
        transform: translateY(-4px);
        border: 1px solid #000000;
        transition: all 0.1s ease;
        box-shadow: 0 2px 0 #000000, 0 4px 0 #000000;
    }
    .pushable:hover .front {
        background: #000000;
        color: #ffffff;
        box-shadow: 0 1px 0 #000000, 0 2px 0 #000000;
        transform: translateY(-2px);
    }
    .pushable:active .front {
        box-shadow: 0 0px 0 #000000, 0 0px 0 #000000;
        transform: translateY(0px);
    }
    .pushable-secondary {
        background: transparent;
        border: none;
        padding: 0;
        cursor: pointer;
        outline-offset: 4px;
        outline: none;
        width: 100%;
    }
    .pushable-secondary .front-secondary {
        display: block;
        padding: 0.4rem 1.5rem;
        font-size: 0.5rem;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
        background: transparent;
        color: #666666;
        transform: translateY(-3px);
        border: 1px solid #cccccc;
        transition: all 0.1s ease;
        box-shadow: 0 1px 0 #cccccc, 0 2px 0 #cccccc;
    }
    .pushable-secondary:hover .front-secondary {
        color: #000000;
        border-color: #000000;
        box-shadow: 0 1px 0 #000000, 0 2px 0 #000000;
        transform: translateY(-2px);
    }
    .pushable-secondary:active .front-secondary {
        box-shadow: 0 0px 0 #000000, 0 0px 0 #000000;
        transform: translateY(0px);
    }
    .streamlit-expanderHeader {
        font-family: 'JetBrains Mono', monospace !important;
        background: #ffffff !important;
        border: 1px solid #000000 !important;
        border-radius: 0px !important;
        color: #000000 !important;
        font-size: 0.6rem !important;
        font-weight: 300 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
    }
    .streamlit-expanderHeader:hover {
        background: #000000 !important;
        color: #ffffff !important;
    }
    .streamlit-expanderContent {
        background: #ffffff !important;
        border: 1px solid #000000 !important;
        border-top: none !important;
        border-radius: 0px !important;
        padding: 0.8rem !important;
    }
    .total-card {
        background: #ffffff;
        border: 1px solid #000000;
        padding: 1.2rem;
        text-align: center;
        margin: 0.8rem 0;
    }
    .total-card .label {
        color: #666666;
        font-size: 0.55rem;
        font-weight: 300;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    .total-card .amount {
        color: #000000;
        font-size: 2rem;
        font-weight: 300;
        letter-spacing: 2px;
        margin: 0.2rem 0;
    }
    .total-card .summary {
        color: #666666;
        font-size: 0.55rem;
        font-weight: 300;
        letter-spacing: 1px;
    }
    .invoice-row {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px solid #f0f0f0;
        font-size: 0.65rem;
        color: #333333;
    }
    .invoice-row .tray {
        font-weight: 300;
        color: #666666;
    }
    .invoice-row .item-name {
        font-weight: 400;
    }
    .invoice-row .item-price {
        font-weight: 400;
        color: #000000;
    }
    .invoice-total {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-top: 2px solid #000000;
        font-size: 0.75rem;
        font-weight: 600;
        color: #000000;
        margin-top: 0.3rem;
    }
    .summary-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0.5rem;
        margin: 0.8rem 0;
    }
    .summary-item {
        border: 1px solid #000000;
        padding: 0.6rem;
        text-align: center;
        background: #ffffff;
    }
    .summary-item .value {
        font-size: 1.1rem;
        font-weight: 300;
        color: #000000;
    }
    .summary-item .desc {
        font-size: 0.45rem;
        color: #666666;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 0.1rem;
    }
    .qr-container {
        border: 1px solid #000000;
        padding: 0.8rem;
        text-align: center;
        background: #ffffff;
        margin: 0.5rem 0;
    }
    .qr-container img {
        max-width: 130px;
        height: auto;
    }
    .qr-container .qr-label {
        font-size: 0.45rem;
        color: #666666;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 0.2rem;
    }
    .qr-container .qr-amount {
        font-size: 1rem;
        font-weight: 300;
        color: #000000;
        margin: 0.15rem 0;
    }
    .qr-container .qr-bank {
        font-size: 0.5rem;
        color: #666666;
        font-weight: 300;
        letter-spacing: 1px;
    }
    .stAlert {
        border-radius: 0px !important;
        background: #f5f5f5 !important;
        border-left: 3px solid #000000 !important;
        color: #333333 !important;
        font-weight: 300 !important;
        font-size: 0.65rem !important;
    }
    .badge-high {
        color: #000000;
        font-size: 0.5rem;
        font-weight: 400;
        letter-spacing: 1px;
        border: 1px solid #000000;
        padding: 0.1rem 0.6rem;
        display: inline-block;
    }
    .badge-medium {
        color: #666666;
        font-size: 0.5rem;
        font-weight: 400;
        letter-spacing: 1px;
        border: 1px solid #666666;
        padding: 0.1rem 0.6rem;
        display: inline-block;
    }
    .badge-low {
        color: #999999;
        font-size: 0.5rem;
        font-weight: 400;
        letter-spacing: 1px;
        border: 1px solid #999999;
        padding: 0.1rem 0.6rem;
        display: inline-block;
    }
    .stImage figcaption {
        color: #666666 !important;
        font-size: 0.5rem !important;
        font-weight: 300 !important;
        letter-spacing: 1px !important;
        text-align: center !important;
    }
    .stFileUploader {
        background: #ffffff !important;
        border: 1px dashed #000000 !important;
        border-radius: 0px !important;
        padding: 0.6rem !important;
    }
    .stFileUploader:hover {
        border-style: solid !important;
    }
    .stFileUploader label {
        color: #666666 !important;
        font-weight: 300 !important;
        letter-spacing: 1px !important;
        font-size: 0.65rem !important;
    }
    .css-1d391kg {
        background-color: #ffffff !important;
        border-right: 1px solid #000000 !important;
    }
    .css-1d391kg .stMarkdown {
        color: #333333 !important;
    }
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #000000 !important;
        font-weight: 300 !important;
        letter-spacing: 2px !important;
        font-size: 0.9rem !important;
    }
    hr {
        border: none;
        border-top: 1px solid #000000;
        margin: 0.8rem 0;
    }
    .footer {
        text-align: center;
        padding: 1rem 0 0.3rem 0;
        color: #cccccc;
        font-size: 0.45rem;
        border-top: 1px solid #000000;
        margin-top: 1rem;
        letter-spacing: 2px;
    }
    ::-webkit-scrollbar {
        width: 3px;
        background: #f5f5f5;
    }
    ::-webkit-scrollbar-thumb {
        background: #000000;
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
    <p>AUTOMATED MEAL RECOGNITION &amp; BILLING</p>
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
            st.caption(f"{mf['name']} ({mf['size']:.1f} MB)")
    else:
        st.error("MODEL NOT FOUND")
    st.markdown("---")
    st.caption(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

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
        if st.button("RECOGNIZE", type="primary", use_container_width=True):
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
            st.success(f"{len(cropped_results)} trays detected")
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
                            scale = min(140 / h, 140 / w)
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
                                st.markdown('<span class="badge-high">HIGH</span>', unsafe_allow_html=True)
                            elif top3_conf[0] > 0.5:
                                st.markdown('<span class="badge-medium">MEDIUM</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="badge-low">LOW</span>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            if detected_foods:
                total_price, details = calculate_total(detected_foods)
                st.markdown("""
                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="value">{}</div>
                        <div class="desc">Items</div>
                    </div>
                    <div class="summary-item">
                        <div class="value">{}</div>
                        <div class="desc">Types</div>
                    </div>
                    <div class="summary-item">
                        <div class="value">{:,} VND</div>
                        <div class="desc">Total</div>
                    </div>
                </div>
                """.format(
                    len(detected_foods),
                    len(set(detected_foods)),
                    total_price
                ), unsafe_allow_html=True)
                with st.expander("INVOICE DETAILS", expanded=True):
                    invoice_html = ""
                    for i, detail in enumerate(details):
                        invoice_html += f"""
                        <div class="invoice-row">
                            <span class="tray">#{i+1}</span>
                            <span class="item-name">{detail['name']}</span>
                            <span class="item-price">{detail['price']:,} VND</span>
                        </div>
                        """
                    invoice_html += f"""
                    <div class="invoice-total">
                        <span>TOTAL</span>
                        <span>{total_price:,} VND</span>
                    </div>
                    """
                    st.markdown(invoice_html, unsafe_allow_html=True)
                    invoice_text = f"INVOICE\n{'-'*30}\n"
                    for i, d in enumerate(details):
                        invoice_text += f"Tray {i+1}: {d['name']} - {d['price']:,} VND\n"
                    invoice_text += f"{'-'*30}\nTOTAL: {total_price:,} VND"
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            label="DOWNLOAD INVOICE",
                            data=invoice_text,
                            file_name=f"invoice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    with col_dl2:
                        try:
                            qr_img = generate_qr_code(total_price)
                            st.markdown(f"""
                            <div class="qr-container">
                                <img src="{qr_img}" alt="QR Code">
                                <div class="qr-amount">{total_price:,} VND</div>
                                <div class="qr-bank">MB BANK · 0393167129</div>
                                <div class="qr-label">LUONG NGOC THUAN</div>
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.caption("QR generation error")
        else:
            st.warning("No trays detected")
        if st.button("RESET", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    else:
        st.info("Upload image and click Recognize")
        st.caption("Automatic segmentation · Real-time inference")

st.markdown("""
<div class="footer">
    <p>FOOD DETECTION SYSTEM v2.0 · ONNX RUNTIME · 10 FOOD CLASSES</p>
</div>
""", unsafe_allow_html=True)
