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

st.set_page_config(
    page_title="Food Detection System",
    page_icon="🍱",
    layout="wide"
)

# ==================== DANH SÁCH MÓN ĂN ====================
CLASS_NAMES = [
    'Cơm trắng', 'Đậu hũ sốt cà', 'Cá hú kho', 
    'Thịt kho trứng', 'Thịt kho', 'Canh chua có cá',
    'Canh chua không cá', 'Sườn nướng', 'Canh rau', 
    'Rau xào', 'Trứng chiên'
]

PRICES = [10000, 25000, 30000, 30000, 25000, 25000, 10000, 30000, 7000, 10000, 25000]

MENU = []
for i, name in enumerate(CLASS_NAMES):
    MENU.append({
        "id": i,
        "name": name,
        "price": PRICES[i],
        "category": "Món ăn"
    })

def get_food_name(food_id):
    if 0 <= food_id < len(CLASS_NAMES):
        return CLASS_NAMES[food_id]
    return "Unknown"

def get_food_price(food_id):
    if 0 <= food_id < len(PRICES):
        return PRICES[food_id]
    return 0

def calculate_total(food_ids):
    details = []
    total = 0
    for fid in food_ids:
        name = get_food_name(fid)
        price = get_food_price(fid)
        details.append({"name": name, "price": price})
        total += price
    return total, details

# ==================== HÀM XỬ LÝ ẢNH ====================
def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def preprocess_image(img, target_size=(224, 224)):
    img_resized = cv2.resize(img, target_size)
    img_array = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
    return img_array

def crop_food_items(image):
    img = cv2.resize(image, (1400, 1300))
    
    regions = {
        "vung_1": img[40:700, 40:760],
        "vung_2": img[40:700, 820:1380],
        "vung_3": img[760:1280, 30:500],
        "vung_4": img[760:1280, 520:920],
        "vung_5": img[760:1280, 950:1380]
    }
    
    cropped_results = []
    for i, (name, crop) in enumerate(regions.items()):
        if crop.size > 0:
            cropped_results.append({
                "id": i + 1,
                "name": name,
                "image": crop
            })
    
    return cropped_results, img

def draw_boxes_fixed(img, results):
    img_copy = img.copy()
    
    positions = {
        1: (40, 700, 40, 760),
        2: (40, 700, 820, 1380),
        3: (760, 1280, 30, 500),
        4: (760, 1280, 520, 920),
        5: (760, 1280, 950, 1380)
    }
    
    for result in results:
        idx = result["id"]
        if idx in positions:
            y1, y2, x1, x2 = positions[idx]
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img_copy, f"Tray {idx}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return img_copy

# ==================== QR CODE ====================
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

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    model_files = ['model.onnx', 'trainaicuoiky1.onnx', 'best_model.onnx']
    
    for name in model_files:
        if os.path.exists(name):
            try:
                session = ort.InferenceSession(name)
                return session
            except Exception:
                continue
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.onnx'):
                try:
                    session = ort.InferenceSession(os.path.join(root, file))
                    return session
                except Exception:
                    continue
    
    return None

# ==================== CSS ====================
st.markdown("""
<style>
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
    .menu-price {
        float: right;
        color: #000000;
        font-weight: 400;
    }
    .invoice-row {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px solid #f0f0f0;
        font-size: 0.65rem;
        color: #333333;
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
    .footer {
        text-align: center;
        padding: 1rem 0 0.3rem 0;
        color: #cccccc;
        font-size: 0.45rem;
        border-top: 1px solid #000000;
        margin-top: 1rem;
        letter-spacing: 2px;
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
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown("""
<div class="header">
    <h1>FOOD DETECTION SYSTEM</h1>
    <p>AUTOMATED MEAL RECOGNITION &amp; BILLING</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### MENU")
    st.markdown("---")
    
    for item in MENU:
        st.markdown(
            f"<div class='menu-item'>{item['name']} "
            f"<span class='menu-price'>{item['price']:,} VND</span></div>",
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    session = load_model()
    if session:
        st.success("✅ MODEL READY")
    else:
        st.error("❌ MODEL NOT FOUND")
        st.info("Upload file .onnx vào thư mục chính")
    
    st.markdown("---")
    st.caption(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# ==================== MAIN ====================
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
                
                st.markdown(f"""
                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="value">{len(detected_foods)}</div>
                        <div class="desc">Items</div>
                    </div>
                    <div class="summary-item">
                        <div class="value">{len(set(detected_foods))}</div>
                        <div class="desc">Types</div>
                    </div>
                    <div class="summary-item">
                        <div class="value">{total_price:,} VND</div>
                        <div class="desc">Total</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("INVOICE DETAILS", expanded=True):
                    invoice_html = ""
                    for i, detail in enumerate(details):
                        invoice_html += f"""
                        <div class="invoice-row">
                            <span>#{i+1}</span>
                            <span>{detail['name']}</span>
                            <span>{detail['price']:,} VND</span>
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
                                <div class="qr-label">{total_price:,} VND</div>
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception:
                            st.caption("QR generation error")
            
        else:
            st.warning("No trays detected")
        
        if st.button("RESET", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    else:
        st.info("Upload image and click Recognize")

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    <p>FOOD DETECTION SYSTEM · ONNX RUNTIME · 10 FOOD CLASSES</p>
</div>
""", unsafe_allow_html=True)
