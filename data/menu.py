# Danh sách món ăn từ bảng - ĐÃ GỘP CANH CHUA
MENU = [
    {"id": 0, "name": "Com trang", "price": 10000, "note": "Mot gia tien nhieu hay it", "category": "Mon chinh"},
    {"id": 1, "name": "Dau hu sot ca", "price": 25000, "note": "", "category": "Mon chinh"},
    {"id": 2, "name": "Ca hu kho", "price": 30000, "note": "", "category": "Mon chinh"},
    {"id": 3, "name": "Thit kho trung", "price": 30000, "note": "Mot trung, them 1 trung + 6000 dong", "category": "Mon chinh"},
    {"id": 4, "name": "Thit kho", "price": 25000, "note": "Khong co trung", "category": "Mon chinh"},
    {"id": 5, "name": "Canh chua", "price": 25000, "note": "Co ca hoac khong ca", "category": "Canh"},
    {"id": 6, "name": "Canh rau", "price": 7000, "note": "Cai hay muong", "category": "Canh"},
    {"id": 7, "name": "Suon nuong", "price": 30000, "note": "", "category": "Mon chinh"},
    {"id": 8, "name": "Rau xao", "price": 10000, "note": "Lagim/cu san/dau que/dau dua", "category": "Rau"},
    {"id": 9, "name": "Trung chien", "price": 25000, "note": "Trung chien thit", "category": "Mon chinh"},
]

# Map ID cũ -> ID mới (gộp canh chua)
ID_MAP = {
    5: 5,  # Canh chua co ca -> Canh chua
    6: 5,  # Canh chua khong ca -> Canh chua
}

def get_food_name(food_id):
    """Lấy tên món ăn theo ID"""
    # Ánh xạ ID cũ sang ID mới
    if food_id in ID_MAP:
        food_id = ID_MAP[food_id]
    
    for item in MENU:
        if item["id"] == food_id:
            return item["name"]
    return "Khong xac dinh"

def get_food_price(food_id):
    """Lấy giá tiền theo ID"""
    if food_id in ID_MAP:
        food_id = ID_MAP[food_id]
    
    for item in MENU:
        if item["id"] == food_id:
            return item["price"]
    return 0

def get_food_category(food_id):
    """Lấy danh mục món ăn"""
    if food_id in ID_MAP:
        food_id = ID_MAP[food_id]
    
    for item in MENU:
        if item["id"] == food_id:
            return item["category"]
    return "Khac"

def get_food_note(food_id):
    """Lấy ghi chú món ăn"""
    if food_id in ID_MAP:
        food_id = ID_MAP[food_id]
    
    for item in MENU:
        if item["id"] == food_id:
            return item["note"]
    return ""

def calculate_total(detected_foods):
    """Tính tổng tiền từ danh sách món ăn đã detect"""
    total = 0
    details = []
    for food_id in detected_foods:
        # Ánh xạ ID cũ sang ID mới
        if food_id in ID_MAP:
            food_id = ID_MAP[food_id]
        
        name = get_food_name(food_id)
        price = get_food_price(food_id)
        category = get_food_category(food_id)
        total += price
        details.append({
            "name": name, 
            "price": price,
            "category": category
        })
    return total, details
