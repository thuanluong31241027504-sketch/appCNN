# Danh sách món ăn từ bảng
MENU = [
    {"id": 0, "name": "Com trang", "price": 10000, "note": "Mot gia tien nhieu hay it"},
    {"id": 1, "name": "Dau hu sot ca", "price": 25000, "note": ""},
    {"id": 2, "name": "Ca hu kho", "price": 30000, "note": ""},
    {"id": 3, "name": "Thit kho trung", "price": 30000, "note": "Mot trung, them 1 trung + 6000 dong"},
    {"id": 4, "name": "Thit kho", "price": 25000, "note": "Khong co trung"},
    {"id": 5, "name": "Canh chua co ca", "price": 25000, "note": ""},
    {"id": 6, "name": "Canh chua khong ca", "price": 10000, "note": ""},
    {"id": 7, "name": "Suon nuong", "price": 30000, "note": ""},
    {"id": 8, "name": "Canh rau", "price": 7000, "note": "Cai hay muong"},
    {"id": 9, "name": "Rau xao", "price": 10000, "note": "Lagim/cu san/dau que/dau dua"},
    {"id": 10, "name": "Trung chien", "price": 25000, "note": "Trung chien thit"},
]

def get_food_name(food_id):
    """Lấy tên món ăn theo ID"""
    for item in MENU:
        if item["id"] == food_id:
            return item["name"]
    return "Khong xac dinh"

def get_food_price(food_id):
    """Lấy giá tiền theo ID"""
    for item in MENU:
        if item["id"] == food_id:
            return item["price"]
    return 0

def calculate_total(detected_foods):
    """Tính tổng tiền từ danh sách món ăn đã detect"""
    total = 0
    details = []
    for food_id in detected_foods:
        name = get_food_name(food_id)
        price = get_food_price(food_id)
        total += price
        details.append({"name": name, "price": price})
    return total, details
