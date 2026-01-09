# Mô hình Phân tích và Dự đoán Giá Cổ phiếu

Dự án phát triển mô hình Machine Learning để phân tích và dự đoán giá cổ phiếu của **top 10 cổ phiếu** hàng đầu trên thị trường chứng khoán Việt Nam với giao diện web trực quan.

## 🎯 Tổng quan

Mô hình sử dụng dữ liệu training từ **toàn bộ năm 2025** và các chỉ số kỹ thuật để dự đoán giá cổ phiếu:
- **Top 10 cổ phiếu** hàng đầu trên thị trường Việt Nam
- **Training Data**: Dữ liệu toàn bộ năm 2025 (01/01/2025 - 31/12/2025)
- **Chỉ số kỹ thuật**: MA (10, 20, 50 ngày), RSI, MACD, Bollinger Bands, Volume Analysis
- **Thuật toán ML**: Random Forest Regressor với tối ưu hóa tham số
- **Nguồn dữ liệu**: Yahoo Finance API (yfinance)
- **Dự đoán**: Giá cổ phiếu ngày tiếp theo với độ chính xác cao

## 📊 Danh sách Cổ phiếu

| STT | Mã CK | Công ty |
|-----|-------|---------|
| 1   | VIC   | Vingroup |
| 2   | VHM   | Vinhomes |
| 3   | HPG   | Hòa Phát Group |
| 4   | VPB   | VPBank |
| 5   | VCB   | Vietcombank |
| 6   | BID   | BIDV |
| 7   | CTG   | VietinBank |
| 8   | TCB   | Techcombank |
| 9   | FPT   | FPT Corporation |
| 10  | VJC   | VietJet Air |

## 🔧 Cài đặt

### 1. Yêu cầu hệ thống
- Python 3.8+
- Kết nối internet (để tải dữ liệu từ API)

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Cấu trúc thư mục
```
groupPJ/
├── stock_analysis_yfinance.py # Lớp chính của mô hình (core model)
├── web_app.py                 # Web application (Flask)
├── start_web.py               # Script khởi động nhanh
├── demo.py                    # Script demo console
├── templates/                 # HTML templates
│   ├── base.html             # Template cơ bản
│   ├── index.html            # Trang chủ
│   ├── stock_detail.html     # Chi tiết cổ phiếu
│   ├── dashboard.html        # Dashboard tổng quan
│   └── error.html            # Trang lỗi
├── requirements.txt           # Dependencies
├── README.md                  # Hướng dẫn này
├── stock_model.joblib        # Mô hình đã train (tạo sau khi chạy)
└── scaler.joblib             # Scaler (tạo sau khi chạy)
```

## 🚀 Sử dụng

### 1. Khởi động nhanh (Khuyến nghị)
```bash
python start_web.py
```

Script này sẽ:
- Tự động kiểm tra mô hình đã huấn luyện
- Huấn luyện mô hình mới nếu chưa có
- Khởi động web application

### 2. Web Application
Sau khi khởi động, truy cập:
- **Trang chủ**: http://127.0.0.1:5000
- **Dashboard**: http://127.0.0.1:5000/dashboard
- **API**: http://127.0.0.1:5000/api/stocks

### 3. Chạy Demo Console
```bash
python demo.py
```

Chọn một trong các tùy chọn:
- `1`: Demo đầy đủ (huấn luyện mô hình mới)
- `2`: Tải mô hình và dự đoán
- `3`: Hiển thị tổng quan dữ liệu

### 2. Sử dụng trong code
```python
from stock_analysis_yfinance import StockAnalysisModel

# Khởi tạo mô hình
model = StockAnalysisModel()

# Thu thập dữ liệu
model.collect_stock_data()

# Tiền xử lý
model.preprocess_data()

# Tính toán chỉ số kỹ thuật
model.calculate_technical_indicators()

# Chuẩn bị features
model.prepare_features()

# Huấn luyện mô hình
model.train_model()

# Dự đoán
predictions = model.predict_next_day('VIC')
```

## 📈 Tính năng chính

### 1. Thu thập dữ liệu
- Sử dụng Yahoo Finance API (yfinance)
- **Training Data**: Toàn bộ năm 2025 (252 ngày giao dịch)
- Thông tin: Open, High, Low, Close, Volume
- Tự động cập nhật dữ liệu mới nhất cho dự đoán

### 2. Tiền xử lý dữ liệu
- Xử lý missing values bằng forward/backward fill
- Loại bỏ outliers sử dụng phương pháp IQR
- Chuẩn hóa dữ liệu với StandardScaler
- Tách dữ liệu train/test (80/20)

### 3. Chỉ số kỹ thuật
- **Moving Average (MA)**: 10, 20, 50 ngày
- **RSI**: Relative Strength Index (14 ngày)
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Upper, Lower, Middle bands
- **Volume Analysis**: Phân tích khối lượng giao dịch

### 4. Mô hình Machine Learning
- **Thuật toán**: Random Forest Regressor với tối ưu hóa
- **Features**: Giá OHLC + 5 chỉ số kỹ thuật chính
- **Target**: Giá đóng cửa ngày tiếp theo
- **Đánh giá**: MSE, RMSE, MAE, R² score
- **Auto-save**: Lưu mô hình tự động sau khi train

### 5. Web Application
- **Framework**: Flask với Bootstrap 4 UI
- **Trang chủ**:
  - Hiển thị 10 cổ phiếu với trạng thái mô hình
  - Giao diện card responsive
  - Navigation dễ sử dụng
- **Trang chi tiết cổ phiếu**:
  - Thông tin giá hiện tại
  - Dự đoán giá ngày tiếp theo
  - Phần trăm thay đổi và xu hướng (TĂNG/GIẢM)
- **Dashboard tổng quan**:
  - Xem toàn bộ dự đoán 10 cổ phiếu
  - So sánh performance các cổ phiếu
- **API Endpoints**:
  - `/api/predict/<symbol>`: Dự đoán cổ phiếu cụ thể
  - `/api/stocks`: Tất cả dự đoán
- **Real-time**: Dữ liệu và dự đoán cập nhật theo thời gian thực

## 📊 Kết quả mô hình

Mô hình được đánh giá qua các metrics:
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

## 🔍 Phân tích Features

Mô hình cung cấp:
- Biểu đồ độ quan trọng của features
- Ranking các chỉ số tác động đến dự đoán
- Phân tích correlation giữa các features

## 💾 Lưu trữ và Tải mô hình

```python
# Lưu mô hình
model.save_model('my_model.joblib', 'my_scaler.joblib')

# Tải mô hình
model.load_model('my_model.joblib', 'my_scaler.joblib')
```

## ⚠️ Lưu ý quan trọng

1. **Không phải lời khuyên đầu tư**: Mô hình chỉ mang tính chất nghiên cứu và học tập
2. **Rủi ro thị trường**: Giá cổ phiếu chịu tác động của nhiều yếu tố không dự đoán được
3. **Kiểm tra kết nối**: Cần kết nối internet ổn định để tải dữ liệu từ API
4. **Thời gian chạy**: Demo đầy đủ có thể mất 5-10 phút tùy thuộc tốc độ mạng

## 🛠️ Khắc phục sự cố

### Lỗi kết nối API
```bash
pip install --upgrade yfinance
```

### Lỗi dependencies
```bash
pip install --upgrade pandas scikit-learn matplotlib seaborn flask yfinance joblib
```

### Lỗi Port 5000 đã được sử dụng
```bash
# Kiểm tra process đang sử dụng port 5000
lsof -i :5000
# Hoặc chạy app trên port khác
python web_app.py --port 5001
```

### Lỗi memory (với dữ liệu lớn)
- Giảm số lượng cổ phiếu từ top 10 xuống (chỉnh sửa trong `top_stocks`)
- Model đã được tối ưu cho training data năm 2025 (252 ngày giao dịch)

## 📝 Tùy chỉnh

### Thêm cổ phiếu mới
Sửa danh sách `top_stocks` trong `StockAnalysisModel.__init__()`

### Thay đổi chỉ số kỹ thuật
Sửa các hàm `calculate_*` trong class

### Thử thuật toán khác
Thay thế `RandomForestRegressor` trong `train_model()`

## 📚 Tài liệu tham khảo

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Bootstrap Documentation](https://getbootstrap.com/)


## 📄 License

Dự án mang tính chất giáo dục và nghiên cứu.