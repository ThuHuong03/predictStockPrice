# VN30 Multi-Model Stock Analysis

Hệ thống phân tích và dự báo xu hướng cổ phiếu VN30 với pipeline end-to-end:
- Thu thập dữ liệu từ Yahoo Finance cho universe VN30.
- Chọn **Top 10 mã theo hiệu suất 5 năm** (tự động).
- Tiền xử lý dữ liệu, EDA và tạo artifacts trực quan.
- Tạo đặc trưng kỹ thuật + hybrid econometric features.
- So sánh nhiều mô hình: **RandomForest**, **XGBoost** (nếu có), **LSTM** (nếu có TensorFlow), và benchmark **ARIMA/GARCH**.
- Cung cấp Flask dashboard + REST API để xem kết quả.

## 1) Kiến trúc tổng quan

### Pipeline phân tích (`stock_analysis_yfinance.py`)
1. Thu thập dữ liệu VN30 (có cache tại `artifacts/cache`).
2. Xếp hạng hiệu suất và chọn Top 10.
3. Tiền xử lý (missing/outlier) và EDA.
4. Tạo technical indicators (EMA, RSI, MACD, Bollinger Bands, OBV, VWAP...).
5. Tạo hybrid econometric features (ARIMA/GARCH theo expanding window).
6. Chuẩn bị target classification: `TANG`, `SIDEWAY`, `GIAM`.
7. Train + đánh giá ML và econometric benchmark.
8. Lưu model bundle và metadata.

### Web layer (`web_app.py`)
- Tải `analysis_bundle.joblib` nếu đã có.
- Nếu chưa có bundle, tự chạy full pipeline để tạo model.
- Refresh dữ liệu gần nhất cho inference.
- Cung cấp dashboard và API.

## 2) Cài đặt

Yêu cầu:
- Python 3.10+
- Internet để tải dữ liệu Yahoo Finance

Cài dependencies:

```bash
pip install -r requirements.txt
```

## 3) Chạy ứng dụng

### Cách 1 (khuyến nghị) - dùng script khởi động nhanh

```bash
python start_web.py
```

Script này sẽ:
- kiểm tra các file model (`stock_model.joblib`, `scaler.joblib`, `analysis_bundle.joblib`);
- hỏi có train mới nếu chưa có model;
- tự khởi chạy Flask app sau khi sẵn sàng.

### Cách 2 - chạy web app trực tiếp

```bash
python web_app.py
```

`web_app.py` sẽ tự initialize model; nếu thiếu bundle thì sẽ chạy pipeline để tạo mới.

### Cách 3 - train trước, chạy sau

```bash
python stock_analysis_yfinance.py
python web_app.py
```

URL mặc định:
- Home: `http://127.0.0.1:5000`
- Dashboard: `http://127.0.0.1:5000/dashboard`
- API stocks: `http://127.0.0.1:5000/api/stocks`

## 4) API chính

- `GET /api/stocks`: danh sách dự báo cho Top 10 mã.
- `GET /api/predict/<symbol>`: dự báo chi tiết cho một mã.
- `GET /api/benchmark`: kết quả benchmark model.
- `GET /api/eda`: EDA summary + đường dẫn artifacts.
- `GET /api/top10-performance`: bảng hiệu suất Top 10.
- `GET /api/refresh`: làm mới dữ liệu thị trường.

## 5) Đánh giá mô hình

### Metrics classification
- Accuracy
- Precision (weighted)
- F1-score (weighted)
- ROC-AUC OVR (weighted, nếu có xác suất dự báo)

### Data significance checks
- **ML checks**: số lượng mẫu, NaN, phân bố lớp, near-zero variance, multicollinearity.
- **Econometric checks**: ADF, Ljung-Box, ARCH LM, độ dài chuỗi.
- Kết quả được lưu trong `data_significance_checks`.

### Benchmark checklist
- `walk_forward_cv`
- `baseline_comparison`
- `regime_analysis`
- `confidence_calibration` (ECE)
- `paper_trading` (return, Sharpe, max drawdown)

## 6) Artifacts và model files

Sau khi chạy pipeline, các file quan trọng:
- `analysis_bundle.joblib` (bundle chính cho web app)
- `stock_model.joblib` (giữ tương thích cũ)
- `scaler.joblib` (metadata tương thích cũ)
- `artifacts/eda/eda_summary.json`
- `artifacts/eda/histogram_daily_return.png`
- `artifacts/eda/boxplot_daily_return_by_symbol.png`
- `artifacts/eda/correlation_matrix.png`
- `artifacts/cache/*.parquet` (cache dữ liệu theo period)

## 7) Cấu trúc project (chính)

```text
predictStockPrice/
|-- stock_analysis_yfinance.py
|-- web_app.py
|-- start_web.py
|-- run_web_server.py
|-- templates/
|-- artifacts/
|   |-- cache/
|   |-- eda/
|   `-- templates/
|-- requirements.txt
`-- README.md
```

## 8) Lưu ý

- Dự án phục vụ mục tiêu học thuật/thực nghiệm, không phải khuyến nghị đầu tư.
- Lần chạy đầu có thể lâu do tải dữ liệu và train nhiều mô hình.
- Một số mô hình là tùy chọn theo dependency:
  - XGBoost: cần cài `xgboost`
  - LSTM: cần cài `tensorflow`
  - GARCH: cần cài `arch`