"""
Script demo để chạy và test mô hình phân tích cổ phiếu
"""

import pandas as pd

from groupPJ.stock_analysis_yfinance import StockAnalysisYFinance


def run_quick_demo():
    """
    Chạy demo nhanh với dữ liệu mẫu
    """
    print("=== DEMO MÔ HÌNH PHÂN TÍCH CỔ PHIẾU ===\n")

    # Khởi tạo mô hình
    model = StockAnalysisYFinance()

    print("1. Thu thập dữ liệu...")
    # Thu thập dữ liệu trong khoảng thời gian ngắn hơn để demo nhanh
    success = model.collect_stock_data(start_date='2023-01-01', end_date='2024-12-31')

    if not success:
        print("Không thể thu thập dữ liệu. Kiểm tra kết nối internet và API vnstock.")
        return

    # Hiển thị thông tin dữ liệu
    print(f"\n✓ Đã thu thập {len(model.data)} records")
    print(f"✓ Số cổ phiếu: {model.data['symbol'].nunique()}")
    print(f"✓ Thời gian: {model.data['date'].min()} đến {model.data['date'].max()}")

    print("\n2. Tiền xử lý dữ liệu...")
    if not model.preprocess_data():
        print("Lỗi trong quá trình tiền xử lý!")
        return

    print("\n3. Tính toán chỉ số kỹ thuật...")
    if not model.calculate_technical_indicators():
        print("Lỗi trong quá trình tính toán chỉ số kỹ thuật!")
        return

    print("\n4. Chuẩn bị features cho mô hình...")
    if not model.prepare_features():
        print("Lỗi trong quá trình chuẩn bị features!")
        return

    print("\n5. Huấn luyện mô hình...")
    if not model.train_model():
        print("Lỗi trong quá trình huấn luyện mô hình!")
        return

    print("\n6. Hiển thị độ quan trọng của features...")
    importance_df = model.feature_importance()

    print("\n7. Lưu mô hình...")
    model.save_model()

    print("\n8. Test dự đoán với một số cổ phiếu...")
    test_symbols = ['VIC', 'VHM', 'HPG', 'VCB', 'FPT']

    for symbol in test_symbols:
        print(f"\n--- Dự đoán cho {symbol} ---")
        predictions = model.predict_next_day(symbol, days=3)

        if predictions:
            current_price = model.processed_data[model.processed_data['symbol'] == symbol]['close'].iloc[-1]
            print(f"Giá hiện tại: {current_price:.2f} VND")
            print("Dự đoán 3 ngày tiếp theo:")
            for i, pred in enumerate(predictions, 1):
                change = ((pred - current_price) / current_price) * 100
                direction = "📈" if change > 0 else "📉"
                print(f"  Ngày {i}: {pred:.2f} VND ({change:+.2f}%) {direction}")

    print("\n=== HOÀN THÀNH DEMO ===")
    print("Mô hình đã được huấn luyện và test thành công!")
    print("Các file đã được tạo:")
    print("- stock_model.joblib (mô hình đã train)")
    print("- scaler.joblib (scaler để chuẩn hóa dữ liệu)")

def load_and_predict():
    """
    Tải mô hình đã lưu và thực hiện dự đoán
    """
    print("=== TẢI MÔ HÌNH VÀ DỰ ĐOÁN ===\n")

    model = StockAnalysisModel()

    # Tải mô hình đã lưu
    if not model.load_model():
        print("Không thể tải mô hình. Hãy chạy demo trước!")
        return

    # Cần có dữ liệu để dự đoán
    print("Thu thập dữ liệu mới nhất...")
    model.collect_stock_data(start_date='2024-01-01', end_date='2024-12-31')
    model.preprocess_data()
    model.calculate_technical_indicators()
    model.prepare_features()

    # Dự đoán
    symbol = 'VIC'
    print(f"\nDự đoán cho {symbol}:")
    predictions = model.predict_next_day(symbol, days=5)

    if predictions:
        for i, pred in enumerate(predictions, 1):
            print(f"Ngày {i}: {pred:.2f} VND")

def show_data_overview():
    """
    Hiển thị tổng quan về dữ liệu
    """
    print("=== TỔNG QUAN DỮ LIỆU ===\n")

    model = StockAnalysisModel()
    print("Danh sách 20 cổ phiếu lớn nhất được sử dụng:")

    for i, symbol in enumerate(model.top_stocks, 1):
        print(f"{i:2d}. {symbol}")

    print(f"\nTổng cộng: {len(model.top_stocks)} cổ phiếu")
    print("Thời gian dữ liệu: 2020-01-01 đến 2024-12-31")
    print("Các chỉ số kỹ thuật:")
    print("- Moving Average (MA): 10, 20, 50 ngày")
    print("- RSI (Relative Strength Index)")
    print("- MACD (Moving Average Convergence Divergence)")
    print("- Bollinger Bands")

if __name__ == "__main__":
    print("Chọn chế độ chạy:")
    print("1. Demo đầy đủ (huấn luyện mô hình mới)")
    print("2. Tải mô hình và dự đoán")
    print("3. Hiển thị tổng quan dữ liệu")

    choice = input("\nNhập lựa chọn (1/2/3): ").strip()

    if choice == "1":
        run_quick_demo()
    elif choice == "2":
        load_and_predict()
    elif choice == "3":
        show_data_overview()
    else:
        print("Lựa chọn không hợp lệ!")
        print("Chạy demo mặc định...")
        run_quick_demo()