"""
Script khởi động nhanh cho web application
Tự động huấn luyện mô hình nếu chưa có
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Lấy đường dẫn thư mục hiện tại
CURRENT_DIR = Path(__file__).parent
MODEL_PATH = CURRENT_DIR / "stock_model.joblib"
SCALER_PATH = CURRENT_DIR / "scaler.joblib"
BUNDLE_PATH = CURRENT_DIR / "analysis_bundle.joblib"

def check_model_exists():
    """Kiểm tra xem mô hình đã được huấn luyện chưa"""
    return MODEL_PATH.exists() and SCALER_PATH.exists() and BUNDLE_PATH.exists()

def train_model():
    """Huấn luyện mô hình mới"""
    print("🔄 Bắt đầu huấn luyện mô hình...")
    print("⏰ Thời gian bắt đầu:", datetime.now().strftime('%H:%M:%S'))

    try:
        # Thay đổi working directory để đảm bảo model được lưu đúng nơi
        os.chdir(CURRENT_DIR)

        from stock_analysis_yfinance import main
        main()
        return True
    except Exception as e:
        print(f"❌ Lỗi khi huấn luyện mô hình: {str(e)}")
        return False

def start_web_app():
    """Khởi động web application"""
    print("\n🚀 Khởi động web application...")

    try:
        # Thay đổi working directory
        os.chdir(CURRENT_DIR)

        from web_app import app, initialize_model

        # Khởi tạo mô hình
        success = initialize_model()

        if success:
            print("\n✅ Web application sẵn sàng!")
            print("🌐 Truy cập tại: http://127.0.0.1:5000")
            print("📊 Dashboard: http://127.0.0.1:5000/dashboard")
            print("🔗 API: http://127.0.0.1:5000/api/stocks")
            print("\nNhấn Ctrl+C để dừng server\n")

            # Chạy Flask app
            app.run(debug=False, host='127.0.0.1', port=5000)
        else:
            print("❌ Không thể khởi tạo mô hình!")
            return False

    except Exception as e:
        print(f"❌ Lỗi khi khởi động web app: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("🏢 STOCK PREDICTION WEB APPLICATION")
    print("📈 Dự báo giá cổ phiếu Việt Nam")
    print("=" * 60)
    print(f"📁 Thư mục làm việc: {CURRENT_DIR}")

    # Kiểm tra mô hình
    if check_model_exists():
        print("✅ Tìm thấy mô hình đã huấn luyện")
        print(f"   - Model: {MODEL_PATH}")
        print(f"   - Scaler: {SCALER_PATH}")
        print(f"   - Bundle: {BUNDLE_PATH}")
        start_web_app()
    else:
        print("⚠️  Không tìm thấy mô hình đã huấn luyện")
        print(f"   - Tìm kiếm tại: {MODEL_PATH}")

        # Hỏi người dùng có muốn huấn luyện không
        response = input("\n❓ Bạn có muốn huấn luyện mô hình mới không? (y/N): ").strip().lower()

        if response in ['y', 'yes', 'có']:
            print("\n🎯 Bắt đầu quá trình huấn luyện...")
            print("⏳ Quá trình này sẽ mất 5-10 phút...")

            if train_model():
                print("\n✅ Huấn luyện mô hình thành công!")
                input("\n🎉 Nhấn Enter để khởi động web application...")
                start_web_app()
            else:
                print("\n❌ Huấn luyện mô hình thất bại!")
                print("\n💡 Hướng dẫn khắc phục:")
                print("1. Kiểm tra kết nối internet")
                print("2. Cài đặt dependencies: pip install -r requirements.txt")
                print("3. Chạy thủ công: python stock_analysis_yfinance.py")
        else:
            print("\n💡 Để sử dụng web application, bạn cần huấn luyện mô hình trước:")
            print("1. Chạy: python stock_analysis_yfinance.py")
            print("2. Sau đó chạy: python web_app.py")
            print("\nHoặc chạy script này và chọn 'y' để tự động huấn luyện.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Đã dừng ứng dụng. Tạm biệt!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Lỗi không mong muốn: {str(e)}")
        sys.exit(1)