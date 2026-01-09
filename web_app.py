"""
Web Application để hiển thị thông tin cổ phiếu và dự báo giá
Sử dụng Flask framework
"""

from flask import Flask, render_template, request, jsonify
import json
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Import Yahoo Finance model
from stock_analysis_yfinance import StockAnalysisYFinance

# Lấy đường dẫn thư mục hiện tại
CURRENT_DIR = Path(__file__).parent

app = Flask(__name__)

# Global variables
model = None
model_loaded = False

# Thông tin tên công ty cho các mã cổ phiếu
STOCK_NAMES = {
    'VIC': 'Vingroup',
    'VHM': 'Vinhomes',
    'HPG': 'Hòa Phát Group',
    'VPB': 'VPBank',
    'VCB': 'Vietcombank',
    'BID': 'BIDV',
    'CTG': 'VietinBank',
    'TCB': 'Techcombank',
    'MSN': 'Masan Group',
    'GAS': 'PetroVietnam Gas',
    'VRE': 'Vincom Retail',
    'TPB': 'TPBank',
    'PLX': 'Petrolimex',
    'MWG': 'Mobile World',
    'MBB': 'MBBank',
    'SSI': 'SSI Securities',
    'POW': 'PetroVietnam Power',
    'FPT': 'FPT Corporation',
    'VJC': 'VietJet Air',
    'HDB': 'HDBank'
}

def initialize_model():
    """
    Khởi tạo và tải mô hình
    """
    global model, model_loaded

    try:
        print("Khởi tạo Yahoo Finance mô hình...")
        model = StockAnalysisYFinance()

        # Thử tải mô hình đã lưu
        model_path = CURRENT_DIR / 'stock_model.joblib'
        if model_path.exists():
            print("Tìm thấy mô hình đã lưu, đang tải...")

            # Load model và scaler files
            import joblib
            model.model = joblib.load(model_path)
            model.scaler = joblib.load(CURRENT_DIR / 'scaler.joblib')

            # Thu thập dữ liệu mới nhất từ Yahoo Finance để dự đoán
            print("Thu thập dữ liệu Yahoo Finance cho predictions...")
            if model.collect_stock_data_yfinance(period="3mo"):  # 3 tháng data cho predictions
                model.preprocess_data()
                model.calculate_technical_indicators()
                model.prepare_features()
                model_loaded = True
                print("Yahoo Finance model đã sẵn sáng!")
                return True
            else:
                print("Không thể thu thập dữ liệu từ Yahoo Finance!")
                return False

        print("Không tìm thấy mô hình đã lưu. Cần huấn luyện mô hình mới.")
        print("Chạy python stock_analysis_model.py trước để huấn luyện mô hình.")
        return False

    except Exception as e:
        print(f"Lỗi khi khởi tạo mô hình: {str(e)}")
        return False

@app.route('/')
def index():
    """
    Trang chủ - hiển thị danh sách cổ phiếu
    """
    stocks_info = []
    for symbol in model.top_stocks if model else []:
        stocks_info.append({
            'symbol': symbol,
            'name': STOCK_NAMES.get(symbol, symbol)
        })

    return render_template('index.html',
                         stocks=stocks_info,
                         model_loaded=model_loaded,
                         current_date=datetime.now().strftime('%d/%m/%Y'))

@app.route('/stock/<symbol>')
def stock_detail(symbol):
    """
    Trang chi tiết cổ phiếu
    """
    if not model_loaded or not model:
        return render_template('error.html', message="Mô hình chưa được tải. Vui lòng chạy huấn luyện mô hình trước.")

    try:
        # Lấy thông tin hiện tại
        current_info = model.get_current_stock_info(symbol)
        if not current_info:
            return render_template('error.html', message=f"Không có dữ liệu cho mã {symbol}")

        # Lấy dự báo
        prediction = model.predict_tomorrow(symbol)
        if not prediction:
            return render_template('error.html', message=f"Không thể dự báo cho mã {symbol}")

        # Chuẩn bị dữ liệu
        stock_data = {
            'symbol': symbol,
            'name': STOCK_NAMES.get(symbol, symbol),
            'current_info': current_info,
            'prediction': prediction
        }

        return render_template('stock_detail.html', stock=stock_data)

    except Exception as e:
        return render_template('error.html', message=f"Lỗi khi xử lý dữ liệu: {str(e)}")

@app.route('/api/predict/<symbol>')
def api_predict(symbol):
    """
    API endpoint để lấy dự báo cho một cổ phiếu
    """
    if not model_loaded or not model:
        return jsonify({
            'success': False,
            'message': 'Mô hình chưa được tải'
        })

    try:
        # Lấy thông tin hiện tại
        current_info = model.get_current_stock_info(symbol)
        if not current_info:
            return jsonify({
                'success': False,
                'message': f'Không có dữ liệu cho mã {symbol}'
            })

        # Lấy dự báo
        prediction = model.predict_tomorrow(symbol)
        if not prediction:
            return jsonify({
                'success': False,
                'message': f'Không thể dự báo cho mã {symbol}'
            })

        return jsonify({
            'success': True,
            'data': {
                'symbol': symbol,
                'name': STOCK_NAMES.get(symbol, symbol),
                'current_info': current_info,
                'prediction': prediction
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/api/stocks')
def api_stocks():
    """
    API endpoint để lấy danh sách tất cả cổ phiếu với dự báo
    """
    if not model_loaded or not model:
        return jsonify({
            'success': False,
            'message': 'Mô hình chưa được tải'
        })

    try:
        results = []
        for symbol in model.top_stocks:
            try:
                current_info = model.get_current_stock_info(symbol)
                prediction = model.predict_tomorrow(symbol)

                if current_info and prediction:
                    results.append({
                        'symbol': symbol,
                        'name': STOCK_NAMES.get(symbol, symbol),
                        'current_price': current_info['current_price'],
                        'predicted_price': prediction['predicted_price'],
                        'price_change': prediction['price_change'],
                        'price_change_percent': prediction['price_change_percent'],
                        'trend': prediction['trend'],
                        'trend_icon': prediction['trend_icon'],
                        'trend_color': prediction['trend_color']
                    })
            except Exception as e:
                print(f"Lỗi khi xử lý {symbol}: {str(e)}")
                continue

        return jsonify({
            'success': True,
            'data': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/dashboard')
def dashboard():
    """
    Dashboard tổng quan tất cả cổ phiếu
    """
    if not model_loaded or not model:
        return render_template('error.html', message="Mô hình chưa được tải")

    return render_template('dashboard.html',
                         current_date=datetime.now().strftime('%d/%m/%Y'),
                         model_loaded=model_loaded)

if __name__ == '__main__':
    print("=== KHỞI ĐỘNG WEB APPLICATION ===")

    # Khởi tạo mô hình
    success = initialize_model()

    if success:
        print("\n🚀 Web server đang chạy tại: http://127.0.0.1:5000")
        print("📊 Dashboard: http://127.0.0.1:5000/dashboard")
        print("🔍 API: http://127.0.0.1:5000/api/stocks")
        print("\nNhấn Ctrl+C để dừng server\n")

        # Chạy Flask app
        app.run(debug=True, host='127.0.0.1', port=5000)
    else:
        print("\n❌ Không thể khởi động web app")
        print("💡 Hướng dẫn:")
        print("1. Chạy: python stock_analysis_model.py")
        print("2. Đợi huấn luyện mô hình hoàn tất")
        print("3. Chạy lại: python web_app.py")
        sys.exit(1)