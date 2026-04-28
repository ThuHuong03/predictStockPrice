"""Flask web app for VN30 multi-model benchmark dashboard."""

from datetime import datetime
import math
from pathlib import Path

from flask import Flask, jsonify, render_template, send_from_directory

from stock_analysis_yfinance import StockAnalysisYFinance

CURRENT_DIR = Path(__file__).parent
app = Flask(__name__)

model = None
model_loaded = False


def sanitize_for_json(value):
    """Convert NaN/Inf recursively to JSON-safe None."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    return value


def initialize_model() -> bool:
    """Initialize model bundle and refresh recent data for inference."""
    global model, model_loaded
    try:
        model = StockAnalysisYFinance()
        bundle_loaded = model.load_model_bundle()

        if not bundle_loaded:
            print("Model bundle not found. Running full pipeline...")
            if not model.run_full_pipeline(period="5y", refresh_cache=False):
                return False
            model_loaded = True
            return True

        selected_symbols = list(model.top_stocks)
        # Keep period consistent with 5-year benchmark/reporting sections.
        if not model.collect_stock_data_yfinance(period="5y", use_cache=True):
            return False
        model.top_stocks = selected_symbols
        model.preprocess_data()
        model.calculate_technical_indicators()
        model.add_hybrid_econometric_features()
        model.prepare_features()
        checks = model.evaluate_data_significance()
        existing_checks = model.benchmark_summary.get("data_significance_checks", {})
        if not existing_checks or not existing_checks.get("ml"):
            model.benchmark_summary["data_significance_checks"] = checks
        model_loaded = True
        print("Model initialized successfully.")
        return True
    except Exception as exc:
        print(f"Initialize error: {exc}")
        return False


@app.route("/")
def index():
    stocks_info = []
    for symbol in model.top_stocks if model else []:
        stocks_info.append({"symbol": symbol, "name": model.stock_name_map.get(symbol, symbol)})
    return render_template(
        "index.html",
        stocks=stocks_info,
        model_loaded=model_loaded,
        current_date=datetime.now().strftime("%d/%m/%Y"),
    )


@app.route("/stock/<symbol>")
def stock_detail(symbol):
    if not model_loaded or not model:
        return render_template("error.html", message="Mô hình chưa được tải.")
    current_info = model.get_current_stock_info(symbol)
    prediction = model.predict_tomorrow(symbol)
    if not current_info or not prediction:
        return render_template("error.html", message=f"Không thể dự báo cho mã {symbol}")
    stock_data = {
        "symbol": symbol,
        "name": model.stock_name_map.get(symbol, symbol),
        "current_info": current_info,
        "prediction": prediction,
    }
    return render_template("stock_detail.html", stock=stock_data)


@app.route("/api/predict/<symbol>")
def api_predict(symbol):
    if not model_loaded or not model:
        return jsonify({"success": False, "message": "Mô hình chưa được tải"})
    current_info = model.get_current_stock_info(symbol)
    prediction = model.predict_tomorrow(symbol)
    if not current_info or not prediction:
        return jsonify({"success": False, "message": f"Không có dữ liệu cho mã {symbol}"})
    return jsonify(
        {
            "success": True,
            "data": {
                "symbol": symbol,
                "name": model.stock_name_map.get(symbol, symbol),
                "current_info": current_info,
                "prediction": prediction,
            },
        }
    )


@app.route("/api/stocks")
def api_stocks():
    if not model_loaded or not model:
        return jsonify({"success": False, "message": "Mô hình chưa được tải"})
    results = []
    for symbol in model.top_stocks:
        current_info = model.get_current_stock_info(symbol)
        prediction = model.predict_tomorrow(symbol)
        if current_info and prediction:
            results.append(
                {
                    "symbol": symbol,
                    "name": model.stock_name_map.get(symbol, symbol),
                    "current_price": current_info["current_price"],
                    "predicted_price": prediction["predicted_price"],
                    "price_change": prediction["price_change"],
                    "price_change_percent": prediction["price_change_percent"],
                    "trend": prediction["trend"],
                    "trend_icon": prediction["trend_icon"],
                    "trend_color": prediction["trend_color"],
                    "confidence": prediction.get("confidence"),
                }
            )
    return jsonify({"success": True, "data": sanitize_for_json(results)})


@app.route("/api/benchmark")
def api_benchmark():
    if not model_loaded or not model:
        return jsonify({"success": False, "message": "Mô hình chưa được tải"})
    return jsonify({"success": True, "data": sanitize_for_json(model.get_benchmark_summary())})


@app.route("/api/eda")
def api_eda():
    if not model_loaded or not model:
        return jsonify({"success": False, "message": "Mô hình chưa được tải"})
    return jsonify({"success": True, "data": sanitize_for_json(model.get_eda_summary())})


@app.route("/api/top10-performance")
def api_top10_performance():
    if not model_loaded or not model:
        return jsonify({"success": False, "message": "Mô hình chưa được tải"})
    return jsonify({"success": True, "data": sanitize_for_json(model.get_top10_performance())})


@app.route("/artifacts/<path:filename>")
def serve_artifact(filename):
    return send_from_directory(CURRENT_DIR / "artifacts", filename)


@app.route("/api/refresh")
def api_refresh():
    global model_loaded
    if not model_loaded or not model:
        return jsonify({"success": False, "message": "Mô hình chưa được tải"})
    try:
        top_symbols = list(model.top_stocks)
        if not model.collect_stock_data_yfinance(period="5y", use_cache=True):
            return jsonify({"success": False, "message": "Không thể refresh dữ liệu Yahoo Finance"})
        model.top_stocks = top_symbols
        model.preprocess_data()
        model.calculate_technical_indicators()
        model.add_hybrid_econometric_features()
        model.prepare_features()
        latest_date = model.processed_data["date"].max()
        return jsonify(
            {
                "success": True,
                "message": "Dữ liệu đã được cập nhật",
                "latest_date": latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date),
                "total_records": int(len(model.processed_data)),
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)})


@app.route("/dashboard")
def dashboard():
    if not model_loaded or not model:
        return render_template("error.html", message="Mô hình chưa được tải")
    return render_template(
        "dashboard.html",
        current_date=datetime.now().strftime("%d/%m/%Y"),
        model_loaded=model_loaded,
    )


if __name__ == "__main__":
    print("=== KHOI DONG WEB APPLICATION ===")
    success = initialize_model()
    if success:
        print("Web: http://127.0.0.1:5000")
        print("Dashboard: http://127.0.0.1:5000/dashboard")
        print("API stocks: http://127.0.0.1:5000/api/stocks")
        # Disable debug reloader to avoid intermittent API failures during auto-restart.
        app.run(debug=False, host="127.0.0.1", port=5000)
    else:
        print("Khong the khoi dong web app")