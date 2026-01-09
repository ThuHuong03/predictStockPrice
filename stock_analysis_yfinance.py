"""
Mô hình phân tích và dự đoán giá cổ phiếu - YAHOO FINANCE VERSION
Sử dụng yfinance API thay vì vnstock - ổn định và không bị dependency loop
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings('ignore')
from pathlib import Path

# Import thư viện ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Import yfinance
try:
    import yfinance as yf
except ImportError:
    print("Cài đặt yfinance: pip install yfinance")
    import sys
    sys.exit(1)

class StockAnalysisYFinance:
    """
    Lớp phân tích và dự đoán giá cổ phiếu sử dụng Yahoo Finance
    """

    def __init__(self):
        self.data = pd.DataFrame()
        self.processed_data = pd.DataFrame()
        self.model = None
        self.scaler = StandardScaler()

        # VN stocks với Yahoo Finance suffixes
        self.vn_stocks = {
            'VIC': 'VIC.VN',    # Vingroup
            'VHM': 'VHM.VN',    # Vinhomes
            'HPG': 'HPG.VN',    # Hòa Phát
            'VPB': 'VPB.VN',    # VPBank
            'VCB': 'VCB.VN',    # Vietcombank
            'BID': 'BID.VN',    # BIDV
            'CTG': 'CTG.VN',    # VietinBank
            'TCB': 'TCB.VN',    # Techcombank
            'FPT': 'FPT.VN',    # FPT Corp
            'VJC': 'VJC.VN'     # VietJet
        }

        self.top_stocks = list(self.vn_stocks.keys())
        print(f"🌍 Yahoo Finance: {len(self.top_stocks)} cổ phiếu VN")
        print(f"📊 Symbols: {', '.join(self.top_stocks)}")

    def test_yfinance_connection(self):
        """Test nhanh kết nối Yahoo Finance"""
        print("🔍 Testing Yahoo Finance connection...")

        try:
            # Test với VIC
            ticker = yf.Ticker("VIC.VN")
            test_data = ticker.history(period="5d")  # 5 ngày gần đây

            if not test_data.empty:
                print(f"✅ Connection OK! Sample VIC data: {len(test_data)} days")
                print(f"📅 Latest: {test_data.index[-1].date()} - Price: {test_data['Close'].iloc[-1]:.2f}")
                return True
            else:
                print("⚠️ No data received - might need different symbols")
                return False

        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            return False

    def collect_stock_data_yfinance(self, period="1y", max_retries=3):
        """
        Thu thập dữ liệu từ Yahoo Finance
        period: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
        """
        print(f"📥 Collecting data from Yahoo Finance (period: {period})...")

        all_data = []
        failed_symbols = []

        for i, (symbol, yf_symbol) in enumerate(self.vn_stocks.items(), 1):
            print(f"📈 [{i:2d}/{len(self.vn_stocks)}] {symbol} ({yf_symbol})...", end=" ", flush=True)

            success = False
            for attempt in range(max_retries):
                try:
                    # Tạo ticker object
                    ticker = yf.Ticker(yf_symbol)

                    # Lấy historical data
                    start_time = time.time()
                    hist_data = ticker.history(period=period, interval="1d")
                    fetch_time = time.time() - start_time

                    if not hist_data.empty and len(hist_data) > 0:
                        # Chuẩn bị dữ liệu
                        hist_data = hist_data.reset_index()
                        hist_data['symbol'] = symbol
                        hist_data['date'] = pd.to_datetime(hist_data['Date'])

                        # Rename columns to match our format
                        hist_data = hist_data.rename(columns={
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume'
                        })

                        # Select relevant columns
                        hist_data = hist_data[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

                        all_data.append(hist_data)
                        print(f"✅ OK ({len(hist_data)} records, {fetch_time:.1f}s)")
                        success = True
                        break

                    else:
                        if attempt < max_retries - 1:
                            print(f"⚠️ Empty data, retry {attempt + 1}...", end=" ", flush=True)
                            time.sleep(1)
                        else:
                            print("📭 EMPTY after retries")
                            failed_symbols.append(f"{symbol}(empty)")

                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"❌ Error, retry {attempt + 1}...", end=" ", flush=True)
                        time.sleep(2)
                    else:
                        print(f"💥 FAILED: {str(e)[:30]}...")
                        failed_symbols.append(f"{symbol}(error)")

            # Rate limiting
            time.sleep(0.3)

        # Tổng kết
        print(f"\n📊 COLLECTION SUMMARY:")
        print(f"   ✅ Success: {len(all_data)}/{len(self.vn_stocks)} stocks")

        if failed_symbols:
            print(f"   ❌ Failed: {', '.join(failed_symbols)}")

        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            print(f"🎉 Total: {len(self.data)} records from {len(all_data)} stocks")
            print(f"📅 Date range: {self.data['date'].min().date()} to {self.data['date'].max().date()}")
            return True
        else:
            print("💥 No data collected!")
            return False

    def preprocess_data(self):
        """Tiền xử lý dữ liệu"""
        print("🔧 Preprocessing data...")

        if self.data.empty:
            print("❌ No data to process!")
            return False

        df = self.data.copy()

        # Kiểm tra missing values
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            print(f"🔍 Found {missing_before} missing values, filling...")

            # Fill missing values
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df.groupby('symbol')[col].fillna(method='ffill')
                    df[col] = df.groupby('symbol')[col].fillna(method='bfill')

        # Sort by symbol and date
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

        self.processed_data = df
        print(f"✅ Preprocessed: {len(self.processed_data)} records")
        return True

    def calculate_technical_indicators(self):
        """Tính toán chỉ số kỹ thuật"""
        print("📊 Calculating technical indicators...")

        if self.processed_data.empty:
            return False

        df = self.processed_data.copy()

        for symbol in df['symbol'].unique():
            print(f"📈 {symbol}...", end=" ", flush=True)

            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy().sort_values('date')

            # Moving Averages
            symbol_data['MA_10'] = symbol_data['close'].rolling(window=10).mean()
            symbol_data['MA_20'] = symbol_data['close'].rolling(window=20).mean()
            symbol_data['MA_50'] = symbol_data['close'].rolling(window=50).mean()

            # RSI
            delta = symbol_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            symbol_data['RSI'] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = symbol_data['close'].ewm(span=12).mean()
            ema_26 = symbol_data['close'].ewm(span=26).mean()
            symbol_data['MACD'] = ema_12 - ema_26
            symbol_data['MACD_Signal'] = symbol_data['MACD'].ewm(span=9).mean()
            symbol_data['MACD_Histogram'] = symbol_data['MACD'] - symbol_data['MACD_Signal']

            # Bollinger Bands
            rolling_mean = symbol_data['close'].rolling(window=20).mean()
            rolling_std = symbol_data['close'].rolling(window=20).std()
            symbol_data['BB_Upper'] = rolling_mean + (rolling_std * 2)
            symbol_data['BB_Lower'] = rolling_mean - (rolling_std * 2)
            symbol_data['BB_Middle'] = rolling_mean

            # Update dataframe with technical indicators
            for col in ['MA_10', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'BB_Upper', 'BB_Lower', 'BB_Middle']:
                if col in symbol_data.columns:
                    df.loc[mask, col] = symbol_data[col].values
            print("✅")

        self.processed_data = df
        print("🎉 Technical indicators completed!")
        return True

    def prepare_features(self):
        """Chuẩn bị features cho ML"""
        print("🎯 Preparing ML features...")

        df = self.processed_data.copy()

        # Create target (next day close price)
        df['target'] = df.groupby('symbol')['close'].shift(-1)

        # Define features
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'MA_10', 'MA_20', 'MA_50', 'RSI',
            'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Upper', 'BB_Lower', 'BB_Middle'
        ]

        # Remove last row of each symbol (no target)
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        last_indices = df.groupby('symbol').tail(1).index
        df = df.drop(last_indices).reset_index(drop=True)

        # Remove NaN
        initial_rows = len(df)
        df = df.dropna()
        removed_rows = initial_rows - len(df)

        if removed_rows > 0:
            print(f"🧹 Removed {removed_rows} NaN rows")

        self.processed_data = df
        self.feature_cols = feature_cols

        print(f"✅ {len(feature_cols)} features, {len(df)} samples ready")
        return len(df) > 0

    def train_model(self, test_size=0.2):
        """Train Random Forest model"""
        print("🤖 Training Random Forest model...")

        if self.processed_data.empty:
            return False

        df = self.processed_data.copy()
        X = df[self.feature_cols]
        y = df['target']

        print(f"📊 Dataset: {len(X)} samples × {len(self.feature_cols)} features")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        start_time = time.time()

        self.model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=2
        )

        self.model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"✅ Training completed in {training_time:.1f}s")
        print(f"\n🏆 MODEL PERFORMANCE:")
        print(f"   📈 R² Score: {r2:.4f}")
        print(f"   📉 RMSE: {rmse:.4f}")
        print(f"   📊 MAE: {mae:.4f}")

        return True

    def save_model(self):
        """Save model và scaler"""
        if self.model is None:
            return False

        current_dir = Path(__file__).parent
        model_path = current_dir / 'stock_model.joblib'
        scaler_path = current_dir / 'scaler.joblib'

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        print(f"💾 Model saved: {model_path}")
        print(f"💾 Scaler saved: {scaler_path}")
        return True

    def get_current_stock_info(self, symbol):
        """Lấy thông tin stock mới nhất"""
        try:
            latest_data = self.processed_data[self.processed_data['symbol'] == symbol].tail(1)
            if latest_data.empty:
                return None

            info = latest_data.iloc[0]
            return {
                'symbol': symbol,
                'date': info['date'].strftime('%Y-%m-%d'),
                'current_price': float(info['close']),
                'open': float(info['open']),
                'high': float(info['high']),
                'low': float(info['low']),
                'volume': int(info['volume']),
                'ma_10': float(info['MA_10']) if pd.notna(info['MA_10']) else None,
                'ma_20': float(info['MA_20']) if pd.notna(info['MA_20']) else None,
                'ma_50': float(info['MA_50']) if pd.notna(info['MA_50']) else None,
                'rsi': float(info['RSI']) if pd.notna(info['RSI']) else None,
                'macd': float(info['MACD']) if pd.notna(info['MACD']) else None,
                'bb_upper': float(info['BB_Upper']) if pd.notna(info['BB_Upper']) else None,
                'bb_lower': float(info['BB_Lower']) if pd.notna(info['BB_Lower']) else None,
            }
        except Exception as e:
            print(f"❌ Error getting {symbol}: {e}")
            return None

    def predict_tomorrow(self, symbol):
        """Dự đoán giá ngày mai"""
        if self.model is None:
            return None

        try:
            current_info = self.get_current_stock_info(symbol)
            if not current_info:
                return None

            # Get latest features
            latest_data = self.processed_data[self.processed_data['symbol'] == symbol].tail(1)
            if latest_data.empty:
                return None

            features = latest_data[self.feature_cols].values
            features_scaled = self.scaler.transform([features[0]])
            predicted_price = self.model.predict(features_scaled)[0]

            current_price = current_info['current_price']
            price_change = predicted_price - current_price
            price_change_percent = (price_change / current_price) * 100

            # Determine trend with SIDEWAY support (threshold: 0.75%)
            sideway_threshold = 0.75
            abs_change_percent = abs(price_change_percent)

            if abs_change_percent < sideway_threshold:
                trend = 'SIDEWAY'
                trend_icon = '➡️'
                trend_color = 'warning'
            elif price_change_percent > 0:
                trend = 'TĂNG'
                trend_icon = '📈'
                trend_color = 'success'
            else:
                trend = 'GIẢM'
                trend_icon = '📉'
                trend_color = 'danger'

            return {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change': price_change,
                'price_change_percent': price_change_percent,
                'trend': trend,
                'trend_icon': trend_icon,
                'trend_color': trend_color,
                'date': current_info['date']
            }
        except Exception as e:
            print(f"❌ Prediction error {symbol}: {e}")
            return None

def main():
    """Main pipeline using Yahoo Finance"""
    print("=" * 70)
    print("🌍 STOCK ANALYSIS MODEL - YAHOO FINANCE VERSION")
    print("📈 Real Vietnamese stock data from Yahoo Finance")
    print("=" * 70)

    start_time = time.time()
    model = StockAnalysisYFinance()

    try:
        # Test connection first
        if not model.test_yfinance_connection():
            print("❌ Cannot connect to Yahoo Finance. Check internet connection.")
            return

        # Step 1: Collect data
        print("\n🌐 STEP 1: COLLECT DATA FROM YAHOO FINANCE")
        if not model.collect_stock_data_yfinance(period="1y"):  # 1 year data
            print("❌ Failed to collect data!")
            return

        # Step 2: Preprocess
        print("\n🔧 STEP 2: PREPROCESS DATA")
        if not model.preprocess_data():
            return

        # Step 3: Technical indicators
        print("\n📊 STEP 3: CALCULATE TECHNICAL INDICATORS")
        if not model.calculate_technical_indicators():
            return

        # Step 4: Prepare features
        print("\n🎯 STEP 4: PREPARE ML FEATURES")
        if not model.prepare_features():
            return

        # Step 5: Train model
        print("\n🤖 STEP 5: TRAIN MODEL")
        if not model.train_model():
            return

        # Step 6: Save model
        print("\n💾 STEP 6: SAVE MODEL")
        model.save_model()

        # Step 7: Test predictions
        print("\n🔮 STEP 7: TEST PREDICTIONS")
        for symbol in list(model.top_stocks)[:5]:  # Test first 5
            prediction = model.predict_tomorrow(symbol)
            if prediction:
                print(f"📊 {symbol}: {prediction['current_price']:.0f} → {prediction['predicted_price']:.0f} ({prediction['price_change_percent']:+.1f}%) {prediction['trend_icon']}")

        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("🎉 YAHOO FINANCE VERSION COMPLETED!")
        print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print("✅ Real VN stock data successfully processed!")
        print("🚀 Ready for web application!")
        print("=" * 70)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()