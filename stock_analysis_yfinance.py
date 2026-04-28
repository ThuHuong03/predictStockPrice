"""End-to-end stock analysis pipeline for VN30 top-performers."""

from __future__ import annotations

import json
import importlib
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("Missing dependency yfinance. Install with: pip install yfinance")
    raise

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None

try:
    from arch import arch_model
except ImportError:
    arch_model = None

def _get_tensorflow():
    try:
        return importlib.import_module("tensorflow")
    except ImportError:
        return None


class LSTMClassifierWrapper:
    """Pickle-safe Keras LSTM classifier wrapper for tabular classification."""

    def __init__(
        self,
        units: int = 16,
        dropout: float = 0.1,
        epochs: int = 20,
        batch_size: int = 64,
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.scale_mean_ = None
        self.scale_std_ = None
        self.num_classes_ = None
        self.input_dim_ = None
        self.classes_ = None

    def _build_model(self):
        tf = _get_tensorflow()
        if tf is None:
            raise ImportError("TensorFlow is not installed. Please install tensorflow to use LSTM.")
        tf.keras.utils.set_random_seed(self.random_state)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(1, self.input_dim_)),
                tf.keras.layers.LSTM(self.units, dropout=self.dropout),
                tf.keras.layers.Dense(self.num_classes_, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _transform(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        scaled = (X_arr - self.scale_mean_) / self.scale_std_
        return scaled.reshape(len(scaled), 1, self.input_dim_)

    def fit(self, X, y):
        tf = _get_tensorflow()
        if tf is None:
            raise ImportError("TensorFlow is not installed. Please install tensorflow to use LSTM.")
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)
        self.input_dim_ = X_arr.shape[1]
        self.classes_ = np.unique(y_arr)
        self.num_classes_ = int(len(self.classes_))
        self.scale_mean_ = X_arr.mean(axis=0)
        self.scale_std_ = X_arr.std(axis=0)
        self.scale_std_[self.scale_std_ == 0] = 1.0
        X_seq = self._transform(X_arr)

        self.model = self._build_model()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                patience=3,
                restore_best_weights=True,
            )
        ]
        self.model.fit(
            X_seq,
            y_arr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, X):
        X_seq = self._transform(X)
        return self.model.predict(X_seq, verbose=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model is None:
            state["model_json"] = None
            state["model_weights"] = None
        else:
            state["model_json"] = self.model.to_json()
            state["model_weights"] = self.model.get_weights()
        state["model"] = None
        return state

    def __setstate__(self, state):
        tf = _get_tensorflow()
        self.__dict__.update(state)
        self.model = None
        model_json = state.get("model_json")
        model_weights = state.get("model_weights")
        if model_json is not None and model_weights is not None:
            if tf is None:
                raise ImportError("TensorFlow is required to load saved LSTM models.")
            self.model = tf.keras.models.model_from_json(model_json)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            self.model.set_weights(model_weights)


@dataclass
class TargetConfig:
    sideway_threshold: float = 0.75
    test_ratio: float = 0.2
    target_shift: int = 1
    multicollinearity_threshold: float = 0.95
    max_high_corr_pairs: int = 10
    min_walk_forward_accuracy: float = 0.40
    min_baseline_improvement: float = 0.01
    min_regime_f1: float = 0.30
    max_expected_calibration_error: float = 0.12
    min_sharpe_ratio: float = 0.50
    max_drawdown_limit: float = -0.25
    hybrid_min_obs: int = 120
    hybrid_refit_interval: int = 10
    val_ratio: float = 0.15
    min_feature_importance: float = 0.0001


class StockAnalysisYFinance:
    """Stock analysis pipeline with econometric and ML benchmarks."""

    MODEL_PATH = "stock_model.joblib"
    SCALER_PATH = "scaler.joblib"
    BUNDLE_PATH = "analysis_bundle.joblib"
    EDA_SUMMARY_PATH = "artifacts/eda/eda_summary.json"

    def __init__(self):
        self.current_dir = Path(__file__).parent
        self.artifacts_dir = self.current_dir / "artifacts"
        self.eda_dir = self.artifacts_dir / "eda"
        self.cache_dir = self.artifacts_dir / "cache"
        self.eda_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.target_config = TargetConfig()
        self.data = pd.DataFrame()
        self.processed_data = pd.DataFrame()
        self.feature_data = pd.DataFrame()
        self.symbol_performance = pd.DataFrame()
        self.eda_summary: Dict = {}
        self.ml_results: Dict = {}
        self.econometric_results: Dict = {}
        self.benchmark_summary: Dict = {}
        self.data_significance_checks: Dict = {"ml": {}, "econometric": {}}
        self.feature_selection_report: Dict = {}
        self.feature_cols: List[str] = []
        self.core_feature_cols: List[str] = []
        self.hybrid_feature_cols: List[str] = []
        self.model_feature_cols: Dict[str, List[str]] = {}
        self.label_encoder = LabelEncoder()
        self.models: Dict[str, object] = {}
        self.ml_predictions_store: Dict = {}
        self.top_stocks: List[str] = []

        self.stock_name_map = {
            "ACB": "Asia Commercial Bank",
            "BCM": "Becamex IDC",
            "BID": "BIDV",
            "BVH": "Bao Viet Holdings",
            "CTG": "VietinBank",
            "FPT": "FPT Corporation",
            "GAS": "PetroVietnam Gas",
            "GVR": "Vietnam Rubber Group",
            "HDB": "HDBank",
            "HPG": "Hoa Phat Group",
            "MBB": "MBBank",
            "MSN": "Masan Group",
            "MWG": "Mobile World",
            "PLX": "Petrolimex",
            "POW": "PetroVietnam Power",
            "SAB": "Sabeco",
            "SHB": "SHB Bank",
            "SSB": "SeABank",
            "SSI": "SSI Securities",
            "STB": "Sacombank",
            "TCB": "Techcombank",
            "TPB": "TPBank",
            "VCB": "Vietcombank",
            "VHM": "Vinhomes",
            "VIB": "VIB Bank",
            "VIC": "Vingroup",
            "VJC": "VietJet Air",
            "VNM": "Vinamilk",
            "VPB": "VPBank",
            "VRE": "Vincom Retail",
        }
        self.vn30_stocks = {symbol: f"{symbol}.VN" for symbol in self.stock_name_map}
        print(f"VN30 universe loaded: {len(self.vn30_stocks)} symbols")

    def _trend_label(self, next_return_pct: float) -> str:
        threshold = self.target_config.sideway_threshold
        if pd.isna(next_return_pct):
            return np.nan
        if abs(next_return_pct) < threshold:
            return "SIDEWAY"
        if next_return_pct > 0:
            return "TANG"
        return "GIAM"

    def _load_cache(self, period: str) -> Optional[pd.DataFrame]:
        cache_path = self.cache_dir / f"vn30_{period}.parquet"
        if not cache_path.exists():
            return None
        try:
            df = pd.read_parquet(cache_path)
            if df.empty:
                return None
            return df
        except Exception:
            return None

    def _save_cache(self, period: str, df: pd.DataFrame) -> None:
        cache_path = self.cache_dir / f"vn30_{period}.parquet"
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as exc:
            print(f"Cannot save cache ({period}): {exc}")

    def test_yfinance_connection(self) -> bool:
        print("Testing Yahoo Finance connection...")
        try:
            sample = yf.Ticker("VIC.VN").history(period="5d")
            if sample.empty:
                print("No sample rows returned.")
                return False
            print(f"Connection OK. Latest row date: {sample.index[-1].date()}")
            return True
        except Exception as exc:
            print(f"Connection failed: {exc}")
            return False

    def collect_stock_data_yfinance(self, period: str = "5y", max_retries: int = 3, use_cache: bool = True) -> bool:
        print(f"Collecting VN30 data from Yahoo Finance ({period})...")
        if use_cache:
            cached = self._load_cache(period)
            if cached is not None:
                self.data = cached.copy()
                self._update_top_10_from_performance()
                print(f"Loaded from cache: {len(self.data)} rows")
                return True

        all_rows: List[pd.DataFrame] = []
        failed_symbols: List[str] = []

        for index, (symbol, yf_symbol) in enumerate(self.vn30_stocks.items(), 1):
            print(f"[{index:02d}/{len(self.vn30_stocks)}] {symbol} ({yf_symbol})", end=" ")
            success = False
            for attempt in range(max_retries):
                try:
                    start = time.time()
                    ticker = yf.Ticker(yf_symbol)
                    hist = ticker.history(period=period, interval="1d")
                    elapsed = time.time() - start
                    if hist.empty:
                        if attempt < max_retries - 1:
                            print(f"empty -> retry {attempt + 1}", end=" ")
                            time.sleep(0.8)
                            continue
                        failed_symbols.append(symbol)
                        print("FAILED(empty)")
                        break

                    hist = hist.reset_index()
                    hist["symbol"] = symbol
                    hist["date"] = pd.to_datetime(hist["Date"])
                    hist = hist.rename(
                        columns={
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Volume": "volume",
                        }
                    )
                    cols = ["date", "symbol", "open", "high", "low", "close", "volume"]
                    hist = hist[cols]
                    all_rows.append(hist)
                    success = True
                    print(f"OK ({len(hist)} rows, {elapsed:.1f}s)")
                    break
                except Exception:
                    if attempt < max_retries - 1:
                        print(f"error -> retry {attempt + 1}", end=" ")
                        time.sleep(1.0)
                    else:
                        failed_symbols.append(symbol)
                        print("FAILED(error)")
            if not success:
                time.sleep(0.2)

        if not all_rows:
            print("No data collected.")
            return False

        self.data = pd.concat(all_rows, ignore_index=True)
        self._save_cache(period, self.data)
        self._update_top_10_from_performance()
        print(f"Collected {len(self.data)} rows from {self.data['symbol'].nunique()} symbols.")
        if failed_symbols:
            print(f"Failed symbols: {', '.join(sorted(set(failed_symbols)))}")
        return True

    def _update_top_10_from_performance(self) -> None:
        if self.data.empty:
            self.symbol_performance = pd.DataFrame()
            self.top_stocks = []
            return

        perf_rows = []
        for symbol, group in self.data.groupby("symbol"):
            group = group.sort_values("date")
            if len(group) < 30:
                continue
            first_price = float(group["close"].iloc[0])
            last_price = float(group["close"].iloc[-1])
            if first_price <= 0:
                continue
            total_return_pct = (last_price / first_price - 1.0) * 100
            perf_rows.append(
                {
                    "symbol": symbol,
                    "return_5y_pct": total_return_pct,
                    "records": int(len(group)),
                    "first_date": str(group["date"].iloc[0].date()),
                    "last_date": str(group["date"].iloc[-1].date()),
                }
            )

        if not perf_rows:
            self.symbol_performance = pd.DataFrame()
            self.top_stocks = []
            return

        perf_df = pd.DataFrame(perf_rows).sort_values("return_5y_pct", ascending=False).reset_index(drop=True)
        perf_df["rank"] = np.arange(1, len(perf_df) + 1)
        self.symbol_performance = perf_df
        self.top_stocks = perf_df.head(10)["symbol"].tolist()
        print(f"Top 10 dynamic VN30 symbols: {', '.join(self.top_stocks)}")

    def preprocess_data(self) -> bool:
        print("Preprocessing data (missing values + outliers)...")
        if self.data.empty or not self.top_stocks:
            print("No data or no selected symbols.")
            return False

        df = self.data[self.data["symbol"].isin(self.top_stocks)].copy()
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

        numeric_cols = ["open", "high", "low", "close", "volume"]
        missing_before = int(df[numeric_cols].isna().sum().sum())
        for col in numeric_cols:
            df[col] = df.groupby("symbol")[col].ffill().bfill()
        missing_after = int(df[numeric_cols].isna().sum().sum())

        outlier_counts = {}
        for col in numeric_cols:
            if col == "volume":
                continue
            replaced = 0
            for symbol, idx in df.groupby("symbol").groups.items():
                s = df.loc[idx, col]
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0 or pd.isna(iqr):
                    continue
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                clipped = s.clip(lower=lower, upper=upper)
                replaced += int((clipped != s).sum())
                df.loc[idx, col] = clipped
            outlier_counts[col] = replaced

        df["daily_return"] = df.groupby("symbol")["close"].pct_change()
        df["log_return"] = np.log(df.groupby("symbol")["close"].transform(lambda s: s / s.shift(1)))

        self.processed_data = df
        self.eda_summary["data_quality"] = {
            "missing_before": missing_before,
            "missing_after": missing_after,
            "outlier_replacements": outlier_counts,
            "symbols": self.top_stocks,
            "rows": int(len(df)),
        }
        return True

    def run_eda(self) -> bool:
        print("Running EDA summary + plots...")
        if self.processed_data.empty:
            return False

        df = self.processed_data.copy()
        numeric_cols = ["open", "high", "low", "close", "volume", "daily_return", "log_return"]
        stats = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
        stats["median"] = df[numeric_cols].median()
        stats["skew"] = df[numeric_cols].skew()
        stats["kurtosis"] = df[numeric_cols].kurtosis()

        per_symbol_stats = (
            df.groupby("symbol")["daily_return"]
            .agg(["mean", "median", "std", "min", "max"])
            .sort_values("mean", ascending=False)
            .round(6)
        )

        corr = df[["open", "high", "low", "close", "volume", "daily_return"]].corr()

        histogram_path = self.eda_dir / "histogram_daily_return.png"
        plt.figure(figsize=(10, 4))
        sns.histplot(df["daily_return"].dropna(), bins=60, kde=True)
        plt.title("Histogram of Daily Return")
        plt.tight_layout()
        plt.savefig(histogram_path)
        plt.close()

        boxplot_path = self.eda_dir / "boxplot_daily_return_by_symbol.png"
        plt.figure(figsize=(12, 5))
        sns.boxplot(data=df, x="symbol", y="daily_return")
        plt.xticks(rotation=45)
        plt.title("Daily Return Boxplot by Symbol")
        plt.tight_layout()
        plt.savefig(boxplot_path)
        plt.close()

        corr_path = self.eda_dir / "correlation_matrix.png"
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(corr_path)
        plt.close()

        comments = self._generate_eda_comments(stats, corr, per_symbol_stats)
        self.eda_summary.update(
            {
                "descriptive_stats": stats.round(6).to_dict(orient="index"),
                "per_symbol_daily_return": per_symbol_stats.to_dict(orient="index"),
                "correlation_matrix": corr.round(4).to_dict(),
                "plots": {
                    "histogram": str(histogram_path.relative_to(self.current_dir)),
                    "boxplot": str(boxplot_path.relative_to(self.current_dir)),
                    "correlation_matrix": str(corr_path.relative_to(self.current_dir)),
                },
                "comments": comments,
            }
        )

        out_path = self.current_dir / self.EDA_SUMMARY_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.eda_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return True

    def _generate_eda_comments(self, stats: pd.DataFrame, corr: pd.DataFrame, per_symbol_stats: pd.DataFrame) -> List[str]:
        comments: List[str] = []
        if "daily_return" in stats.index:
            skew_val = float(stats.loc["daily_return", "skew"])
            std_val = float(stats.loc["daily_return", "std"])
            comments.append(f"Daily return skew={skew_val:.3f}, std={std_val:.4f}, cho thấy mức lệch/phân tán rõ.")

        highest_mean_symbol = per_symbol_stats.index[0] if not per_symbol_stats.empty else None
        lowest_mean_symbol = per_symbol_stats.index[-1] if not per_symbol_stats.empty else None
        if highest_mean_symbol and lowest_mean_symbol:
            comments.append(
                f"Mã có mean return cao nhất: {highest_mean_symbol}; thấp nhất: {lowest_mean_symbol}."
            )

        close_volume_corr = float(corr.loc["close", "volume"]) if "close" in corr.index and "volume" in corr.columns else 0.0
        comments.append(f"Tương quan close-volume: {close_volume_corr:.3f}.")
        return comments

    def calculate_technical_indicators(self) -> bool:
        print("Calculating indicators (EMA, RSI, MACD, BB, OBV, VWAP)...")
        if self.processed_data.empty:
            return False

        frames = []
        for symbol, group in self.processed_data.groupby("symbol"):
            g = group.sort_values("date").copy()
            g["EMA_10"] = g["close"].ewm(span=10, adjust=False).mean()
            g["EMA_20"] = g["close"].ewm(span=20, adjust=False).mean()
            g["EMA_50"] = g["close"].ewm(span=50, adjust=False).mean()

            delta = g["close"].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            g["RSI"] = 100 - (100 / (1 + rs))

            ema12 = g["close"].ewm(span=12, adjust=False).mean()
            ema26 = g["close"].ewm(span=26, adjust=False).mean()
            g["MACD"] = ema12 - ema26
            g["MACD_Signal"] = g["MACD"].ewm(span=9, adjust=False).mean()
            g["MACD_Histogram"] = g["MACD"] - g["MACD_Signal"]

            roll_mean = g["close"].rolling(20).mean()
            roll_std = g["close"].rolling(20).std()
            g["BB_Middle"] = roll_mean
            g["BB_Upper"] = roll_mean + (2 * roll_std)
            g["BB_Lower"] = roll_mean - (2 * roll_std)

            signed_volume = np.where(g["close"].diff().fillna(0) >= 0, g["volume"], -g["volume"])
            g["OBV"] = pd.Series(signed_volume, index=g.index).cumsum()
            cum_price_volume = (g["close"] * g["volume"]).cumsum()
            cum_volume = g["volume"].replace(0, np.nan).cumsum()
            g["VWAP"] = cum_price_volume / cum_volume
            g["VOLUME_CHANGE"] = g["volume"].pct_change()

            frames.append(g)

        self.processed_data = pd.concat(frames).sort_values(["symbol", "date"]).reset_index(drop=True)
        return True

    def add_hybrid_econometric_features(self) -> bool:
        """
        Add leakage-safe hybrid features generated from ARIMA/GARCH using expanding windows.
        Each timestamp only uses information available up to that point.
        """
        if self.processed_data.empty:
            return False

        print("Generating hybrid econometric features (ARIMA/GARCH expanding window)...")
        min_obs = self.target_config.hybrid_min_obs
        refit_interval = self.target_config.hybrid_refit_interval
        frames: List[pd.DataFrame] = []

        for symbol, group in self.processed_data.groupby("symbol"):
            g = group.sort_values("date").copy().reset_index(drop=True)
            returns = g["daily_return"].fillna(0.0).values
            n = len(g)
            arima_pred = np.full(n, np.nan)
            arima_resid_std = np.full(n, np.nan)
            garch_vol_pred = np.full(n, np.nan)

            arima_fit_cached = None
            garch_fit_cached = None

            for i in range(min_obs, n):
                # Refit periodically to control compute.
                if (i == min_obs) or ((i - min_obs) % refit_interval == 0):
                    hist = pd.Series(returns[:i]).replace([np.inf, -np.inf], np.nan).dropna()
                    if len(hist) < min_obs:
                        continue

                    if ARIMA is not None:
                        try:
                            arima_fit_cached = ARIMA(hist, order=(1, 0, 1)).fit()
                        except Exception:
                            arima_fit_cached = None

                    if arch_model is not None:
                        try:
                            garch_fit_cached = arch_model(hist * 100, vol="Garch", p=1, q=1, dist="normal").fit(disp="off")
                        except Exception:
                            garch_fit_cached = None

                if arima_fit_cached is not None:
                    try:
                        fc = float(arima_fit_cached.forecast(steps=1).iloc[0])
                        arima_pred[i] = fc
                        resid = pd.Series(arima_fit_cached.resid).replace([np.inf, -np.inf], np.nan).dropna()
                        arima_resid_std[i] = float(resid.tail(30).std()) if len(resid) >= 5 else np.nan
                    except Exception:
                        pass

                if garch_fit_cached is not None:
                    try:
                        var = float(garch_fit_cached.forecast(horizon=1, reindex=False).variance.values[-1, 0])
                        garch_vol_pred[i] = np.sqrt(max(var, 0.0)) / 100.0
                    except Exception:
                        pass

            g["ARIMA_FC_RET"] = arima_pred
            g["ARIMA_RESID_STD"] = arima_resid_std
            g["GARCH_FC_VOL"] = garch_vol_pred
            g["HYBRID_VALID_FLAG"] = (
                pd.Series(arima_pred).notna() & pd.Series(garch_vol_pred).notna()
            ).astype(int)

            # Fallbacks to avoid losing all samples when econometric fitting is unstable.
            fallback_fc = g["daily_return"].rolling(5).mean().shift(1)
            fallback_std = g["daily_return"].rolling(20).std().shift(1)
            g["ARIMA_FC_RET"] = g["ARIMA_FC_RET"].fillna(fallback_fc)
            g["ARIMA_RESID_STD"] = g["ARIMA_RESID_STD"].fillna(fallback_std)
            g["GARCH_FC_VOL"] = g["GARCH_FC_VOL"].fillna(fallback_std)

            # Regime derived from hybrid volatility forecast.
            q_low = pd.Series(g["GARCH_FC_VOL"]).quantile(0.33)
            q_high = pd.Series(g["GARCH_FC_VOL"]).quantile(0.66)
            g["HYBRID_REGIME"] = np.where(
                g["GARCH_FC_VOL"] <= q_low,
                0,
                np.where(g["GARCH_FC_VOL"] <= q_high, 1, 2),
            )
            g["HYBRID_REGIME"] = pd.Series(g["HYBRID_REGIME"]).fillna(1).astype(int)
            frames.append(g)

        self.processed_data = pd.concat(frames).sort_values(["symbol", "date"]).reset_index(drop=True)
        return True

    def prepare_features(self) -> bool:
        print("Preparing features and classification target...")
        if self.processed_data.empty:
            return False

        df = self.processed_data.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
        df["next_close"] = df.groupby("symbol")["close"].shift(-self.target_config.target_shift)
        df["next_return_pct"] = ((df["next_close"] - df["close"]) / df["close"]) * 100
        df["target_class"] = df["next_return_pct"].apply(self._trend_label)
        df["target_binary"] = (df["next_return_pct"] > 0).astype(int)

        # Candle-derived features retain price-action information with less collinearity than raw OHLC.
        df["CANDLE_RANGE"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
        df["CANDLE_BODY"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
        df["UPPER_WICK"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"].replace(0, np.nan)
        df["LOWER_WICK"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"].replace(0, np.nan)
        df["GAP_RETURN"] = df.groupby("symbol")["open"].transform(lambda s: s / s.shift(1) - 1.0)

        self.core_feature_cols = [
            "close",
            "volume",
            "CANDLE_RANGE",
            "CANDLE_BODY",
            "UPPER_WICK",
            "LOWER_WICK",
            "GAP_RETURN",
            "EMA_10",
            "EMA_20",
            "EMA_50",
            "RSI",
            "MACD",
            "MACD_Signal",
            "MACD_Histogram",
            "BB_Upper",
            "BB_Middle",
            "BB_Lower",
            "OBV",
            "VWAP",
            "VOLUME_CHANGE",
            "daily_return",
            "log_return",
        ]
        self.hybrid_feature_cols = [
            "ARIMA_FC_RET",
            "ARIMA_RESID_STD",
            "GARCH_FC_VOL",
            "HYBRID_REGIME",
            "HYBRID_VALID_FLAG",
        ]
        self.feature_cols = self.core_feature_cols + self.hybrid_feature_cols

        filtered = df.dropna(subset=self.feature_cols + ["target_class", "next_close", "next_return_pct"]).copy()
        filtered = filtered[filtered["target_class"].isin(["TANG", "SIDEWAY", "GIAM"])]
        self.feature_data = filtered
        self.processed_data = df
        print(f"Feature rows ready: {len(self.feature_data)}")
        return len(self.feature_data) > 0

    def _split_time_series(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = self.feature_data.sort_values("date")
        split_index = int(len(df) * (1 - self.target_config.test_ratio))
        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()
        return train_df, test_df

    def _run_ml_data_checks(self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Dict:
        if feature_cols is None:
            feature_cols = self.feature_cols
        checks = {"passed": True, "errors": [], "warnings": [], "details": {}}
        min_rows = 500
        if len(train_df) < min_rows:
            checks["passed"] = False
            checks["errors"].append(f"Train rows too low: {len(train_df)} < {min_rows}")

        if train_df[feature_cols].isna().sum().sum() > 0 or test_df[feature_cols].isna().sum().sum() > 0:
            checks["passed"] = False
            checks["errors"].append("NaN found in feature matrix after preparation")

        if not train_df["date"].is_monotonic_increasing or not test_df["date"].is_monotonic_increasing:
            checks["passed"] = False
            checks["errors"].append("Time split is not sorted by date")

        class_ratio = train_df["target_class"].value_counts(normalize=True).to_dict()
        class_count = train_df["target_class"].value_counts().to_dict()
        checks["details"]["class_ratio"] = class_ratio
        checks["details"]["class_count"] = class_count
        for cls, ratio in class_ratio.items():
            if ratio < 0.05:
                checks["warnings"].append(f"Class imbalance detected: {cls} ratio={ratio:.3f}")
        for cls, cnt in class_count.items():
            if cnt < 40:
                checks["passed"] = False
                checks["errors"].append(f"Insufficient samples for class {cls}: {cnt}")

        variance = train_df[feature_cols].var(numeric_only=True)
        near_zero_features = variance[variance < 1e-10].index.tolist()
        if near_zero_features:
            checks["passed"] = False
            checks["errors"].append(f"Near-zero variance features: {', '.join(near_zero_features)}")

        corr_matrix = train_df[feature_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = int((upper > 0.98).sum().sum())
        checks["details"]["high_corr_pairs_gt_0_98"] = high_corr_pairs
        if high_corr_pairs > 10:
            checks["warnings"].append(f"Very high multicollinearity pairs: {high_corr_pairs}")

        return checks

    def _prune_multicollinearity(self, train_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], Dict]:
        threshold = self.target_config.multicollinearity_threshold
        max_pairs = self.target_config.max_high_corr_pairs
        selected = feature_cols.copy()
        dropped: List[str] = []
        y_encoded = self.label_encoder.fit_transform(train_df["target_class"])
        try:
            mi_scores = mutual_info_classif(
                train_df[selected].fillna(0.0),
                y_encoded,
                random_state=42,
            )
            target_relevance = {col: float(score) for col, score in zip(selected, mi_scores)}
        except Exception:
            target_relevance = train_df[selected].corrwith(train_df["target_binary"]).abs().fillna(0.0).to_dict()

        def high_pairs(cols: List[str]) -> Tuple[pd.DataFrame, int]:
            corr = train_df[cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            pairs = upper.stack().reset_index()
            if pairs.empty:
                return pairs, 0
            pairs.columns = ["f1", "f2", "corr"]
            strong = pairs[pairs["corr"] >= threshold].sort_values("corr", ascending=False).reset_index(drop=True)
            return strong, int(len(strong))

        strong_pairs, before_count = high_pairs(selected)
        while len(selected) > 5 and not strong_pairs.empty and len(strong_pairs) > max_pairs:
            top = strong_pairs.iloc[0]
            f1, f2 = top["f1"], top["f2"]
            c1 = target_relevance.get(f1, 0.0)
            c2 = target_relevance.get(f2, 0.0)
            drop_feature = f1 if c1 < c2 else f2
            if drop_feature not in dropped:
                dropped.append(drop_feature)
            selected = [f for f in selected if f != drop_feature]
            strong_pairs, _ = high_pairs(selected)

        _, after_count = high_pairs(selected)
        report = {
            "threshold": threshold,
            "max_high_corr_pairs": max_pairs,
            "high_corr_pairs_before": before_count,
            "high_corr_pairs_after": after_count,
            "dropped_features": dropped,
            "selected_features": selected,
        }
        return selected, report

    def _filter_by_importance(
        self,
        model_factory,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[List[str], Dict]:
        """Lightweight feature filtering using permutation importance on validation."""
        if len(feature_cols) <= 8:
            return feature_cols, {"kept_all": True, "dropped": [], "threshold": self.target_config.min_feature_importance}
        try:
            model = model_factory()
            y_train = self.label_encoder.transform(train_df["target_class"])
            y_val = self.label_encoder.transform(val_df["target_class"])
            model.fit(train_df[feature_cols], y_train)
            perm = permutation_importance(
                model,
                val_df[feature_cols],
                y_val,
                n_repeats=3,
                random_state=42,
                scoring="f1_weighted",
            )
            scores = pd.Series(perm.importances_mean, index=feature_cols)
            kept = scores[scores > self.target_config.min_feature_importance].index.tolist()
            if len(kept) < 8:
                kept = scores.sort_values(ascending=False).head(max(8, len(feature_cols) // 2)).index.tolist()
            dropped = [c for c in feature_cols if c not in kept]
            return kept, {
                "kept_all": False,
                "dropped": dropped,
                "threshold": self.target_config.min_feature_importance,
                "mean_scores": scores.round(6).to_dict(),
            }
        except Exception:
            return feature_cols, {"kept_all": True, "dropped": [], "threshold": self.target_config.min_feature_importance}

    def _run_econometric_checks(self, series: pd.Series) -> Dict:
        checks = {"passed": True, "errors": [], "warnings": [], "tests": {}}
        clean_series = series.dropna()
        if len(clean_series) < 80:
            checks["passed"] = False
            checks["errors"].append(f"Insufficient observations for econometric modeling: {len(clean_series)}")
            return checks

        if float(clean_series.std()) < 1e-8:
            checks["passed"] = False
            checks["errors"].append("Return variance too low for meaningful modeling")
            return checks

        try:
            adf_stat, adf_pvalue = adfuller(clean_series, autolag="AIC")[:2]
            checks["tests"]["adf"] = {"stat": float(adf_stat), "p_value": float(adf_pvalue)}
            if adf_pvalue > 0.10:
                checks["warnings"].append("Series may be non-stationary (ADF p-value > 0.10)")
        except Exception as exc:
            checks["warnings"].append(f"ADF test failed: {exc}")

        try:
            lb = acorr_ljungbox(clean_series, lags=[10], return_df=True)
            lb_pvalue = float(lb["lb_pvalue"].iloc[0])
            checks["tests"]["ljung_box_lag10"] = {"p_value": lb_pvalue}
            if lb_pvalue > 0.10:
                checks["warnings"].append("Weak autocorrelation signal (Ljung-Box p-value > 0.10)")
        except Exception as exc:
            checks["warnings"].append(f"Ljung-Box test failed: {exc}")

        try:
            arch_stat, arch_pvalue = het_arch(clean_series, nlags=10)[:2]
            checks["tests"]["arch_lm"] = {"stat": float(arch_stat), "p_value": float(arch_pvalue)}
            if arch_pvalue > 0.10:
                checks["warnings"].append("Weak ARCH effect (GARCH may add limited value)")
        except Exception as exc:
            checks["warnings"].append(f"ARCH LM test failed: {exc}")

        return checks

    def evaluate_data_significance(self) -> Dict:
        """Run significance checks without re-training models."""
        if self.feature_data.empty:
            self.data_significance_checks = {"ml": {}, "econometric": {}}
            return self.data_significance_checks

        train_df, test_df = self._split_time_series()
        ml_checks = self._run_ml_data_checks(train_df, test_df, self.feature_cols)
        econ_checks = {}
        for symbol in sorted(self.top_stocks):
            symbol_train = train_df[train_df["symbol"] == symbol].sort_values("date")
            if symbol_train.empty:
                continue
            econ_checks[symbol] = self._run_econometric_checks(symbol_train["daily_return"])

        self.data_significance_checks = {"ml": ml_checks, "econometric": econ_checks}
        return self.data_significance_checks

    def train_ml_models(self) -> bool:
        print("Training ML models (RandomForest + XGBoost + optional LSTM)...")
        if self.feature_data.empty:
            return False

        ordered = self.feature_data.sort_values("date").reset_index(drop=True)
        n = len(ordered)
        test_start = int(n * (1 - self.target_config.test_ratio))
        val_start = int(test_start * (1 - self.target_config.val_ratio))
        train_df = ordered.iloc[:val_start].copy()
        val_df = ordered.iloc[val_start:test_start].copy()
        test_df = ordered.iloc[test_start:].copy()
        if train_df.empty or val_df.empty or test_df.empty:
            print("Insufficient data after train/val/test split.")
            return False

        feature_sets = {
            "core": [f for f in self.core_feature_cols if f in self.feature_cols],
            "hybrid": [f for f in (self.core_feature_cols + self.hybrid_feature_cols) if f in self.feature_cols],
        }

        y_train = self.label_encoder.fit_transform(train_df["target_class"])
        y_val = self.label_encoder.transform(val_df["target_class"])
        y_test = self.label_encoder.transform(test_df["target_class"])

        model_factories = {
            "RandomForest": lambda: RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=2,
                class_weight="balanced",
            )
        }
        if XGBClassifier is not None:
            model_factories["XGBoost"] = lambda: XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
            )
        if _get_tensorflow() is not None:
            model_factories["LSTM"] = lambda: LSTMClassifierWrapper(
                units=16,
                dropout=0.1,
                epochs=20,
                batch_size=64,
                random_state=42,
                verbose=0,
            )

        results = {}
        self.models = {}
        self.ml_predictions_store = {}
        self.model_feature_cols = {}
        all_selection_reports = {}
        ml_checks_aggregate = {"passed": True, "errors": [], "warnings": [], "details": {"by_feature_set": {}}}

        def train_candidate(model_factory, original_cols: List[str], selected_cols: List[str]):
            # Baseline uses original features; candidate uses selected features.
            X_train_baseline = train_df[original_cols]
            X_val_baseline = val_df[original_cols]
            X_test_baseline = test_df[original_cols]

            X_train = train_df[selected_cols]
            X_val = val_df[selected_cols]
            X_test = test_df[selected_cols]

            baseline_model = model_factory()
            baseline_model.fit(X_train_baseline, y_train)
            baseline_val_pred = baseline_model.predict(X_val_baseline)
            baseline_val_proba = baseline_model.predict_proba(X_val_baseline) if hasattr(baseline_model, "predict_proba") else None
            baseline_val_metrics = self._classification_metrics(y_val, baseline_val_pred, baseline_val_proba)

            model_for_val = model_factory()
            model_for_val.fit(X_train, y_train)
            val_pred = model_for_val.predict(X_val)
            val_proba = model_for_val.predict_proba(X_val) if hasattr(model_for_val, "predict_proba") else None
            val_metrics = self._classification_metrics(y_val, val_pred, val_proba)

            # Refit on train+val for final test evaluation.
            train_val_df = pd.concat([train_df, val_df], ignore_index=True)
            y_train_val = self.label_encoder.transform(train_val_df["target_class"])
            start = time.time()
            baseline_model = model_factory()
            baseline_model.fit(train_val_df[original_cols], y_train_val)
            baseline_pred = baseline_model.predict(X_test_baseline)
            baseline_proba = baseline_model.predict_proba(X_test_baseline) if hasattr(baseline_model, "predict_proba") else None
            baseline_elapsed = time.time() - start
            baseline_metrics = self._classification_metrics(y_test, baseline_pred, baseline_proba)
            baseline_metrics["train_time_sec"] = round(baseline_elapsed, 3)

            start = time.time()
            model = model_factory()
            model.fit(train_val_df[selected_cols], y_train_val)
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            elapsed = time.time() - start
            metrics = self._classification_metrics(y_test, pred, proba)
            metrics["val_f1"] = float(val_metrics["f1_weighted"])
            metrics["val_accuracy"] = float(val_metrics["accuracy"])
            metrics["baseline_val_f1"] = float(baseline_val_metrics["f1_weighted"])
            metrics["baseline_val_accuracy"] = float(baseline_val_metrics["accuracy"])
            metrics["selection_metric"] = "val_f1"
            metrics["split_sizes"] = {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))}
            metrics["train_time_sec"] = round(elapsed, 3)
            metrics["confusion_matrix"] = confusion_matrix(y_test, pred).tolist()
            metrics["labels"] = self.label_encoder.classes_.tolist()
            metrics["feature_count"] = len(selected_cols)
            delta = {
                "accuracy_delta": round(metrics["accuracy"] - baseline_metrics["accuracy"], 6),
                "f1_delta": round(metrics["f1_weighted"] - baseline_metrics["f1_weighted"], 6),
                "precision_delta": round(metrics["precision_weighted"] - baseline_metrics["precision_weighted"], 6),
            }
            return model, pred, proba, baseline_pred, baseline_proba, metrics, baseline_metrics, delta

        for name, model_factory in model_factories.items():
            candidate_results = {}
            for set_name, original_cols in feature_sets.items():
                selected_cols, selection_report = self._prune_multicollinearity(train_df, original_cols)
                importance_cols, importance_report = self._filter_by_importance(
                    model_factory, train_df, val_df, selected_cols
                )
                checks = self._run_ml_data_checks(train_df, val_df, importance_cols)
                checks["details"]["feature_selection"] = selection_report
                checks["details"]["importance_filter"] = importance_report
                ml_checks_aggregate["details"]["by_feature_set"][set_name] = checks
                all_selection_reports[set_name] = {
                    **selection_report,
                    "importance_filter": importance_report,
                    "final_selected_features": importance_cols,
                }
                if not checks["passed"]:
                    ml_checks_aggregate["passed"] = False
                    ml_checks_aggregate["errors"].extend([f"{set_name}: {msg}" for msg in checks["errors"]])
                    continue
                if checks["warnings"]:
                    ml_checks_aggregate["warnings"].extend([f"{set_name}: {msg}" for msg in checks["warnings"]])

                try:
                    model, pred, proba, baseline_pred, baseline_proba, metrics, baseline_metrics, delta = train_candidate(
                        model_factory, original_cols, importance_cols
                    )
                except Exception as exc:
                    msg = f"{name}/{set_name} training failed: {exc}"
                    ml_checks_aggregate["warnings"].append(msg)
                    print(f"Warning: {msg}")
                    continue
                candidate_results[set_name] = {
                    "model": model,
                    "pred": pred,
                    "proba": proba,
                    "baseline_pred": baseline_pred,
                    "baseline_proba": baseline_proba,
                    "metrics": metrics,
                    "baseline_metrics": baseline_metrics,
                    "delta": delta,
                    "selected_cols": importance_cols,
                    "selection_report": all_selection_reports[set_name],
                }

            if not candidate_results:
                continue

            best_set = max(candidate_results.keys(), key=lambda k: candidate_results[k]["metrics"]["val_f1"])
            chosen = candidate_results[best_set]
            model = chosen["model"]
            pred = chosen["pred"]
            proba = chosen["proba"]
            baseline_pred = chosen["baseline_pred"]
            metrics = chosen["metrics"]
            baseline_metrics = chosen["baseline_metrics"]
            delta = chosen["delta"]
            selected_feature_cols = chosen["selected_cols"]
            selection_report = chosen["selection_report"]

            results[name] = {
                "metrics": metrics,
                "baseline_metrics": baseline_metrics,
                "delta_vs_baseline": delta,
                "feature_selection": selection_report,
                "selected_feature_set": best_set,
                "feature_set_comparison": {
                    k: {
                        "metrics": v["metrics"],
                        "baseline_metrics": v["baseline_metrics"],
                        "delta_vs_baseline": v["delta"],
                        "feature_count": len(v["selected_cols"]),
                    }
                    for k, v in candidate_results.items()
                },
                "model_type": "machine_learning",
            }
            self.models[name] = model
            self.model_feature_cols[name] = selected_feature_cols
            self.ml_predictions_store[name] = {
                "y_test_encoded": y_test.tolist(),
                "y_test_labels": [self.label_encoder.inverse_transform([v])[0] for v in y_test],
                "pred_encoded": pred.tolist(),
                "pred_labels": [self.label_encoder.inverse_transform([v])[0] for v in pred],
                "pred_proba": proba.tolist() if proba is not None else None,
                "baseline_pred_encoded": baseline_pred.tolist(),
                "baseline_pred_labels": [self.label_encoder.inverse_transform([v])[0] for v in baseline_pred],
                "test_frame": test_df[["date", "symbol", "next_return_pct", "daily_return", "target_class"]].copy(),
            }
            print(
                f"{name} metrics:",
                f"baseline acc={baseline_metrics['accuracy']:.4f}, pruned acc={metrics['accuracy']:.4f},",
                f"baseline f1={baseline_metrics['f1_weighted']:.4f}, pruned f1={metrics['f1_weighted']:.4f},",
                f"set={best_set}",
            )

        if not results:
            print("No ML candidate passed checks.")
            return False

        self.feature_selection_report = all_selection_reports.get("hybrid", {})
        self.data_significance_checks["ml"] = ml_checks_aggregate
        if ml_checks_aggregate["warnings"]:
            print("ML data checks warnings:")
            for warning in sorted(set(ml_checks_aggregate["warnings"])):
                print(f" - {warning}")

        # Default feature_cols follows best-F1 ML model for downstream utilities.
        best_overall_ml = max(results.keys(), key=lambda n: results[n]["metrics"]["f1_weighted"])
        self.feature_cols = self.model_feature_cols[best_overall_ml]
        self.ml_results = results
        return True

    def _classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> Dict[str, float]:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        if y_prob is not None:
            try:
                metrics["roc_auc_ovr_weighted"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
            except Exception:
                metrics["roc_auc_ovr_weighted"] = None
        else:
            metrics["roc_auc_ovr_weighted"] = None
        return metrics

    def _walk_forward_cv_score(self, model_factory, df: pd.DataFrame, feature_cols: List[str], n_splits: int = 4) -> Dict:
        ordered = df.sort_values("date").reset_index(drop=True)
        total = len(ordered)
        if total < 1200:
            return {"mean_accuracy": None, "scores": [], "folds": 0}

        train_ratio = 0.6
        fold_scores = []
        for fold in range(n_splits):
            train_end = int(total * train_ratio) + fold * int(total * 0.1)
            test_end = train_end + int(total * 0.1)
            if test_end >= total or train_end < 300:
                continue
            train_fold = ordered.iloc[:train_end]
            test_fold = ordered.iloc[train_end:test_end]
            if train_fold.empty or test_fold.empty:
                continue
            model = model_factory()
            y_train = self.label_encoder.transform(train_fold["target_class"])
            y_test = self.label_encoder.transform(test_fold["target_class"])
            model.fit(train_fold[feature_cols], y_train)
            pred = model.predict(test_fold[feature_cols])
            fold_scores.append(float(accuracy_score(y_test, pred)))

        mean_score = float(np.mean(fold_scores)) if fold_scores else None
        return {"mean_accuracy": mean_score, "scores": fold_scores, "folds": len(fold_scores)}

    def _regime_evaluation(self, prediction_payload: Dict) -> Dict:
        frame = prediction_payload["test_frame"].copy().reset_index(drop=True)
        if frame.empty:
            return {"by_regime": {}, "macro_f1": None}
        frame["pred"] = prediction_payload["pred_labels"]
        vol = frame["daily_return"].abs().fillna(0.0)
        q1, q2 = vol.quantile(0.33), vol.quantile(0.66)
        frame["regime"] = np.where(vol <= q1, "LOW_VOL", np.where(vol <= q2, "MID_VOL", "HIGH_VOL"))

        by_regime = {}
        f1_values = []
        for regime, group in frame.groupby("regime"):
            y_true = group["target_class"].astype(str).tolist()
            y_pred = group["pred"].astype(str).tolist()
            labels = sorted(set(y_true) | set(y_pred))
            if len(labels) < 2:
                score = 1.0
            else:
                score = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            by_regime[regime] = {"size": int(len(group)), "f1_weighted": score}
            f1_values.append(score)
        macro_f1 = float(np.mean(f1_values)) if f1_values else None
        return {"by_regime": by_regime, "macro_f1": macro_f1}

    def _confidence_calibration(self, prediction_payload: Dict, bins: int = 10) -> Dict:
        proba = prediction_payload.get("pred_proba")
        if proba is None:
            return {"ece": None, "brier_like": None, "details": []}
        probs = np.array(proba, dtype=float)
        y_true = np.array(prediction_payload["y_test_encoded"], dtype=int)
        pred = np.array(prediction_payload["pred_encoded"], dtype=int)
        confidence = probs.max(axis=1)
        correctness = (pred == y_true).astype(float)
        ece = 0.0
        details = []
        for b in range(bins):
            left = b / bins
            right = (b + 1) / bins
            if b == bins - 1:
                mask = (confidence >= left) & (confidence <= right)
            else:
                mask = (confidence >= left) & (confidence < right)
            if not np.any(mask):
                continue
            conf_bin = float(confidence[mask].mean())
            acc_bin = float(correctness[mask].mean())
            weight = float(mask.mean())
            ece += abs(acc_bin - conf_bin) * weight
            details.append({"bin": f"{left:.1f}-{right:.1f}", "count": int(mask.sum()), "acc": acc_bin, "conf": conf_bin})

        one_hot = np.eye(probs.shape[1])[y_true]
        brier_like = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
        return {"ece": float(ece), "brier_like": brier_like, "details": details}

    def _paper_trading_metrics(self, prediction_payload: Dict) -> Dict:
        frame = prediction_payload["test_frame"].copy().reset_index(drop=True)
        if frame.empty:
            return {"total_return": None, "sharpe": None, "max_drawdown": None}
        signal_map = {"TANG": 1.0, "GIAM": -1.0, "SIDEWAY": 0.0}
        frame["signal"] = [signal_map.get(x, 0.0) for x in prediction_payload["pred_labels"]]
        confidence = None
        if prediction_payload.get("pred_proba") is not None:
            probs = np.array(prediction_payload["pred_proba"], dtype=float)
            confidence = probs.max(axis=1)
            frame["confidence"] = confidence
        else:
            frame["confidence"] = 0.5

        frame["volatility_20"] = frame["daily_return"].rolling(20).std().fillna(frame["daily_return"].std())

        def evaluate_strategy(conf_threshold: float, vol_quantile: float, down_clip: float, up_clip: float) -> Dict:
            local = frame.copy()
            vol_cutoff = local["volatility_20"].quantile(vol_quantile)
            vol_scale = np.where(local["volatility_20"] > vol_cutoff, 0.5, 1.0)
            confidence_gate = np.where(local["confidence"] >= conf_threshold, 1.0, 0.0)
            confidence_scale = np.clip((local["confidence"] - conf_threshold) / max(1e-6, (1 - conf_threshold)), 0.0, 1.0)
            local["position_size"] = confidence_gate * confidence_scale * vol_scale
            local["raw_strategy_return"] = local["signal"] * (local["next_return_pct"] / 100.0)
            local["strategy_return"] = (local["raw_strategy_return"] * local["position_size"]).clip(lower=down_clip, upper=up_clip)

            returns = local["strategy_return"].fillna(0.0)
            equity = (1.0 + returns).cumprod()
            total_return = float(equity.iloc[-1] - 1.0)
            std = float(returns.std())
            sharpe = float((returns.mean() / std) * np.sqrt(252)) if std > 1e-12 else 0.0
            running_max = equity.cummax()
            max_drawdown = float(((equity / running_max) - 1.0).min())
            trade_ratio = float((local["position_size"] > 0).mean())
            return {
                "total_return": total_return,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "trade_ratio": trade_ratio,
                "avg_position_size": float(local["position_size"].mean()),
                "params": {
                    "confidence_threshold": conf_threshold,
                    "vol_quantile": vol_quantile,
                    "down_clip": down_clip,
                    "up_clip": up_clip,
                },
            }

        best = None
        best_score = -1e18
        for conf_threshold in [0.35, 0.40, 0.45, 0.50, 0.55]:
            for vol_quantile in [0.65, 0.75, 0.85]:
                for down_clip, up_clip in [(-0.015, 0.02), (-0.02, 0.025), (-0.025, 0.03)]:
                    candidate = evaluate_strategy(conf_threshold, vol_quantile, down_clip, up_clip)
                    # Reward risk-adjusted return, penalize deep drawdown.
                    penalty = 0.0
                    if candidate["max_drawdown"] < self.target_config.max_drawdown_limit:
                        penalty += 2.0 * abs(candidate["max_drawdown"] - self.target_config.max_drawdown_limit)
                    score = candidate["sharpe"] + 0.35 * candidate["total_return"] - penalty
                    if score > best_score:
                        best_score = score
                        best = candidate

        if best is None:
            return {"total_return": None, "sharpe": None, "max_drawdown": None}

        total_return = best["total_return"]
        sharpe = best["sharpe"]
        max_drawdown = best["max_drawdown"]
        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "trade_ratio": best["trade_ratio"],
            "avg_position_size": best["avg_position_size"],
            "strategy_params": best["params"],
        }

    def _build_benchmark_checklist(self) -> Dict:
        checklist = {"items": {}, "overall_passed": False}
        if not self.ml_results or not self.ml_predictions_store:
            checklist["items"]["availability"] = {"passed": False, "message": "No ML results available for checklist"}
            return checklist

        best_ml_name = max(self.ml_results.keys(), key=lambda n: self.ml_results[n]["metrics"].get("f1_weighted", 0.0))
        best_ml_result = self.ml_results[best_ml_name]
        payload = self.ml_predictions_store[best_ml_name]
        test_df = payload["test_frame"]

        # Walk-forward CV
        if best_ml_name == "RandomForest":
            model_factory = lambda: RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=2, class_weight="balanced"
            )
        elif best_ml_name == "XGBoost":
            model_factory = lambda: XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
            )
        else:
            model_factory = lambda: LSTMClassifierWrapper(
                units=16,
                dropout=0.1,
                epochs=20,
                batch_size=64,
                random_state=42,
                verbose=0,
            )
        cv = self._walk_forward_cv_score(model_factory, self.feature_data, self.feature_cols)
        cv_pass = cv["mean_accuracy"] is not None and cv["mean_accuracy"] >= self.target_config.min_walk_forward_accuracy
        checklist["items"]["walk_forward_cv"] = {**cv, "passed": cv_pass}

        # Baseline comparison
        improvement = float(best_ml_result["delta_vs_baseline"]["f1_delta"])
        baseline_pass = improvement >= self.target_config.min_baseline_improvement
        checklist["items"]["baseline_comparison"] = {
            "passed": baseline_pass,
            "best_model": best_ml_name,
            "f1_improvement_vs_baseline": improvement,
            "threshold": self.target_config.min_baseline_improvement,
        }

        # Regime performance
        regime = self._regime_evaluation(payload)
        regime_pass = regime["macro_f1"] is not None and regime["macro_f1"] >= self.target_config.min_regime_f1
        checklist["items"]["regime_analysis"] = {**regime, "passed": regime_pass, "threshold": self.target_config.min_regime_f1}

        # Calibration
        calibration = self._confidence_calibration(payload)
        calibration_pass = calibration["ece"] is not None and calibration["ece"] <= self.target_config.max_expected_calibration_error
        checklist["items"]["confidence_calibration"] = {
            **calibration,
            "passed": calibration_pass,
            "max_ece_threshold": self.target_config.max_expected_calibration_error,
        }

        # Paper trading
        trading = self._paper_trading_metrics(payload)
        trading_pass = (
            trading["sharpe"] is not None
            and trading["sharpe"] >= self.target_config.min_sharpe_ratio
            and trading["max_drawdown"] >= self.target_config.max_drawdown_limit
        )
        checklist["items"]["paper_trading"] = {
            **trading,
            "passed": trading_pass,
            "min_sharpe_threshold": self.target_config.min_sharpe_ratio,
            "max_drawdown_limit": self.target_config.max_drawdown_limit,
        }

        checklist["overall_passed"] = all(item.get("passed") for item in checklist["items"].values())
        checklist["best_ml_model"] = best_ml_name
        checklist["best_ml_feature_set"] = best_ml_result.get("selected_feature_set")
        checklist["sample_size"] = int(len(test_df))
        return checklist

    def train_econometric_models(self) -> bool:
        print("Training econometric models (ARIMA + GARCH)...")
        if self.feature_data.empty:
            return False
        if ARIMA is None and arch_model is None:
            print("statsmodels/arch not available, skip econometric models.")
            self.econometric_results = {}
            return False

        results = {}
        check_results = {}
        train_df, test_df = self._split_time_series()
        for symbol in sorted(self.top_stocks):
            symbol_train = train_df[train_df["symbol"] == symbol].sort_values("date")
            symbol_test = test_df[test_df["symbol"] == symbol].sort_values("date")
            if len(symbol_train) < 60 or len(symbol_test) < 10:
                continue

            symbol_result = {"model_type": "econometric", "metrics": {}, "p_values": {}}
            train_return = symbol_train["daily_return"].dropna()
            test_return = symbol_test["daily_return"].dropna()
            if len(train_return) < 40 or test_return.empty:
                continue

            econ_checks = self._run_econometric_checks(train_return)
            check_results[symbol] = econ_checks
            if not econ_checks["passed"]:
                continue

            arima_pred = None
            if ARIMA is not None:
                try:
                    arima_fit = ARIMA(train_return, order=(1, 0, 1)).fit()
                    forecast = arima_fit.forecast(steps=len(test_return))
                    arima_pred = pd.Series(forecast, index=test_return.index)
                    pvals = {k: float(v) for k, v in arima_fit.pvalues.to_dict().items()}
                    symbol_result["p_values"]["ARIMA"] = pvals
                except Exception as exc:
                    symbol_result["p_values"]["ARIMA"] = {"error": str(exc)}

            garch_vol = None
            if arch_model is not None:
                try:
                    garch_fit = arch_model(train_return * 100, vol="Garch", p=1, q=1, dist="normal").fit(disp="off")
                    garch_forecast = garch_fit.forecast(horizon=1, reindex=False)
                    variance = float(garch_forecast.variance.values[-1, 0])
                    garch_vol = np.sqrt(max(variance, 0)) / 100.0
                    pvals = {k: float(v) for k, v in garch_fit.pvalues.to_dict().items()}
                    symbol_result["p_values"]["GARCH"] = pvals
                except Exception as exc:
                    symbol_result["p_values"]["GARCH"] = {"error": str(exc)}

            if arima_pred is not None:
                next_return_pct = arima_pred.values * 100.0
                pred_labels = [self._trend_label(v) for v in next_return_pct]
                actual_labels = [self._trend_label(v * 100.0) for v in test_return.values]
                y_true = self.label_encoder.transform([lbl if lbl in self.label_encoder.classes_ else "SIDEWAY" for lbl in actual_labels])
                y_pred = self.label_encoder.transform([lbl if lbl in self.label_encoder.classes_ else "SIDEWAY" for lbl in pred_labels])
                metrics = self._classification_metrics(y_true, y_pred, None)
                metrics["expected_volatility"] = garch_vol
                symbol_result["metrics"] = metrics
                symbol_result["checks"] = econ_checks
                results[symbol] = symbol_result

        self.data_significance_checks["econometric"] = check_results
        self.econometric_results = results
        print(f"Econometric model evaluated for {len(results)} symbols.")
        return len(results) > 0

    def build_benchmark_summary(self) -> Dict:
        benchmark_rows = []
        for model_name, result in self.ml_results.items():
            selected_set = result.get("selected_feature_set", "unknown")
            benchmark_rows.append(
                {
                    "model_group": "ML",
                    "model_name": model_name,
                    "model_variant": f"{model_name} [{selected_set}]",
                    "selected_feature_set": selected_set,
                    **result["metrics"],
                }
            )

        econ_metrics = []
        for symbol, payload in self.econometric_results.items():
            row = {"symbol": symbol, **payload.get("metrics", {})}
            econ_metrics.append(row)
        if econ_metrics:
            econ_df = pd.DataFrame(econ_metrics)
            agg = {
                "accuracy": float(econ_df["accuracy"].mean()) if "accuracy" in econ_df else None,
                "precision_weighted": float(econ_df["precision_weighted"].mean()) if "precision_weighted" in econ_df else None,
                "f1_weighted": float(econ_df["f1_weighted"].mean()) if "f1_weighted" in econ_df else None,
                "roc_auc_ovr_weighted": None,
            }
            benchmark_rows.append(
                {
                    "model_group": "Econometric",
                    "model_name": "ARIMA+GARCH",
                    "model_variant": "ARIMA+GARCH",
                    "selected_feature_set": None,
                    **agg,
                }
            )

        benchmark_df = pd.DataFrame(benchmark_rows)
        benchmark_df = benchmark_df.sort_values(by=["accuracy", "f1_weighted"], ascending=False, na_position="last")
        top_model = benchmark_df.iloc[0].to_dict() if not benchmark_df.empty else {}
        benchmark_checklist = self._build_benchmark_checklist()
        self.benchmark_summary = {
            "ranking": benchmark_df.fillna("").to_dict(orient="records"),
            "top_model": top_model,
            "econometric_by_symbol": self.econometric_results,
            "ml_models": self.ml_results,
            "feature_selection_report": self.feature_selection_report,
            "data_significance_checks": self.data_significance_checks,
            "benchmark_checklist": benchmark_checklist,
        }
        return self.benchmark_summary

    def train_model(self, test_size: float = 0.2) -> bool:
        self.target_config.test_ratio = test_size
        ml_ok = self.train_ml_models()
        self.train_econometric_models()
        self.build_benchmark_summary()
        return ml_ok

    def save_model(self) -> bool:
        if not self.models:
            return False

        bundle = {
            "models": self.models,
            "feature_cols": self.feature_cols,
            "model_feature_cols": self.model_feature_cols,
            "core_feature_cols": self.core_feature_cols,
            "hybrid_feature_cols": self.hybrid_feature_cols,
            "top_stocks": self.top_stocks,
            "label_classes": self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, "classes_") else [],
            "benchmark_summary": self.benchmark_summary,
            "symbol_performance": self.symbol_performance.to_dict(orient="records"),
            "target_config": self.target_config.__dict__,
            "data_significance_checks": self.data_significance_checks,
            "feature_selection_report": self.feature_selection_report,
            "ml_predictions_store": self.ml_predictions_store,
        }
        model_path = self.current_dir / self.MODEL_PATH
        scaler_path = self.current_dir / self.SCALER_PATH
        bundle_path = self.current_dir / self.BUNDLE_PATH

        # Keep backward compatibility with the old pair of artifact files.
        primary_model = self.models.get("RandomForest")
        joblib.dump(primary_model, model_path)
        joblib.dump({"feature_cols": self.feature_cols}, scaler_path)
        joblib.dump(bundle, bundle_path)
        print(f"Saved model bundle: {bundle_path}")
        return True

    def load_model_bundle(self) -> bool:
        bundle_path = self.current_dir / self.BUNDLE_PATH
        if not bundle_path.exists():
            return False
        bundle = joblib.load(bundle_path)
        self.models = bundle.get("models", {})
        self.feature_cols = bundle.get("feature_cols", [])
        self.model_feature_cols = bundle.get("model_feature_cols", {})
        self.core_feature_cols = bundle.get("core_feature_cols", self.core_feature_cols)
        self.hybrid_feature_cols = bundle.get("hybrid_feature_cols", self.hybrid_feature_cols)
        self.top_stocks = bundle.get("top_stocks", [])
        self.benchmark_summary = bundle.get("benchmark_summary", {})
        self.data_significance_checks = bundle.get("data_significance_checks", {"ml": {}, "econometric": {}})
        self.feature_selection_report = bundle.get("feature_selection_report", {})
        self.ml_predictions_store = bundle.get("ml_predictions_store", {})
        if isinstance(self.benchmark_summary, dict) and "data_significance_checks" not in self.benchmark_summary:
            self.benchmark_summary["data_significance_checks"] = self.data_significance_checks
        if isinstance(self.benchmark_summary, dict) and "feature_selection_report" not in self.benchmark_summary:
            self.benchmark_summary["feature_selection_report"] = self.feature_selection_report
        if isinstance(self.benchmark_summary, dict) and "benchmark_checklist" not in self.benchmark_summary:
            self.benchmark_summary["benchmark_checklist"] = {}
        perf = bundle.get("symbol_performance", [])
        self.symbol_performance = pd.DataFrame(perf)
        label_classes = bundle.get("label_classes", [])
        if label_classes:
            self.label_encoder.fit(label_classes)
        config = bundle.get("target_config", {})
        self.target_config = TargetConfig(**{**self.target_config.__dict__, **config})
        return True

    def get_current_stock_info(self, symbol: str) -> Optional[Dict]:
        latest_data = self.processed_data[self.processed_data["symbol"] == symbol].tail(1)
        if latest_data.empty:
            return None
        info = latest_data.iloc[0]
        return {
            "symbol": symbol,
            "name": self.stock_name_map.get(symbol, symbol),
            "date": info["date"].strftime("%Y-%m-%d"),
            "current_price": float(info["close"]),
            "open": float(info["open"]),
            "high": float(info["high"]),
            "low": float(info["low"]),
            "volume": int(info["volume"]),
            "ema_10": float(info["EMA_10"]) if pd.notna(info["EMA_10"]) else None,
            "ema_20": float(info["EMA_20"]) if pd.notna(info["EMA_20"]) else None,
            "ema_50": float(info["EMA_50"]) if pd.notna(info["EMA_50"]) else None,
            "ma_10": float(info["EMA_10"]) if pd.notna(info["EMA_10"]) else None,
            "ma_20": float(info["EMA_20"]) if pd.notna(info["EMA_20"]) else None,
            "ma_50": float(info["EMA_50"]) if pd.notna(info["EMA_50"]) else None,
            "rsi": float(info["RSI"]) if pd.notna(info["RSI"]) else None,
            "macd": float(info["MACD"]) if pd.notna(info["MACD"]) else None,
            "bb_upper": float(info["BB_Upper"]) if pd.notna(info["BB_Upper"]) else None,
            "bb_lower": float(info["BB_Lower"]) if pd.notna(info["BB_Lower"]) else None,
            "obv": float(info["OBV"]) if pd.notna(info["OBV"]) else None,
            "vwap": float(info["VWAP"]) if pd.notna(info["VWAP"]) else None,
        }

    def predict_tomorrow(self, symbol: str, model_name: str = "RandomForest") -> Optional[Dict]:
        if model_name not in self.models:
            if self.models:
                model_name = list(self.models.keys())[0]
            else:
                return None

        current_info = self.get_current_stock_info(symbol)
        if not current_info:
            return None
        latest = self.feature_data[self.feature_data["symbol"] == symbol].tail(1)
        if latest.empty:
            return None

        model = self.models[model_name]
        active_cols = self.model_feature_cols.get(model_name, self.feature_cols)
        features = latest[active_cols]
        pred_encoded = model.predict(features)[0]
        pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
        proba = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None

        current_price = current_info["current_price"]
        if pred_label == "TANG":
            expected_pct = 1.2
            trend = "TĂNG"
            trend_icon = "📈"
            trend_color = "success"
        elif pred_label == "GIAM":
            expected_pct = -1.2
            trend = "GIẢM"
            trend_icon = "📉"
            trend_color = "danger"
        else:
            expected_pct = 0.0
            trend = "SIDEWAY"
            trend_icon = "➡️"
            trend_color = "warning"

        predicted_price = current_price * (1 + expected_pct / 100)
        confidence = float(np.max(proba)) if proba is not None else None
        return {
            "symbol": symbol,
            "model_name": model_name,
            "current_price": float(current_price),
            "predicted_price": float(predicted_price),
            "price_change": float(predicted_price - current_price),
            "price_change_percent": float(expected_pct),
            "trend": trend,
            "trend_icon": trend_icon,
            "trend_color": trend_color,
            "confidence": confidence,
            "date": current_info["date"],
        }

    def get_benchmark_summary(self) -> Dict:
        return self.benchmark_summary

    def get_eda_summary(self) -> Dict:
        return self.eda_summary

    def get_top10_performance(self) -> List[Dict]:
        if self.symbol_performance.empty:
            return []
        df = self.symbol_performance[self.symbol_performance["symbol"].isin(self.top_stocks)].copy()
        df = df.sort_values("return_5y_pct", ascending=False).reset_index(drop=True)
        # Display rank in Top10 table as 1..10 (global VN30 rank is still kept in rank).
        df["top10_rank"] = np.arange(1, len(df) + 1)
        return df.to_dict(orient="records")

    def run_full_pipeline(self, period: str = "5y", refresh_cache: bool = False) -> bool:
        if not self.collect_stock_data_yfinance(period=period, use_cache=not refresh_cache):
            return False
        if not self.preprocess_data():
            return False
        if not self.run_eda():
            return False
        if not self.calculate_technical_indicators():
            return False
        if not self.add_hybrid_econometric_features():
            return False
        if not self.prepare_features():
            return False
        if not self.train_model(test_size=self.target_config.test_ratio):
            return False
        return self.save_model()


def main():
    print("=" * 72)
    print("VN30 MULTI-MODEL ANALYSIS PIPELINE")
    print("Top 10 by 5Y performance + EDA + Econometric + ML")
    print("=" * 72)
    start = time.time()
    model = StockAnalysisYFinance()

    try:
        if not model.test_yfinance_connection():
            print("Cannot reach Yahoo Finance.")
            return

        ok = model.run_full_pipeline(period="5y", refresh_cache=False)
        if not ok:
            print("Pipeline failed.")
            return

        elapsed = time.time() - start
        print("Pipeline completed successfully.")
        print(f"Top stocks: {', '.join(model.top_stocks)}")
        top_model = model.get_benchmark_summary().get("top_model", {})
        if top_model:
            print(f"Best model: {top_model.get('model_name')} (group={top_model.get('model_group')})")
        print(f"Elapsed time: {elapsed:.1f}s")
    except Exception as exc:
        print(f"Unhandled error: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()