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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None

try:
    from arch import arch_model
except ImportError:
    arch_model = None

# Scale daily returns for arch numerical stability (mean/variance forecasts are mapped back).
_ECON_RETURN_SCALE = 100.0


def _joint_ar1_garch_fit_forecast(train_scaled: np.ndarray, horizon: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
    """
    Joint AR(1) mean with GARCH(1,1) innovations on scaled returns (train_scaled ≈ r * _ECON_RETURN_SCALE).
    Returns multi-step mean and conditional volatility in original return units.
    """
    if arch_model is None:
        return None, None, None
    y = np.asarray(train_scaled, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 60:
        return None, None, None
    try:
        am = arch_model(y, mean="ARX", lags=[1], vol="Garch", p=1, q=1, dist="normal")
        res = am.fit(disp="off")
        fc = res.forecast(horizon=int(horizon), reindex=False)
        mu_s = np.asarray(fc.mean.values[0], dtype=float)
        var_s = np.asarray(fc.variance.values[0], dtype=float)
        mu = mu_s / _ECON_RETURN_SCALE
        vol = np.sqrt(np.maximum(var_s, 0.0)) / _ECON_RETURN_SCALE
        pvals = {str(k): float(v) for k, v in res.pvalues.items()}
        return mu, vol, {"pvalues": pvals}
    except Exception as exc:
        return None, None, {"error": str(exc)}


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
    use_seasonal_features: bool = True
    use_regime_features: bool = True
    use_market_context_features: bool = True
    # Rolling-origin trend eval (train only rows with date < dt). None = all test calendar days; <= 0 skips.
    rolling_forecast_max_dates: Optional[int] = None
    rolling_forecast_include_econ_step: bool = True  # parallel joint AR(1)-GARCH class forecast vs labels
    rolling_forecast_include_garch_vol: bool = False  # if True, collects σ each step (slower)
    # AFML-style meta-labeling: rule primary (EMA cross) + triple-barrier outcome; ML predicts P(signal useful).
    use_meta_labeling: bool = True
    meta_barrier_horizon: int = 10
    meta_pt_sl: Tuple[float, float] = (1.5, 1.5)
    meta_prob_trade_threshold: float = 0.5


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
        self.regime_feature_cols: List[str] = []
        self.hybrid_feature_cols: List[str] = []
        self.seasonal_feature_cols: List[str] = []
        self.market_context_feature_cols: List[str] = []
        self.model_feature_cols: Dict[str, List[str]] = {}
        self.label_encoder = LabelEncoder()
        self.meta_label_encoder = LabelEncoder()
        self.models: Dict[str, object] = {}
        self.ml_predictions_store: Dict = {}
        self.top_stocks: List[str] = []
        self._rolling_fusion_weight_ml: Optional[float] = None
        self._use_meta_labeling: bool = False
        self.meta_aux_cols: List[str] = []
        self.meta_feature_cols_full: List[str] = []

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
        Leakage-safe columns from joint AR(1)-GARCH(1,1) only (arch). Periodic refit on expanding history.
        Rows stay NaN where the joint model did not produce a forecast (no ARIMA-only / GARCH-only fallback).
        """
        if self.processed_data.empty:
            return False
        if arch_model is None:
            print("arch not installed; skipping hybrid econometric features.")
            return False

        print("Generating hybrid econometric features (joint AR(1)-GARCH expanding window only)...")
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

            joint_fit_cached = None

            for i in range(min_obs, n):
                # Refit periodically to control compute.
                if (i == min_obs) or ((i - min_obs) % refit_interval == 0):
                    hist = pd.Series(returns[:i]).replace([np.inf, -np.inf], np.nan).dropna()
                    if len(hist) < min_obs:
                        continue

                    joint_fit_cached = None
                    try:
                        ys = hist.values.astype(float) * _ECON_RETURN_SCALE
                        am = arch_model(ys, mean="ARX", lags=[1], vol="Garch", p=1, q=1, dist="normal")
                        joint_fit_cached = am.fit(disp="off")
                    except Exception:
                        joint_fit_cached = None

                if joint_fit_cached is not None:
                    try:
                        fc = joint_fit_cached.forecast(horizon=1, reindex=False)
                        mu_s = float(fc.mean.values[0, 0])
                        var_s = float(fc.variance.values[0, 0])
                        arima_pred[i] = mu_s / _ECON_RETURN_SCALE
                        garch_vol_pred[i] = np.sqrt(max(var_s, 0.0)) / _ECON_RETURN_SCALE
                        arima_resid_std[i] = garch_vol_pred[i]
                    except Exception:
                        pass

            g["ARIMA_FC_RET"] = arima_pred
            g["ARIMA_RESID_STD"] = arima_resid_std
            g["GARCH_FC_VOL"] = garch_vol_pred
            g["HYBRID_VALID_FLAG"] = (
                pd.Series(arima_pred).notna() & pd.Series(garch_vol_pred).notna()
            ).astype(int)

            vol_s = g["GARCH_FC_VOL"]
            if vol_s.notna().sum() >= 30:
                q_low = vol_s.quantile(0.33)
                q_high = vol_s.quantile(0.66)
                g["HYBRID_REGIME"] = np.where(
                    vol_s.isna(),
                    np.nan,
                    np.where(vol_s <= q_low, 0, np.where(vol_s <= q_high, 1, 2)),
                )
                g["HYBRID_REGIME"] = pd.Series(g["HYBRID_REGIME"]).fillna(1).astype(int)
            else:
                g["HYBRID_REGIME"] = 1
            frames.append(g)

        self.processed_data = pd.concat(frames).sort_values(["symbol", "date"]).reset_index(drop=True)
        return True

    @staticmethod
    def _triple_barrier_on_group(g: pd.DataFrame, t1: int, pt: float, sl: float) -> pd.DataFrame:
        """Per-symbol path-dependent triple barrier (upper / lower / vertical) in return space."""
        g = g.copy()
        n = len(g)
        hits = np.full(n, np.nan)
        rets = np.full(n, np.nan)
        close = g["close"].to_numpy(dtype=float)
        vol = g["VOL_20"].to_numpy(dtype=float)
        for i in range(0, max(0, n - t1 - 1)):
            if np.isnan(vol[i]) or vol[i] <= 0:
                continue
            path = close[i : i + t1 + 1] / close[i] - 1.0
            ub = float(vol[i]) * pt
            lb = -float(vol[i]) * sl
            hit_up_j = None
            hit_lo_j = None
            for j in range(1, len(path)):
                if path[j] >= ub:
                    hit_up_j = j
                    break
            for j in range(1, len(path)):
                if path[j] <= lb:
                    hit_lo_j = j
                    break
            if hit_up_j is not None and (hit_lo_j is None or hit_up_j < hit_lo_j):
                hits[i] = 1.0
                rets[i] = float(path[hit_up_j])
            elif hit_lo_j is not None and (hit_up_j is None or hit_lo_j < hit_up_j):
                hits[i] = -1.0
                rets[i] = float(path[hit_lo_j])
            else:
                hits[i] = 0.0
                rets[i] = float(path[-1])
        g["barrier_hit"] = hits
        g["barrier_path_ret"] = rets
        return g

    def _apply_meta_labeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Primary EMA10×EMA50 side + triple-barrier meta_label (1 = primary would have been rewarded)."""
        cfg = self.target_config
        df = df.copy()
        df["primary_side"] = np.where(df["EMA_10"] > df["EMA_50"], 1.0, -1.0)
        pieces: List[pd.DataFrame] = []
        for _, g in df.groupby("symbol", sort=False):
            pieces.append(
                self._triple_barrier_on_group(
                    g,
                    int(cfg.meta_barrier_horizon),
                    float(cfg.meta_pt_sl[0]),
                    float(cfg.meta_pt_sl[1]),
                )
            )
        out = pd.concat(pieces, axis=0).sort_values(["symbol", "date"]).reset_index(drop=True)
        ps = out["primary_side"].to_numpy(dtype=float)
        hit = out["barrier_hit"].to_numpy(dtype=float)
        ret = out["barrier_path_ret"].to_numpy(dtype=float)
        ml = np.full(len(out), np.nan)
        for i in range(len(out)):
            if np.isnan(hit[i]):
                continue
            side = ps[i]
            h = hit[i]
            r = ret[i]
            if np.isnan(r):
                continue
            if side > 0:
                ml[i] = 1.0 if (h == 1 or (h == 0 and r > 0)) else 0.0
            else:
                ml[i] = 1.0 if (h == -1 or (h == 0 and r < 0)) else 0.0
        out["meta_label"] = ml
        return out

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
        # Regime/trend features focused on short-term turbulence and momentum shift.
        grouped_ret = df.groupby("symbol")["daily_return"]
        df["RET_1D_LAG1"] = grouped_ret.shift(1)
        df["RET_1D_LAG2"] = grouped_ret.shift(2)
        df["RET_5D"] = grouped_ret.transform(lambda s: s.rolling(5).sum())
        df["RET_10D"] = grouped_ret.transform(lambda s: s.rolling(10).sum())
        df["VOL_5"] = grouped_ret.transform(lambda s: s.rolling(5).std())
        df["VOL_10"] = grouped_ret.transform(lambda s: s.rolling(10).std())
        df["VOL_20"] = grouped_ret.transform(lambda s: s.rolling(20).std())
        df["VOL_RATIO_5_20"] = df["VOL_5"] / df["VOL_20"].replace(0, np.nan)
        vol_mean_20 = df.groupby("symbol")["VOL_20"].transform(lambda s: s.rolling(20).mean())
        vol_std_20 = df.groupby("symbol")["VOL_20"].transform(lambda s: s.rolling(20).std())
        df["VOL_ZSCORE_20"] = (df["VOL_20"] - vol_mean_20) / vol_std_20.replace(0, np.nan)
        df["CLOSE_TO_EMA20"] = (df["close"] / df["EMA_20"].replace(0, np.nan)) - 1.0
        df["CLOSE_TO_EMA50"] = (df["close"] / df["EMA_50"].replace(0, np.nan)) - 1.0

        # Market-context features from cross-sectional VN30 universe each trading day.
        market_daily = (
            df.groupby("date")
            .agg(
                MKT_RET_1D=("daily_return", "mean"),
                BREADTH_UP_RATIO=("daily_return", lambda s: float((s > 0).mean())),
                BREADTH_DOWN_RATIO=("daily_return", lambda s: float((s < 0).mean())),
                MKT_TOTAL_VOLUME=("volume", "sum"),
            )
            .sort_index()
        )
        market_daily["MKT_RET_5D"] = market_daily["MKT_RET_1D"].rolling(5).sum()
        market_daily["MKT_RET_10D"] = market_daily["MKT_RET_1D"].rolling(10).sum()
        market_daily["MKT_VOL_20"] = market_daily["MKT_RET_1D"].rolling(20).std()
        market_daily["MKT_VOLUME_CHANGE"] = market_daily["MKT_TOTAL_VOLUME"].pct_change()
        market_daily["BREADTH_SPREAD"] = market_daily["BREADTH_UP_RATIO"] - market_daily["BREADTH_DOWN_RATIO"]
        df = df.merge(market_daily, left_on="date", right_index=True, how="left")
        df["REL_STRENGTH_1D"] = df["daily_return"] - df["MKT_RET_1D"]
        df["REL_STRENGTH_5D"] = df["RET_5D"] - df["MKT_RET_5D"]
        df["VOL_SHARE_OF_MARKET"] = df["volume"] / df["MKT_TOTAL_VOLUME"].replace(0, np.nan)

        # Seasonal/calendar features to better capture recurring time effects.
        df["DOW"] = df["date"].dt.dayofweek
        df["DOM"] = df["date"].dt.day
        df["MONTH"] = df["date"].dt.month
        df["QUARTER"] = df["date"].dt.quarter
        df["IS_MONTH_END"] = df["date"].dt.is_month_end.astype(int)
        df["IS_QUARTER_END"] = df["date"].dt.is_quarter_end.astype(int)
        df["DOW_SIN"] = np.sin(2 * np.pi * df["DOW"] / 7.0)
        df["DOW_COS"] = np.cos(2 * np.pi * df["DOW"] / 7.0)
        df["MONTH_SIN"] = np.sin(2 * np.pi * (df["MONTH"] - 1) / 12.0)
        df["MONTH_COS"] = np.cos(2 * np.pi * (df["MONTH"] - 1) / 12.0)
        df["DOM_SIN"] = np.sin(2 * np.pi * (df["DOM"] - 1) / 31.0)
        df["DOM_COS"] = np.cos(2 * np.pi * (df["DOM"] - 1) / 31.0)
        # Trading-calendar seasonality features (month/quarter edge effects).
        month_key = [df["symbol"], df["date"].dt.to_period("M")]
        quarter_key = [df["symbol"], df["date"].dt.to_period("Q")]
        df["TDAY_IN_MONTH"] = df.groupby(month_key).cumcount() + 1
        df["TDAYS_IN_MONTH"] = df.groupby(month_key)["date"].transform("count")
        df["TDAYS_TO_MONTH_END"] = df["TDAYS_IN_MONTH"] - df["TDAY_IN_MONTH"]
        df["TDAY_IN_QUARTER"] = df.groupby(quarter_key).cumcount() + 1
        df["TDAYS_IN_QUARTER"] = df.groupby(quarter_key)["date"].transform("count")
        df["TDAYS_TO_QUARTER_END"] = df["TDAYS_IN_QUARTER"] - df["TDAY_IN_QUARTER"]
        df["IS_MONTH_START_3D"] = (df["TDAY_IN_MONTH"] <= 3).astype(int)
        df["IS_MONTH_END_3D"] = (df["TDAYS_TO_MONTH_END"] <= 2).astype(int)
        df["IS_QUARTER_START_5D"] = (df["TDAY_IN_QUARTER"] <= 5).astype(int)
        df["IS_QUARTER_END_5D"] = (df["TDAYS_TO_QUARTER_END"] <= 4).astype(int)
        df["MONTH_PROGRESS"] = df["TDAY_IN_MONTH"] / df["TDAYS_IN_MONTH"].replace(0, np.nan)
        df["QUARTER_PROGRESS"] = df["TDAY_IN_QUARTER"] / df["TDAYS_IN_QUARTER"].replace(0, np.nan)

        if self.target_config.use_meta_labeling:
            df = self._apply_meta_labeling(df)
            self.meta_aux_cols = ["primary_side"]
        else:
            self.meta_aux_cols = []

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
        self.regime_feature_cols = [
            "RET_1D_LAG1",
            "RET_1D_LAG2",
            "RET_5D",
            "RET_10D",
            "VOL_5",
            "VOL_10",
            "VOL_20",
            "VOL_RATIO_5_20",
            "VOL_ZSCORE_20",
            "CLOSE_TO_EMA20",
            "CLOSE_TO_EMA50",
        ]
        self.hybrid_feature_cols = [
            "ARIMA_FC_RET",
            "ARIMA_RESID_STD",
            "GARCH_FC_VOL",
            "HYBRID_REGIME",
            "HYBRID_VALID_FLAG",
        ]
        self.seasonal_feature_cols = [
            "DOW",
            "DOM",
            "MONTH",
            "QUARTER",
            "IS_MONTH_END",
            "IS_QUARTER_END",
            "DOW_SIN",
            "DOW_COS",
            "MONTH_SIN",
            "MONTH_COS",
            "DOM_SIN",
            "DOM_COS",
            "TDAY_IN_MONTH",
            "TDAYS_IN_MONTH",
            "TDAYS_TO_MONTH_END",
            "TDAY_IN_QUARTER",
            "TDAYS_IN_QUARTER",
            "TDAYS_TO_QUARTER_END",
            "IS_MONTH_START_3D",
            "IS_MONTH_END_3D",
            "IS_QUARTER_START_5D",
            "IS_QUARTER_END_5D",
            "MONTH_PROGRESS",
            "QUARTER_PROGRESS",
        ]
        self.market_context_feature_cols = [
            "MKT_RET_1D",
            "MKT_RET_5D",
            "MKT_RET_10D",
            "MKT_VOL_20",
            "MKT_TOTAL_VOLUME",
            "MKT_VOLUME_CHANGE",
            "BREADTH_UP_RATIO",
            "BREADTH_DOWN_RATIO",
            "BREADTH_SPREAD",
            "REL_STRENGTH_1D",
            "REL_STRENGTH_5D",
            "VOL_SHARE_OF_MARKET",
        ]
        regime_cols = self.regime_feature_cols if self.target_config.use_regime_features else []
        seasonal_cols = self.seasonal_feature_cols if self.target_config.use_seasonal_features else []
        market_cols = self.market_context_feature_cols if self.target_config.use_market_context_features else []
        # ML inputs only: hybrid_feature_cols (ARIMA/GARCH forecasts on processed_data) are never appended.
        self.feature_cols = self.core_feature_cols + regime_cols + seasonal_cols + market_cols
        self.meta_feature_cols_full = self.feature_cols + (self.meta_aux_cols if self.target_config.use_meta_labeling else [])

        if self.target_config.use_meta_labeling:
            req = self.meta_feature_cols_full + ["target_class", "meta_label", "next_close", "next_return_pct"]
            filtered = df.dropna(subset=req).copy()
            filtered = filtered[filtered["target_class"].isin(["TANG", "SIDEWAY", "GIAM"])]
            filtered["meta_label"] = filtered["meta_label"].astype(int)
        else:
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

    def _run_ml_data_checks(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        y_col: str = "target_class",
        binary: bool = False,
    ) -> Dict:
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

        class_ratio = train_df[y_col].value_counts(normalize=True).to_dict()
        class_count = train_df[y_col].value_counts().to_dict()
        checks["details"]["class_ratio"] = class_ratio
        checks["details"]["class_count"] = class_count
        min_ratio = 0.05 if not binary else 0.08
        min_per_class = 40 if not binary else 80
        for cls, ratio in class_ratio.items():
            if ratio < min_ratio:
                checks["warnings"].append(f"Class imbalance detected: {cls} ratio={ratio:.3f}")
        for cls, cnt in class_count.items():
            if cnt < min_per_class:
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

    def _prune_multicollinearity(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        y_col: str = "target_class",
        encoder: Optional[LabelEncoder] = None,
    ) -> Tuple[List[str], Dict]:
        threshold = self.target_config.multicollinearity_threshold
        max_pairs = self.target_config.max_high_corr_pairs
        selected = feature_cols.copy()
        dropped: List[str] = []
        enc = encoder if encoder is not None else self.label_encoder
        y_encoded = enc.fit_transform(train_df[y_col])
        try:
            mi_scores = mutual_info_classif(
                train_df[selected].fillna(0.0),
                y_encoded,
                random_state=42,
            )
            target_relevance = {col: float(score) for col, score in zip(selected, mi_scores)}
        except Exception:
            target_relevance = train_df[selected].corrwith(train_df[y_col]).abs().fillna(0.0).to_dict()

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
        y_col: str = "target_class",
        encoder: Optional[LabelEncoder] = None,
        perm_scoring: str = "f1_weighted",
    ) -> Tuple[List[str], Dict]:
        """Lightweight feature filtering using permutation importance on validation."""
        if len(feature_cols) <= 8:
            return feature_cols, {"kept_all": True, "dropped": [], "threshold": self.target_config.min_feature_importance}
        try:
            model = model_factory()
            enc = encoder if encoder is not None else self.label_encoder
            y_train = enc.transform(train_df[y_col])
            y_val = enc.transform(val_df[y_col])
            model.fit(train_df[feature_cols], y_train)
            perm = permutation_importance(
                model,
                val_df[feature_cols],
                y_val,
                n_repeats=3,
                random_state=42,
                scoring=perm_scoring,
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

    @staticmethod
    def _symbol_date_return_arrays(processed: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Per symbol: (sorted date array, return array) for O(1) slices with date < dt via searchsorted."""
        if processed is None or processed.empty:
            return {}
        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for sym, g in processed.sort_values("date").groupby("symbol", sort=False):
            out[str(sym)] = (g["date"].to_numpy(), np.asarray(g["daily_return"].values, dtype=float))
        return out

    def _rolling_joint_mu_sigma_before_dt(
        self,
        sym: str,
        dt,
        sym_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]],
        tail_max: int = 500,
        min_obs: int = 60,
    ) -> Tuple[Optional[float], Optional[float]]:
        """One-step joint AR(1)-GARCH on returns strictly before calendar dt (same as strict rolling)."""
        if arch_model is None or sym not in sym_arrays:
            return None, None
        dates, rets = sym_arrays[sym]
        pos = int(np.searchsorted(dates, dt, side="left"))
        hist = rets[:pos]
        hist = hist[np.isfinite(hist)]
        if len(hist) < min_obs:
            return None, None
        tail = hist[-tail_max:]
        mu_arr, vol_arr, _ = _joint_ar1_garch_fit_forecast(tail * _ECON_RETURN_SCALE, horizon=1)
        if mu_arr is None or len(mu_arr) < 1:
            return None, None
        mv = float(mu_arr[0])
        sig = float(vol_arr[0]) if vol_arr is not None and len(vol_arr) > 0 else None
        return mv, sig

    def _append_rolling_econ_ordered(
        self,
        dt,
        hold_df: pd.DataFrame,
        sym_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]],
        include_garch_vol: bool,
        arima_true: List[str],
        arima_pred: List[str],
        garch_vol_rows: List[float],
    ) -> None:
        """One joint fit per (dt, symbol); append labels in hold_df row order to stay aligned with ML."""
        if arch_model is None or not sym_arrays:
            for _, row in hold_df.iterrows():
                arima_true.append(str(row["target_class"]))
                arima_pred.append("SIDEWAY")
            return
        fc_cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for sym in hold_df["symbol"].unique():
            fc_cache[str(sym)] = self._rolling_joint_mu_sigma_before_dt(str(sym), dt, sym_arrays)
        for _, row in hold_df.iterrows():
            sym = str(row["symbol"])
            mu, sig = fc_cache[sym]
            arima_true.append(str(row["target_class"]))
            if mu is not None:
                tr = self._trend_label(mu * 100.0)
                arima_pred.append("SIDEWAY" if pd.isna(tr) else str(tr))
            else:
                arima_pred.append("SIDEWAY")
            if include_garch_vol and sig is not None:
                garch_vol_rows.append(sig)

    def _rolling_append_arima_garch_row(
        self,
        dt,
        row: pd.Series,
        arima_true: List[str],
        arima_pred: List[str],
        garch_vol_rows: List[float],
        include_garch_vol: bool,
    ) -> None:
        """Single-row wrapper; builds symbol index (prefer batched _append_rolling_econ_ordered in rolling loops)."""
        if self.processed_data is None or self.processed_data.empty:
            arima_true.append(str(row["target_class"]))
            arima_pred.append("SIDEWAY")
            return
        sym_arrays = self._symbol_date_return_arrays(self.processed_data)
        self._append_rolling_econ_ordered(
            dt,
            pd.DataFrame([row]),
            sym_arrays,
            include_garch_vol,
            arima_true,
            arima_pred,
            garch_vol_rows,
        )

    def evaluate_rolling_forecast_short(
        self,
        max_dates: Optional[int] = None,
        min_train_rows: Optional[int] = None,
        include_arima_garch_step: bool = True,
        include_garch_vol: bool = True,
        per_symbol_models: bool = False,
        min_train_rows_per_symbol: int = 220,
    ) -> Dict:
        """
        True rolling-origin evaluation: for each test calendar date dt, train only on rows with date < dt,
        then predict rows at date dt (no leakage from dt into features beyond same-row contemporaneous fields).

        Uses standard ML feature columns (never econometric forecast columns as inputs). Optionally fits fresh
        joint AR(1)-GARCH per symbol on returns strictly before dt (arch only; no separate ARIMA/GARCH fallback).
        Econometric step: one joint fit per (calendar dt, symbol) via pre-indexed return arrays (no per-row refit).

        max_dates:
            None — use every test-period trading day (full rolling, may be slow).
            int — cap number of distinct test dates from the start of the test window.

        per_symbol_models:
            False — one pooled RandomForest trained on all symbols each dt (panel model).
            True — one RandomForest per symbol, trained only on that symbol's history before dt (FX-style).
        """
        if self.feature_data.empty:
            return {"ok": False, "error": "feature_data empty"}

        ml_cols = list(self.feature_cols)
        ordered = self.feature_data.sort_values("date").reset_index(drop=True)
        n = len(ordered)
        test_start = int(n * (1 - self.target_config.test_ratio))
        if test_start <= 0 or test_start >= n:
            return {"ok": False, "error": "invalid time split"}

        all_test_dates = sorted(ordered.iloc[test_start:]["date"].unique().tolist())
        if max_dates is None:
            test_dates = all_test_dates
        else:
            test_dates = all_test_dates[: max(0, int(max_dates))]
        if not test_dates:
            return {"ok": False, "error": "no test dates"}

        pool_min = min_train_rows if min_train_rows is not None else (min_train_rows_per_symbol if per_symbol_models else 800)

        rf_factory = lambda: RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            random_state=42,
            n_jobs=2,
            class_weight="balanced",
        )

        if hasattr(self.label_encoder, "classes_") and len(self.label_encoder.classes_) > 0:
            global_classes: List[str] = [str(c) for c in self.label_encoder.classes_]
        else:
            global_classes = sorted({str(x) for x in ordered["target_class"].dropna().unique().tolist()})

        sym_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if include_arima_garch_step and arch_model is not None and self.processed_data is not None and len(self.processed_data) > 0:
            sym_arrays = self._symbol_date_return_arrays(self.processed_data)

        def _align_proba_row(proba_row: np.ndarray, le_classes, g_classes: List[str]) -> np.ndarray:
            arr = np.zeros(len(g_classes), dtype=float)
            lc = [str(c) for c in le_classes]
            for j, gc in enumerate(g_classes):
                if gc in lc:
                    arr[j] = float(proba_row[lc.index(gc)])
            s = float(arr.sum())
            if s > 1e-12:
                arr /= s
            return arr

        def _fuse_weighted(ml_p: np.ndarray, econ_lbl: str, w: float) -> str:
            h = np.zeros(len(global_classes), dtype=float)
            el = str(econ_lbl)
            if el in global_classes:
                h[global_classes.index(el)] = 1.0
            elif "SIDEWAY" in global_classes:
                h[global_classes.index("SIDEWAY")] = 1.0
            else:
                h[0] = 1.0
            ww = float(np.clip(w, 0.0, 1.0))
            fused = ww * ml_p + (1.0 - ww) * h
            fused /= fused.sum() + 1e-12
            return str(global_classes[int(np.argmax(fused))])

        def collect_for_dates(
            dates_iter: List,
        ) -> Tuple[List[str], List[str], List[np.ndarray], List[str], List[str], List[float], List[str]]:
            ml_true_l: List[str] = []
            ml_pred_l: List[str] = []
            ml_proba_l: List[np.ndarray] = []
            arima_true_l: List[str] = []
            arima_pred_l: List[str] = []
            garch_vol_l: List[float] = []
            dates_u: List[str] = []
            for dt in dates_iter:
                hold_part = ordered[ordered["date"] == dt]
                hold_clean_all = hold_part.dropna(subset=ml_cols + ["target_class"])
                if per_symbol_models:
                    for sym in sorted(hold_clean_all["symbol"].unique()):
                        train_clean = ordered[(ordered["date"] < dt) & (ordered["symbol"] == sym)].dropna(
                            subset=ml_cols + ["target_class"]
                        )
                        hold_clean = hold_clean_all[hold_clean_all["symbol"] == sym]
                        if len(train_clean) < min_train_rows_per_symbol or hold_clean.empty:
                            continue
                        le = LabelEncoder()
                        y_train = le.fit_transform(train_clean["target_class"])
                        rf = rf_factory()
                        rf.fit(train_clean[ml_cols], y_train)
                        proba = rf.predict_proba(hold_clean[ml_cols])
                        pred_enc = rf.predict(hold_clean[ml_cols])
                        pred_labels = le.inverse_transform(pred_enc)
                        for i in range(len(hold_clean)):
                            ml_true_l.append(str(hold_clean["target_class"].iloc[i]))
                            ml_pred_l.append(str(pred_labels[i]))
                            ml_proba_l.append(_align_proba_row(proba[i], le.classes_, global_classes))
                        if include_arima_garch_step and arch_model is not None:
                            self._append_rolling_econ_ordered(
                                dt,
                                hold_clean,
                                sym_arrays,
                                include_garch_vol,
                                arima_true_l,
                                arima_pred_l,
                                garch_vol_l,
                            )
                else:
                    train_part = ordered[ordered["date"] < dt]
                    train_clean = train_part.dropna(subset=ml_cols + ["target_class"])
                    hold_clean = hold_clean_all
                    if len(train_clean) < pool_min or hold_clean.empty:
                        continue
                    le = LabelEncoder()
                    y_train = le.fit_transform(train_clean["target_class"])
                    rf = rf_factory()
                    rf.fit(train_clean[ml_cols], y_train)
                    proba = rf.predict_proba(hold_clean[ml_cols])
                    pred_enc = rf.predict(hold_clean[ml_cols])
                    pred_labels = le.inverse_transform(pred_enc)
                    for i in range(len(hold_clean)):
                        ml_true_l.append(str(hold_clean["target_class"].iloc[i]))
                        ml_pred_l.append(str(pred_labels[i]))
                        ml_proba_l.append(_align_proba_row(proba[i], le.classes_, global_classes))
                    if include_arima_garch_step and arch_model is not None:
                        self._append_rolling_econ_ordered(
                            dt,
                            hold_clean,
                            sym_arrays,
                            include_garch_vol,
                            arima_true_l,
                            arima_pred_l,
                            garch_vol_l,
                        )
                dates_u.append(dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt))
            return ml_true_l, ml_pred_l, ml_proba_l, arima_true_l, arima_pred_l, garch_vol_l, dates_u

        val_start = int(test_start * (1 - self.target_config.val_ratio))
        val_dates = sorted(ordered.iloc[val_start:test_start]["date"].unique().tolist())
        best_w = 1.0
        fusion_val_f1_by_w: Dict[str, float] = {}
        best_val_f1_ens: Optional[float] = None
        if include_arima_garch_step and arch_model is not None and len(val_dates) > 0:
            vt, _vp, vpr, _vat, vap, _, _dv = collect_for_dates(val_dates)
            if len(vpr) == len(vap) == len(vt) and len(vt) > 0:
                for w in np.linspace(0.0, 1.0, 11):
                    ens_v = [_fuse_weighted(vpr[i], vap[i], float(w)) for i in range(len(vt))]
                    fv = float(f1_score(vt, ens_v, average="weighted", zero_division=0))
                    fusion_val_f1_by_w[f"{w:.2f}"] = fv
                    if best_val_f1_ens is None or fv > best_val_f1_ens:
                        best_val_f1_ens = fv
                        best_w = float(w)

        ml_true, ml_pred, ml_proba_rows, arima_true, arima_pred, garch_vol_rows, dates_used = collect_for_dates(
            test_dates
        )

        ensemble_pred: List[str] = []
        if (
            include_arima_garch_step
            and arch_model is not None
            and len(ml_proba_rows) == len(arima_pred) == len(ml_true)
            and len(ml_true) > 0
        ):
            ensemble_pred = [_fuse_weighted(ml_proba_rows[i], arima_pred[i], best_w) for i in range(len(ml_true))]
        else:
            ensemble_pred = list(ml_pred)

        def acc(a: List[str], b: List[str]) -> Optional[float]:
            if not a or len(a) != len(b):
                return None
            return float(accuracy_score(a, b))

        hybrid_agree = None
        if ml_pred and arima_pred and len(ml_pred) == len(arima_pred):
            hybrid_agree = float(np.mean(np.array(ml_pred) == np.array(arima_pred)))

        ml_f1w: Optional[float] = (
            float(f1_score(ml_true, ml_pred, average="weighted", zero_division=0)) if ml_true else None
        )
        arima_f1w: Optional[float] = None
        if arima_true and arima_pred and len(arima_true) == len(arima_pred):
            arima_f1w = float(f1_score(arima_true, arima_pred, average="weighted", zero_division=0))

        ens_f1w: Optional[float] = (
            float(f1_score(ml_true, ensemble_pred, average="weighted", zero_division=0)) if ml_true else None
        )
        acc_ml = acc(ml_true, ml_pred)
        acc_ar = acc(arima_true, arima_pred) if arima_pred else None
        acc_ens = acc(ml_true, ensemble_pred) if ml_true else None
        f1_delta_ml_minus_econ: Optional[float] = None
        if ml_f1w is not None and arima_f1w is not None:
            f1_delta_ml_minus_econ = float(ml_f1w - arima_f1w)
        acc_delta_ml_minus_econ: Optional[float] = None
        if isinstance(acc_ml, float) and isinstance(acc_ar, float):
            acc_delta_ml_minus_econ = float(acc_ml - acc_ar)
        f1_delta_ensemble_minus_ml: Optional[float] = None
        if ens_f1w is not None and ml_f1w is not None:
            f1_delta_ensemble_minus_ml = float(ens_f1w - ml_f1w)
        f1_delta_ensemble_minus_econ: Optional[float] = None
        if ens_f1w is not None and arima_f1w is not None:
            f1_delta_ensemble_minus_econ = float(ens_f1w - arima_f1w)
        acc_delta_ensemble_minus_ml: Optional[float] = None
        if isinstance(acc_ens, float) and isinstance(acc_ml, float):
            acc_delta_ensemble_minus_ml = float(acc_ens - acc_ml)

        out = {
            "ok": True,
            "max_dates_requested": max_dates if max_dates is not None else "full_test_period",
            "per_symbol_models": per_symbol_models,
            "dates_evaluated": len(dates_used),
            "date_list": dates_used,
            "ml_samples": len(ml_true),
            "ml_accuracy": acc_ml,
            "ml_f1_weighted": ml_f1w,
            "arima_accuracy_vs_labels": acc_ar,
            "arima_f1_weighted": arima_f1w,
            "arima_samples": len(arima_true),
            "accuracy_delta_ml_minus_econ": acc_delta_ml_minus_econ,
            "f1_delta_ml_minus_econ": f1_delta_ml_minus_econ,
            "hybrid_ml_arima_agreement": hybrid_agree,
            "garch_vol_mean": float(np.mean(garch_vol_rows)) if garch_vol_rows else None,
            "ml_feature_count": len(ml_cols),
            "fusion_weight_ml_tuned": float(best_w),
            "fusion_val_f1_by_w": fusion_val_f1_by_w if fusion_val_f1_by_w else None,
            "fusion_val_f1_best": best_val_f1_ens,
            "ensemble_accuracy": acc_ens,
            "ensemble_f1_weighted": ens_f1w,
            "f1_delta_ensemble_minus_ml": f1_delta_ensemble_minus_ml,
            "f1_delta_ensemble_minus_econ": f1_delta_ensemble_minus_econ,
            "accuracy_delta_ensemble_minus_ml": acc_delta_ensemble_minus_ml,
            "note": "ML retrains each dt using rows with date < dt; joint AR(1)-GARCH refit per symbol on "
            "returns before dt (arch only). If per_symbol_models=True, ML is one RF per symbol (FX-style). "
            "max_dates=None uses every test day. "
            "arima_* = econometric track (joint AR(1)-GARCH mean return → trend label). "
            "*_delta_ml_minus_econ > 0 means ML better than econometric on that metric. "
            "Ensemble = w·P(ML) + (1−w)·onehot(econ) on validation-tuned w (grid on rolling val dates only; "
            "no test peek). If arch/econ disabled, w=1 and ensemble matches ML.",
        }
        return out

    def _attach_rolling_forecast_to_benchmark(self) -> None:
        """Append rolling-origin evaluation to benchmark_summary (after build_benchmark_summary)."""
        cfg = self.target_config
        cap = cfg.rolling_forecast_max_dates
        if self.feature_data.empty or (cap is not None and cap <= 0):
            self.benchmark_summary["rolling_forecast_eval"] = {
                "skipped": True,
                "reason": "rolling_forecast_max_dates <= 0 or no feature_data",
            }
            return

        max_dates_arg: Optional[int] = None if cap is None else int(cap)
        cap_label = "full_test_window" if cap is None else str(int(cap))
        print(
            f"Rolling-origin trend benchmark (max_dates={cap_label}, pooled + per-symbol ML; "
            f"econ_step={cfg.rolling_forecast_include_econ_step})..."
        )
        block: Dict = {"skipped": False, "max_dates_cap": cap_label, "pooled": {}, "per_symbol": {}}
        try:
            block["pooled"] = self.evaluate_rolling_forecast_short(
                max_dates=max_dates_arg,
                min_train_rows=800,
                include_arima_garch_step=bool(cfg.rolling_forecast_include_econ_step),
                include_garch_vol=bool(cfg.rolling_forecast_include_garch_vol),
                per_symbol_models=False,
            )
        except Exception as exc:
            block["pooled"] = {"ok": False, "error": str(exc)}
        try:
            block["per_symbol"] = self.evaluate_rolling_forecast_short(
                max_dates=max_dates_arg,
                min_train_rows_per_symbol=220,
                include_arima_garch_step=bool(cfg.rolling_forecast_include_econ_step),
                include_garch_vol=bool(cfg.rolling_forecast_include_garch_vol),
                per_symbol_models=True,
            )
        except Exception as exc:
            block["per_symbol"] = {"ok": False, "error": str(exc)}

        self.benchmark_summary["rolling_forecast_eval"] = block
        for key in ("pooled", "per_symbol"):
            sub = block.get(key) or {}
            if sub.get("ok") and sub.get("ml_f1_weighted") is not None:
                acc = sub.get("ml_accuracy")
                acc_s = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)
                extra = ""
                ef1 = sub.get("arima_f1_weighted")
                if isinstance(ef1, float):
                    extra += f" econ_F1w={ef1:.4f}"
                df1 = sub.get("f1_delta_ml_minus_econ")
                if isinstance(df1, float):
                    extra += f" ΔF1(ml−econ)={df1:+.4f}"
                ens_f1 = sub.get("ensemble_f1_weighted")
                if isinstance(ens_f1, float):
                    dem = sub.get("f1_delta_ensemble_minus_ml")
                    dem_s = f" ΔF1(ens−ml)={dem:+.4f}" if isinstance(dem, float) else ""
                    w_tune = sub.get("fusion_weight_ml_tuned")
                    w_s = f" w_ml={w_tune:.2f}" if isinstance(w_tune, float) else ""
                    extra += f" | ens_F1w={ens_f1:.4f}{dem_s}{w_s}"
                print(
                    f"  rolling {key}: ML acc={acc_s} F1w={float(sub['ml_f1_weighted']):.4f}{extra} "
                    f"samples={sub.get('ml_samples')}"
                )

        pooled_b = block.get("pooled") or {}
        w_save = pooled_b.get("fusion_weight_ml_tuned")
        if pooled_b.get("ok") and isinstance(w_save, (int, float)):
            self._rolling_fusion_weight_ml = float(w_save)

    def _train_ml_models_meta(self) -> bool:
        print(
            "Training ML models (AFML meta-labeling: EMA primary + triple-barrier, binary P(useful))..."
        )
        if self.feature_data.empty or "meta_label" not in self.feature_data.columns:
            return False
        meta_cols = self.meta_feature_cols_full
        if not meta_cols or "primary_side" not in self.feature_data.columns:
            return False

        ordered = self.feature_data.sort_values("date").reset_index(drop=True)
        n = len(ordered)
        test_start = int(n * (1 - self.target_config.test_ratio))
        val_start = int(test_start * (1 - self.target_config.val_ratio))
        train_df = ordered.iloc[:val_start].copy()
        val_df = ordered.iloc[val_start:test_start].copy()
        test_df = ordered.iloc[test_start:].copy()
        if train_df.empty or val_df.empty or test_df.empty:
            print("Insufficient data after train/val/test split (meta).")
            return False

        active_regime_cols = [f for f in self.regime_feature_cols if f in meta_cols]
        active_market_cols = [f for f in self.market_context_feature_cols if f in meta_cols]
        core_base = [f for f in self.core_feature_cols if f in meta_cols]
        core_with_regime = core_base + active_regime_cols + active_market_cols
        feature_sets = {
            "core": core_with_regime,
            "core_seasonal": [
                f for f in (core_with_regime + self.seasonal_feature_cols) if f in meta_cols
            ],
        }
        unique_feature_sets: Dict[str, List[str]] = {}
        seen_signatures = set()
        for set_name, cols in feature_sets.items():
            signature = tuple(cols)
            if signature and signature not in seen_signatures:
                unique_feature_sets[set_name] = cols
                seen_signatures.add(signature)
        feature_sets = unique_feature_sets

        y_train = self.meta_label_encoder.fit_transform(train_df["meta_label"])
        y_val = self.meta_label_encoder.transform(val_df["meta_label"])
        y_test = self.meta_label_encoder.transform(test_df["meta_label"])

        model_factories = {
            "RandomForest": lambda: RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=2,
                class_weight="balanced",
            ),
            "LogisticRegression": lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
            "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
            ),
            "GradientBoosting": lambda: GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                random_state=42,
            ),
        }
        if XGBClassifier is not None:
            model_factories["XGBoost"] = lambda: XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            )
        if LGBMClassifier is not None:
            model_factories["LightGBM"] = lambda: LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=48,
                class_weight="balanced",
                objective="binary",
                random_state=42,
                verbose=-1,
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

        results: Dict = {}
        self.models = {}
        self.ml_predictions_store = {}
        self.model_feature_cols = {}
        all_selection_reports: Dict = {}
        ml_checks_aggregate = {"passed": True, "errors": [], "warnings": [], "details": {"by_feature_set": {}}}

        def train_candidate(model_factory, original_cols: List[str], selected_cols: List[str]):
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

            train_val_df = pd.concat([train_df, val_df], ignore_index=True)
            y_train_val = self.meta_label_encoder.transform(train_val_df["meta_label"])
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
            metrics["labels"] = self.meta_label_encoder.classes_.tolist()
            metrics["feature_count"] = len(selected_cols)
            metrics["meta_labeling"] = True
            delta = {
                "accuracy_delta": round(metrics["accuracy"] - baseline_metrics["accuracy"], 6),
                "f1_delta": round(metrics["f1_weighted"] - baseline_metrics["f1_weighted"], 6),
                "precision_delta": round(metrics["precision_weighted"] - baseline_metrics["precision_weighted"], 6),
            }
            return model, pred, proba, baseline_pred, baseline_proba, metrics, baseline_metrics, delta

        for name, model_factory in model_factories.items():
            candidate_results = {}
            for set_name, original_cols in feature_sets.items():
                selected_cols, selection_report = self._prune_multicollinearity(
                    train_df, original_cols, y_col="meta_label", encoder=self.meta_label_encoder
                )
                importance_cols, importance_report = self._filter_by_importance(
                    model_factory,
                    train_df,
                    val_df,
                    selected_cols,
                    y_col="meta_label",
                    encoder=self.meta_label_encoder,
                    perm_scoring="f1",
                )
                checks = self._run_ml_data_checks(
                    train_df, val_df, importance_cols, y_col="meta_label", binary=True
                )
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
                    msg = f"{name}/{set_name} training failed (meta): {exc}"
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

            enc_inv = self.meta_label_encoder
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
                "model_type": "meta_labeling",
            }
            self.models[name] = model
            self.model_feature_cols[name] = selected_feature_cols
            self.ml_predictions_store[name] = {
                "y_test_encoded": y_test.tolist(),
                "y_test_labels": [enc_inv.inverse_transform([v])[0] for v in y_test],
                "pred_encoded": pred.tolist(),
                "pred_labels": [enc_inv.inverse_transform([v])[0] for v in pred],
                "pred_proba": proba.tolist() if proba is not None else None,
                "baseline_pred_encoded": baseline_pred.tolist(),
                "baseline_pred_labels": [enc_inv.inverse_transform([v])[0] for v in baseline_pred],
                "test_frame": test_df[
                    ["date", "symbol", "next_return_pct", "daily_return", "target_class", "meta_label", "primary_side"]
                ].copy(),
            }
            print(
                f"{name} (meta) metrics:",
                f"baseline acc={baseline_metrics['accuracy']:.4f}, pruned acc={metrics['accuracy']:.4f},",
                f"baseline f1={baseline_metrics['f1_weighted']:.4f}, pruned f1={metrics['f1_weighted']:.4f},",
                f"set={best_set}",
            )

        if not results:
            print("No ML candidate passed checks (meta-labeling).")
            self._use_meta_labeling = False
            return False

        self.data_significance_checks["ml"] = ml_checks_aggregate
        if ml_checks_aggregate["warnings"]:
            print("ML data checks warnings (meta):")
            for warning in sorted(set(ml_checks_aggregate["warnings"])):
                print(f" - {warning}")

        best_overall_ml = max(results.keys(), key=lambda n: results[n]["metrics"]["f1_weighted"])
        self.feature_cols = self.model_feature_cols[best_overall_ml]
        self.feature_selection_report = results[best_overall_ml]["feature_selection"]
        self.ml_results = results
        self._use_meta_labeling = True
        return True

    def train_ml_models(self) -> bool:
        if self.target_config.use_meta_labeling:
            return self._train_ml_models_meta()
        self._use_meta_labeling = False
        print(
            "Training ML models (RF, XGBoost, LogReg, HistGB, GBT, optional LightGBM, optional LSTM)..."
        )
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

        active_regime_cols = [f for f in self.regime_feature_cols if f in self.feature_cols]
        active_market_cols = [f for f in self.market_context_feature_cols if f in self.feature_cols]
        core_base = [f for f in self.core_feature_cols if f in self.feature_cols]
        core_with_regime = core_base + active_regime_cols + active_market_cols
        feature_sets = {
            "core": core_with_regime,
            "core_seasonal": [
                f for f in (core_with_regime + self.seasonal_feature_cols) if f in self.feature_cols
            ],
        }
        # Drop duplicate feature sets when seasonal features are disabled or unavailable.
        unique_feature_sets = {}
        seen_signatures = set()
        for set_name, cols in feature_sets.items():
            signature = tuple(cols)
            if signature and signature not in seen_signatures:
                unique_feature_sets[set_name] = cols
                seen_signatures.add(signature)
        feature_sets = unique_feature_sets

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
            ),
            "LogisticRegression": lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    multi_class="multinomial",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
            "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
            ),
            "GradientBoosting": lambda: GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                random_state=42,
            ),
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
        if LGBMClassifier is not None:
            model_factories["LightGBM"] = lambda: LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=48,
                class_weight="balanced",
                objective="multiclass",
                random_state=42,
                verbose=-1,
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

        self.data_significance_checks["ml"] = ml_checks_aggregate
        if ml_checks_aggregate["warnings"]:
            print("ML data checks warnings:")
            for warning in sorted(set(ml_checks_aggregate["warnings"])):
                print(f" - {warning}")

        # Default feature_cols follows best-F1 ML model for downstream utilities.
        best_overall_ml = max(results.keys(), key=lambda n: results[n]["metrics"]["f1_weighted"])
        self.feature_cols = self.model_feature_cols[best_overall_ml]
        self.feature_selection_report = results[best_overall_ml]["feature_selection"]
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
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    metrics["roc_auc_ovr_weighted"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                else:
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
        print("Training econometric models (joint AR(1)-GARCH only via arch; no split-model fallback)...")
        if self.feature_data.empty:
            return False
        if arch_model is None:
            print("arch package required for joint AR(1)-GARCH; skip econometric models.")
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

            train_return = symbol_train["daily_return"].dropna()
            test_return = symbol_test["daily_return"].dropna()
            if len(train_return) < 60 or test_return.empty:
                continue

            econ_checks = self._run_econometric_checks(train_return)
            check_results[symbol] = econ_checks
            if not econ_checks["passed"]:
                continue

            train_scaled = train_return.values.astype(float) * _ECON_RETURN_SCALE
            h = len(test_return)
            mu_joint, vol_joint, meta_joint = _joint_ar1_garch_fit_forecast(train_scaled, horizon=h)

            if mu_joint is None or len(mu_joint) < h:
                msg = meta_joint.get("error", "short_history_or_fit_failed") if isinstance(meta_joint, dict) else "fit_failed"
                print(f"  Joint AR1-GARCH skipped {symbol}: {msg}")
                continue

            symbol_result: Dict = {
                "model_type": "econometric",
                "metrics": {},
                "p_values": {},
                "fit_method": "joint_AR1_GARCH",
            }
            if isinstance(meta_joint, dict) and "pvalues" in meta_joint:
                symbol_result["p_values"]["joint_AR1_GARCH"] = meta_joint["pvalues"]

            garch_vol = float(vol_joint[0]) if vol_joint is not None and len(vol_joint) >= 1 else None

            mean_fc_ret = mu_joint[:h]
            pred_labels = [self._trend_label(float(v * 100.0)) for v in mean_fc_ret]
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
        print(f"Econometric model evaluated for {len(results)} symbols (joint only).")
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
                    "model_name": "AR1-GARCH (joint)",
                    "model_variant": "ARX(1)-GARCH(1,1) joint (arch) only",
                    "selected_feature_set": None,
                    **agg,
                }
            )

        benchmark_df = pd.DataFrame(benchmark_rows)
        benchmark_df = benchmark_df.sort_values(by=["accuracy", "f1_weighted"], ascending=False, na_position="last")
        top_raw = benchmark_df.iloc[0].to_dict() if not benchmark_df.empty else {}

        def _top_model_scalar_ok(val: object) -> bool:
            if isinstance(val, (list, tuple, dict)):
                return False
            if isinstance(val, np.ndarray):
                return False
            try:
                return not bool(pd.isna(val))
            except (ValueError, TypeError):
                return False

        top_model = {k: v for k, v in top_raw.items() if _top_model_scalar_ok(v)}
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
        if not self.feature_data.empty and getattr(self, "_use_meta_labeling", False):
            tc = sorted(self.feature_data["target_class"].dropna().unique())
            if tc:
                self.label_encoder.fit(tc)
        self.train_econometric_models()
        self.build_benchmark_summary()
        self._attach_rolling_forecast_to_benchmark()
        return ml_ok

    def save_model(self) -> bool:
        if not self.models:
            return False

        bundle = {
            "models": self.models,
            "feature_cols": self.feature_cols,
            "model_feature_cols": self.model_feature_cols,
            "core_feature_cols": self.core_feature_cols,
            "regime_feature_cols": self.regime_feature_cols,
            "hybrid_feature_cols": self.hybrid_feature_cols,
            "seasonal_feature_cols": self.seasonal_feature_cols,
            "market_context_feature_cols": self.market_context_feature_cols,
            "top_stocks": self.top_stocks,
            "label_classes": self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, "classes_") else [],
            "use_meta_labeling": self._use_meta_labeling,
            "meta_label_classes": self.meta_label_encoder.classes_.tolist()
            if hasattr(self.meta_label_encoder, "classes_")
            else [],
            "meta_prob_trade_threshold": float(self.target_config.meta_prob_trade_threshold),
            "benchmark_summary": self.benchmark_summary,
            "symbol_performance": self.symbol_performance.to_dict(orient="records"),
            "target_config": self.target_config.__dict__,
            "data_significance_checks": self.data_significance_checks,
            "feature_selection_report": self.feature_selection_report,
            "ml_predictions_store": self.ml_predictions_store,
            "rolling_fusion_weight_ml": self._rolling_fusion_weight_ml,
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
        import sys

        bundle_path = self.current_dir / self.BUNDLE_PATH
        if not bundle_path.exists():
            return False
        # Bundles created via `python stock_analysis_yfinance.py` pickle LSTM etc. as __main__.ClassName;
        # when loading from web_app / start_web, point __main__ at this module for unpickling only.
        _prev_main = sys.modules.get("__main__")
        try:
            sys.modules["__main__"] = sys.modules[__name__]
            try:
                bundle = joblib.load(bundle_path)
            except TypeError as exc:
                if "StringDtype" in str(exc) or "positional arguments" in str(exc):
                    raise RuntimeError(
                        "Không unpickle bundle: pandas/numpy trên máy này khác lúc train (joblib). "
                        "Trên Colab: chạy  %pip install -U 'pandas>=2.2' 'numpy>=1.26'  "
                        "rồi Runtime → Restart session, sau đó chạy lại từ ô pip."
                    ) from exc
                raise
        finally:
            if _prev_main is not None:
                sys.modules["__main__"] = _prev_main
        self.models = bundle.get("models", {})
        self.feature_cols = bundle.get("feature_cols", [])
        self.model_feature_cols = bundle.get("model_feature_cols", {})
        self.core_feature_cols = bundle.get("core_feature_cols", self.core_feature_cols)
        self.regime_feature_cols = bundle.get("regime_feature_cols", self.regime_feature_cols)
        self.hybrid_feature_cols = bundle.get("hybrid_feature_cols", self.hybrid_feature_cols)
        self.seasonal_feature_cols = bundle.get("seasonal_feature_cols", self.seasonal_feature_cols)
        self.market_context_feature_cols = bundle.get("market_context_feature_cols", self.market_context_feature_cols)
        self.top_stocks = bundle.get("top_stocks", [])
        self.benchmark_summary = bundle.get("benchmark_summary", {})
        self.data_significance_checks = bundle.get("data_significance_checks", {"ml": {}, "econometric": {}})
        self.feature_selection_report = bundle.get("feature_selection_report", {})
        self.ml_predictions_store = bundle.get("ml_predictions_store", {})
        w_load = bundle.get("rolling_fusion_weight_ml")
        self._rolling_fusion_weight_ml = float(w_load) if isinstance(w_load, (int, float)) else None
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
        meta_classes = bundle.get("meta_label_classes", [])
        if meta_classes:
            self.meta_label_encoder.fit(meta_classes)
        um = bundle.get("use_meta_labeling")
        self._use_meta_labeling = bool(um) if um is not None else False
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

    def predict_trend_fusion(self, symbol: str, model_name: str = "RandomForest") -> Optional[Dict]:
        """
        Inference: blend ML class probabilities with joint AR(1)-GARCH one-step mean → trend label.
        Weight w on ML is `rolling_fusion_weight_ml` from the last rolling val tune (saved in bundle), else 1.0.
        """
        if model_name not in self.models:
            if self.models:
                model_name = list(self.models.keys())[0]
            else:
                return None
        latest = self.feature_data[self.feature_data["symbol"] == symbol].tail(1)
        if latest.empty:
            return None
        model = self.models[model_name]
        active_cols = self.model_feature_cols.get(model_name, self.feature_cols)
        features = latest[active_cols]
        proba_m = model.predict_proba(features)[0]
        g_classes = [str(c) for c in self.label_encoder.classes_]
        if getattr(self, "_use_meta_labeling", False) and "primary_side" in latest.columns:
            prob_map = {int(np.asarray(c).item()): float(p) for c, p in zip(model.classes_, proba_m)}
            p_trade = float(prob_map.get(1, 0.0))
            thr = float(self.target_config.meta_prob_trade_threshold)
            ml_p = np.zeros(len(g_classes), dtype=float)
            if p_trade < thr:
                if "SIDEWAY" in g_classes:
                    ml_p[g_classes.index("SIDEWAY")] = 1.0
                else:
                    ml_p[:] = 1.0 / max(len(g_classes), 1)
            else:
                ps = float(latest["primary_side"].iloc[0])
                if ps > 0 and "TANG" in g_classes:
                    ml_p[g_classes.index("TANG")] = p_trade
                elif ps <= 0 and "GIAM" in g_classes:
                    ml_p[g_classes.index("GIAM")] = p_trade
                rest = max(0.0, 1.0 - p_trade)
                if "SIDEWAY" in g_classes:
                    ml_p[g_classes.index("SIDEWAY")] = rest
                s2 = float(ml_p.sum())
                if s2 > 1e-12:
                    ml_p /= s2
            meta_extras = {"meta_prob_trade": p_trade, "meta_gated": p_trade < thr, "primary_side": float(latest["primary_side"].iloc[0])}
        elif getattr(self, "_use_meta_labeling", False):
            prob_map = {int(np.asarray(c).item()): float(p) for c, p in zip(model.classes_, proba_m)}
            p_trade = float(prob_map.get(1, 0.0))
            ml_p = np.zeros(len(g_classes), dtype=float)
            if "SIDEWAY" in g_classes:
                ml_p[g_classes.index("SIDEWAY")] = 1.0
            else:
                ml_p[:] = 1.0 / max(len(g_classes), 1)
            meta_extras = {"meta_prob_trade": p_trade, "meta_gated": True, "primary_side": None}
        else:
            meta_extras = {}
            cls_ord = [int(np.asarray(c).item()) for c in model.classes_]
            ml_p = np.zeros(len(g_classes), dtype=float)
            for j, c_int in enumerate(cls_ord):
                lbl = str(self.label_encoder.inverse_transform([c_int])[0])
                if lbl in g_classes:
                    ml_p[g_classes.index(lbl)] = float(proba_m[j])
            s = float(ml_p.sum())
            if s > 1e-12:
                ml_p /= s

        w_eff = self._rolling_fusion_weight_ml
        if w_eff is None:
            w_eff = 1.0
        w_eff = float(np.clip(w_eff, 0.0, 1.0))

        econ_lbl = "SIDEWAY"
        if arch_model is not None and self.processed_data is not None and not self.processed_data.empty:
            d_last = latest["date"].iloc[0]
            idx = self._symbol_date_return_arrays(self.processed_data)
            mu, _ = self._rolling_joint_mu_sigma_before_dt(str(symbol), d_last, idx)
            if mu is not None:
                tl = self._trend_label(float(mu) * 100.0)
                if tl is not None and not pd.isna(tl):
                    econ_lbl = str(tl)

        h = np.zeros(len(g_classes), dtype=float)
        if econ_lbl in g_classes:
            h[g_classes.index(econ_lbl)] = 1.0
        elif "SIDEWAY" in g_classes:
            h[g_classes.index("SIDEWAY")] = 1.0
        else:
            h[0] = 1.0
        fused = w_eff * ml_p + (1.0 - w_eff) * h
        fused /= fused.sum() + 1e-12
        pred_label = g_classes[int(np.argmax(fused))]
        out = {
            "symbol": symbol,
            "model_name": model_name,
            "trend_label": pred_label,
            "fusion_weight_ml": w_eff,
            "econ_label": econ_lbl,
            "ml_argmax_label": g_classes[int(np.argmax(ml_p))],
            "fused_proba": {g_classes[i]: float(fused[i]) for i in range(len(g_classes))},
        }
        out.update(meta_extras)
        return out

    def predict_tomorrow_fusion(self, symbol: str, model_name: str = "RandomForest") -> Optional[Dict]:
        """
        Like predict_tomorrow but trend / implied price use ML + joint AR(1)-GARCH fusion (rolling-tuned w_ml).
        """
        current_info = self.get_current_stock_info(symbol)
        if not current_info:
            return None
        fusion = self.predict_trend_fusion(symbol, model_name=model_name)
        if not fusion:
            return None
        pred_label = str(fusion["trend_label"])
        current_price = float(current_info["current_price"])
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
        fp = fusion.get("fused_proba") or {}
        confidence = float(max(fp.values())) if fp else None
        return {
            "symbol": symbol,
            "model_name": model_name,
            "current_price": current_price,
            "predicted_price": float(predicted_price),
            "price_change": float(predicted_price - current_price),
            "price_change_percent": float(expected_pct),
            "trend": trend,
            "trend_icon": trend_icon,
            "trend_color": trend_color,
            "confidence": confidence,
            "date": current_info["date"],
            "fusion_weight_ml": fusion.get("fusion_weight_ml"),
            "econ_label": fusion.get("econ_label"),
            "ml_argmax_label": fusion.get("ml_argmax_label"),
            "fused_proba": fp,
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
        proba = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None
        if getattr(self, "_use_meta_labeling", False) and "primary_side" in latest.columns:
            prob_map = (
                {int(np.asarray(c).item()): float(p) for c, p in zip(model.classes_, proba)} if proba is not None else {}
            )
            p_trade = float(prob_map.get(1, 0.0))
            thr = float(self.target_config.meta_prob_trade_threshold)
            if p_trade < thr or np.isnan(latest["primary_side"].iloc[0]):
                pred_label = "SIDEWAY"
            else:
                pred_label = "TANG" if float(latest["primary_side"].iloc[0]) > 0 else "GIAM"
        elif getattr(self, "_use_meta_labeling", False):
            pred_label = "SIDEWAY"
        else:
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
        if getattr(self, "_use_meta_labeling", False) and proba is not None:
            prob_map = {int(np.asarray(c).item()): float(p) for c, p in zip(model.classes_, proba)}
            confidence = float(prob_map.get(1, 0.0))
        else:
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