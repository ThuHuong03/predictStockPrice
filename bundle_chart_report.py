"""
Đọc `analysis_bundle.joblib` (đã train) từ StockAnalysisYFinance.load_model_bundle()
và xuất các biểu đồ báo cáo cuối vào thư mục chỉ định.

Chạy CLI:  python bundle_chart_report.py [--root .] [--out artifacts/colab_charts]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def plot_ranking_bar(summary: Dict[str, Any], out_path: Path) -> bool:
    rows = summary.get("ranking") or []
    if not rows:
        return False
    df = pd.DataFrame(rows)
    name_col = "model_variant" if "model_variant" in df.columns else "model_name"
    if name_col not in df.columns:
        return False
    df = df.head(12).copy()
    df["_label"] = df[name_col].astype(str).str.slice(0, 42)
    x = np.arange(len(df))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    acc = [_safe_float(v) for v in df.get("accuracy", [])]
    f1 = [_safe_float(v) for v in df.get("f1_weighted", [])]
    ax.bar(x - w / 2, [v if v is not None else 0 for v in acc], width=w, label="accuracy")
    ax.bar(x + w / 2, [v if v is not None else 0 for v in f1], width=w, label="f1_weighted")
    ax.set_xticks(x)
    ax.set_xticklabels(df["_label"], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Benchmark ranking (accuracy vs F1 weighted)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def plot_checklist(summary: Dict[str, Any], out_path: Path) -> bool:
    chk = summary.get("benchmark_checklist") or {}
    items = chk.get("items") or {}
    if not items:
        return False
    keys = list(items.keys())
    vals = [1.0 if items[k].get("passed") else 0.0 for k in keys]
    fig, ax = plt.subplots(figsize=(8, max(3, len(keys) * 0.35)))
    ax.barh(keys[::-1], vals[::-1], color=["#2ecc71" if v > 0.5 else "#e74c3c" for v in vals[::-1]])
    ax.set_xlim(-0.05, 1.15)
    ax.set_title("Benchmark checklist (1 = pass)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def plot_confusion_ml(ml_models: Dict[str, Any], out_dir: Path) -> List[Path]:
    out_paths: List[Path] = []
    for name, payload in (ml_models or {}).items():
        met = (payload or {}).get("metrics") or {}
        cm = met.get("confusion_matrix")
        labels = met.get("labels")
        if not cm or not labels:
            continue
        arr = np.asarray(cm, dtype=float)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            arr,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title(f"Confusion matrix — {name}")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        p = out_dir / f"confusion_{name.replace(' ', '_')}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        out_paths.append(p)
    return out_paths


def plot_econometric_symbols(summary: Dict[str, Any], out_path: Path) -> bool:
    econ = summary.get("econometric_by_symbol") or {}
    rows = []
    for sym, payload in econ.items():
        m = (payload or {}).get("metrics") or {}
        rows.append({"symbol": sym, "accuracy": _safe_float(m.get("accuracy"))})
    if not rows:
        return False
    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df["symbol"].astype(str), df["accuracy"].fillna(0))
    ax.set_ylim(0, 1.05)
    ax.set_title("Econometric (joint) — accuracy by symbol")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def plot_symbol_performance(symbol_performance: pd.DataFrame, out_path: Path) -> bool:
    if symbol_performance is None or symbol_performance.empty:
        return False
    col = None
    for c in ("return_5y_pct", "total_return_pct", "return_pct"):
        if c in symbol_performance.columns:
            col = c
            break
    if col is None:
        return False
    df = symbol_performance.copy()
    if "symbol" not in df.columns:
        return False
    df = df.sort_values(col, ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df["symbol"].astype(str), df[col].astype(float))
    ax.set_title(f"Top symbols by {col}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def plot_rolling_forecast(summary: Dict[str, Any], out_path: Path) -> bool:
    block = summary.get("rolling_forecast_eval") or {}
    if block.get("skipped"):
        return False
    rows = []
    for block_name in ("pooled", "per_symbol"):
        sub = block.get(block_name) or {}
        if not sub.get("ok"):
            continue
        rows.append(
            {
                "mode": block_name,
                "ml_acc": _safe_float(sub.get("ml_accuracy")),
                "ml_f1": _safe_float(sub.get("ml_f1_weighted")),
                "econ_f1": _safe_float(sub.get("arima_f1_weighted")),
                "ensemble_f1": _safe_float(sub.get("ensemble_f1_weighted")),
            }
        )
    if not rows:
        return False
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df["ml_f1"].fillna(0), width=w, label="ML F1w")
    if df["econ_f1"].notna().any():
        ax.bar(x, df["econ_f1"].fillna(0), width=w, label="econ F1w")
    if df["ensemble_f1"].notna().any():
        ax.bar(x + w, df["ensemble_f1"].fillna(0), width=w, label="ensemble F1w")
    ax.set_xticks(x)
    ax.set_xticklabels(df["mode"])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Rolling-origin evaluation (trend classification)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def patch_lstm_unpickle_allow_missing_tf() -> None:
    """
    Bundle có thể chứa LSTM pickle; nếu không có TensorFlow, bản gốc raise ImportError.
    Cho phép load bundle chỉ để đọc benchmark_summary / vẽ chart (model LSTM không khôi phục).
    """
    import stock_analysis_yfinance as syf

    def _report_setstate(self: Any, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.model = None
        model_json = state.get("model_json")
        model_weights = state.get("model_weights")
        if model_json is None or model_weights is None:
            return
        tf = syf._get_tensorflow()
        if tf is None:
            return
        self.model = tf.keras.models.model_from_json(model_json)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model.set_weights(model_weights)

    syf.LSTMClassifierWrapper.__setstate__ = _report_setstate  # type: ignore[assignment]


def generate_all_charts(model: Any, out_dir: Path | str) -> Dict[str, Any]:
    """
    model: instance đã gọi load_model_bundle() (StockAnalysisYFinance).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = getattr(model, "benchmark_summary", None) or {}
    report: Dict[str, Any] = {"out_dir": str(out.resolve()), "figures": []}

    p = out / "ranking_metrics.png"
    if plot_ranking_bar(summary, p):
        report["figures"].append(str(p))

    p = out / "checklist_pass.png"
    if plot_checklist(summary, p):
        report["figures"].append(str(p))

    ml_models = summary.get("ml_models") or {}
    for fp in plot_confusion_ml(ml_models, out):
        report["figures"].append(str(fp))

    p = out / "econometric_by_symbol.png"
    if plot_econometric_symbols(summary, p):
        report["figures"].append(str(p))

    sp = getattr(model, "symbol_performance", None)
    if isinstance(sp, pd.DataFrame):
        p = out / "symbol_performance.png"
        if plot_symbol_performance(sp, p):
            report["figures"].append(str(p))

    p = out / "rolling_forecast_eval.png"
    if plot_rolling_forecast(summary, p):
        report["figures"].append(str(p))

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("artifacts/colab_charts"))
    args = parser.parse_args()

    root = args.root.resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)
    from stock_analysis_yfinance import StockAnalysisYFinance

    patch_lstm_unpickle_allow_missing_tf()
    m = StockAnalysisYFinance()
    if not m.load_model_bundle():
        raise SystemExit(f"Không đọc được bundle tại {root / StockAnalysisYFinance.BUNDLE_PATH}")
    rep = generate_all_charts(m, args.out)
    n = len(rep.get("figures", []))
    outd = rep.get("out_dir", "")
    print(f"Saved {n} figures to {outd}")


if __name__ == "__main__":
    main()
