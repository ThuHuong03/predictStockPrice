"""Quick ablation runner for core vs hybrid feature sets."""

from __future__ import annotations

import json

from stock_analysis_yfinance import StockAnalysisYFinance


def main() -> None:
    model = StockAnalysisYFinance()
    ok = model.run_full_pipeline(period="5y", refresh_cache=False)
    print(f"PIPELINE_OK={ok}")
    if not ok:
        return

    summary = model.get_benchmark_summary()
    ml_models = summary.get("ml_models", {})
    report = {
        "top_model": summary.get("top_model", {}),
        "best_ml_model": max(ml_models, key=lambda n: ml_models[n]["metrics"]["f1_weighted"]) if ml_models else None,
        "models": {},
    }

    for name, payload in ml_models.items():
        comparison = payload.get("feature_set_comparison", {})
        report["models"][name] = {
            "selected_feature_set": payload.get("selected_feature_set"),
            "metrics": payload.get("metrics"),
            "feature_set_comparison": {
                k: {
                    "acc": v.get("metrics", {}).get("accuracy"),
                    "f1": v.get("metrics", {}).get("f1_weighted"),
                    "roc_auc": v.get("metrics", {}).get("roc_auc_ovr_weighted"),
                    "feature_count": v.get("feature_count"),
                }
                for k, v in comparison.items()
            },
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

