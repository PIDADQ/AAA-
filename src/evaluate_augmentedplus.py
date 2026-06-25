"""
增强测试集鲁棒性评估脚本
==========================
使用训练好的 D-STAR 模型分别测试五个增强版本的测试集，
输出每个版本的 Accuracy / Precision / Recall / F1 / ROC-AUC，
并生成横向对比表格与混淆矩阵图。

使用方法:
    # 基本用法（需要先训练好模型，或指定已有 checkpoint）
    python evaluate_augmented.py

    # 指定模型 checkpoint 路径
    python evaluate_augmented.py --model_path outputs/best_model.pt

    # 不使用知识图谱预处理（加速）
    python evaluate_augmented.py --no_kg

    # 仅测试部分数据集
    python evaluate_augmented.py --datasets original combined

依赖:
    pip install torch transformers scikit-learn pandas matplotlib seaborn
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer

# ── 动态路径解析（兼容 云端/本地 两种目录结构）─────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in [str(SCRIPT_DIR), str(SCRIPT_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from dstar_model import create_dstar_model, DSTARModel
    from dataset import FraudCallDataset, clean_text
except ModuleNotFoundError:
    _CLOUD_MODULES = str(SCRIPT_DIR / "code1_model_raw")
    sys.path.insert(0, _CLOUD_MODULES)
    from dstar_model import create_dstar_model, DSTARModel
    from dataset import FraudCallDataset, clean_text

# ── 字体设置：使用通用英文字体 ────────────────────────────────────────
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

# ═══════════════════════════════════════════════════════════════
# 1. 配置
# ═══════════════════════════════════════════════════════════════

MODEL_CONFIG = {
    "model_name":         "distilbert-base-multilingual-cased",
    "num_encoder_layers": 2,
    "num_heads":          8,
    "hidden_dim":         768,
    "ffn_dim":            3072,
    "alpha":              0.5,
    "lambda_sparse":      0.01,
    "dropout_rate":       0.1,
    "num_labels":         2,
    "freeze_distilbert":  False,
    "batch_size":         32,
    "max_seq_len":        512,
}

AUGMENTED_FILES = {
    "original":    "增强测试集_original.csv",
    "credibility": "增强测试集_credibility.csv",
    "urgency":     "增强测试集_urgency.csv",
    "emotional":   "增强测试集_emotional.csv",
    "combined":    "增强测试集_combined.csv",
}

# 绘图用英文标签
STRATEGY_LABELS = {
    "original":    "Original",
    "credibility": "Credibility",
    "urgency":     "Urgency",
    "emotional":   "Emotional",
    "combined":    "Combined",
}

# ═══════════════════════════════════════════════════════════════
# 2. 工具函数
# ═══════════════════════════════════════════════════════════════

def load_model(model_path: str, config: dict, device: torch.device) -> DSTARModel:
    """加载训练好的 D-STAR checkpoint"""
    model = create_dstar_model(
        pretrained_model     = config["model_name"],
        num_labels           = config["num_labels"],
        freeze_distilbert    = config["freeze_distilbert"],
        num_encoder_layers   = config["num_encoder_layers"],
        num_heads            = config["num_heads"],
        hidden_dim           = config["hidden_dim"],
        ffn_dim              = config["ffn_dim"],
        alpha                = config["alpha"],
        dropout_rate         = config["dropout_rate"],
        lambda_sparse        = config["lambda_sparse"],
    )
    try:
        import numpy as np
        import torch.serialization as _ts
        with _ts.safe_globals([np.core.multiarray.scalar, np.ndarray, np.dtype]):
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"  [√] Model loaded: {model_path}")
    return model


def preprocess_df(df: pd.DataFrame, use_kg: bool = False, kg_mode: str = "append") -> pd.DataFrame:
    """对 DataFrame 的 specific_dialogue_content 列做预处理"""
    df = df.copy()
    df = df.dropna(subset=["is_fraud"]).reset_index(drop=True)
    df["is_fraud"] = df["is_fraud"].astype(int)

    if use_kg:
        from knowledge_graph import KnowledgeGraphProcessor
        kg = KnowledgeGraphProcessor()
        df["processed_text"] = [kg.enrich_text_with_kg(t, kg_mode) for t in df["specific_dialogue_content"]]
    else:
        df["processed_text"] = df["specific_dialogue_content"].apply(clean_text)

    df["processed_text"] = df["processed_text"].replace("", "[EMPTY]")
    return df


@torch.no_grad()
def evaluate_loader(model: DSTARModel, loader: DataLoader, device: torch.device) -> dict:
    """对一个 DataLoader 跑推理，返回完整指标"""
    all_labels, all_preds, all_probs = [], [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs["logits"]
        probs   = torch.softmax(logits, dim=-1).cpu().numpy()
        preds   = np.argmax(probs, axis=-1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_probs = np.array(all_probs)

    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="binary", zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_probs[:, 1])
    else:
        metrics["roc_auc"] = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred

    return metrics

# ═══════════════════════════════════════════════════════════════
# 3. 可视化（全英文）
# ═══════════════════════════════════════════════════════════════

def plot_metrics_bar(results: dict, output_path: str):
    """绘制各策略在四个指标上的柱状图对比（全英文）"""
    keys    = list(results.keys())
    labels  = [STRATEGY_LABELS.get(k, k) for k in keys]
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    titles  = ["Accuracy", "Precision", "Recall", "F1-Score"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Augmented Test Set Robustness Evaluation\nD-STAR Model Performance Comparison", fontsize=14, fontweight="bold")

    for ax, metric, color, title in zip(axes.flatten(), metrics, colors, titles):
        values = [results[k][metric] for k in keys]
        bars   = ax.bar(labels, values, color=color, alpha=0.85, edgecolor="white", width=0.55)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5)
        ref = results.get("original", {}).get(metric, None)
        if ref is not None:
            ax.axhline(ref, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [√] Metric comparison chart saved: {output_path}")


def plot_confusion_matrices(results: dict, output_path: str):
    """绘制各策略的混淆矩阵（全英文）"""
    keys   = list(results.keys())
    n      = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    fig.suptitle("Confusion Matrices Comparison (D-STAR)", fontsize=13, fontweight="bold")

    for ax, key in zip(axes, keys):
        cm = np.array(results[key]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Fraud"],
                    yticklabels=["Normal", "Fraud"],
                    linewidths=0.5, linecolor="white", cbar=False)
        ax.set_title(STRATEGY_LABELS.get(key, key), fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [√] Confusion matrices saved: {output_path}")


def plot_robustness_drop(results: dict, output_path: str):
    """绘制各策略相对于原始集的指标下降折线图（全英文）"""
    if "original" not in results:
        return

    ref    = results["original"]
    keys   = [k for k in results if k != "original"]
    labels = [STRATEGY_LABELS.get(k, k) for k in keys]
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    m_names = ["Accuracy", "Precision", "Recall", "F1"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(keys))
    width = 0.2

    for i, (m, c, mn) in enumerate(zip(metrics, colors, m_names)):
        drops = [results[k][m] - ref[m] for k in keys]
        ax.bar(x + i * width, drops, width, label=mn, color=c, alpha=0.85, edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Δ from Original (↑Improvement ↓Degradation)")
    ax.set_title("Performance Impact of Augmentation Strategies", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [√] Robustness drop chart saved: {output_path}")

# ═══════════════════════════════════════════════════════════════
# 4. 主流程
# ═══════════════════════════════════════════════════════════════

def run_evaluation(args):
    print("\n" + "=" * 65)
    print("  Augmented Test Set Robustness Evaluation — D-STAR")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")

    model_path = args.model_path or str(SCRIPT_DIR / "outputs" / "best_model.pt")
    if not os.path.exists(model_path):
        print(f"\n[Error] Model not found: {model_path}")
        sys.exit(1)

    print(f"\n[Step 1] Loading model: {model_path}")
    model = load_model(model_path, MODEL_CONFIG, device)

    print(f"\n[Step 2] Loading tokenizer: {MODEL_CONFIG['model_name']}")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_CONFIG["model_name"])
    except Exception:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_CONFIG["model_name"], local_files_only=True)

    output_dir = SCRIPT_DIR / "augmented_eval_results"
    output_dir.mkdir(exist_ok=True)

    target_keys = args.datasets or list(AUGMENTED_FILES.keys())
    results = {}
    report_rows = []

    print(f"\n[Step 3] Start evaluation ({len(target_keys)} datasets)")
    print("-" * 65)

    for key in target_keys:
        if key not in AUGMENTED_FILES:
            continue
        csv_file = args.data_dir / AUGMENTED_FILES[key]
        if not csv_file.exists():
            print(f"  [!] Missing: {csv_file}")
            continue

        print(f"\n  ▶ [{STRATEGY_LABELS[key]}] {csv_file.name}")
        df = pd.read_csv(str(csv_file), encoding="utf-8-sig")
        df = preprocess_df(df, use_kg=args.use_kg)
        print(f"    Samples: {len(df)} | Fraud ratio: {df['is_fraud'].mean():.1%}")

        dataset = FraudCallDataset(
            texts=df["processed_text"].tolist(),
            labels=df["is_fraud"].tolist(),
            tokenizer=tokenizer,
            max_length=MODEL_CONFIG["max_seq_len"]
        )
        loader = DataLoader(dataset, batch_size=MODEL_CONFIG["batch_size"], shuffle=False)
        metrics = evaluate_loader(model, loader, device)
        results[key] = metrics

        # ========== 新增：保存预测错误样本 ==========
        y_true = metrics["y_true"]
        y_pred = metrics["y_pred"]
        # 复制原始数据，追加预测标签
        df_error = df.copy()
        df_error["pred_label"] = y_pred
        df_error["true_label"] = y_true
        # 筛选预测错误的行
        df_wrong = df_error[df_error["true_label"] != df_error["pred_label"]].reset_index(drop=True)

        # 保存错分样本csv
        error_save_path = output_dir / f"wrong_predictions_{key}.csv"
        df_wrong.to_csv(str(error_save_path), index=False, encoding="utf-8-sig")
        print(f"    [Saved error cases] Wrong samples: {len(df_wrong)} → {error_save_path.name}")
        # ============================================

        print(f"    Accuracy : {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall   : {metrics['recall']:.4f}")
        print(f"    F1-Score : {metrics['f1']:.4f}")
        print(f"    ROC-AUC  : {metrics['roc_auc']:.4f}")

        report_rows.append({
            "Dataset": STRATEGY_LABELS[key],
            "Strategy": key,
            "Samples": len(df),
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1-Score": round(metrics["f1"], 4),
            "ROC-AUC": round(metrics["roc_auc"], 4) if not np.isnan(metrics["roc_auc"]) else "N/A"
        })

    print("\n" + "=" * 65)
    print("  Summary Table")
    print("=" * 65)
    summary_df = pd.DataFrame(report_rows)
    print(summary_df.to_string(index=False))

    summary_csv = output_dir / "summary_metrics.csv"
    summary_df.to_csv(str(summary_csv), index=False, encoding="utf-8-sig")
    print(f"\n  [√] Summary saved: {summary_csv}")

    if len(results) >= 2:
        print("\n[Step 4] Generating charts...")
        plot_metrics_bar(results, str(output_dir / "metrics_comparison.png"))
        plot_confusion_matrices(results, str(output_dir / "confusion_matrices.png"))
        if "original" in results:
            plot_robustness_drop(results, str(output_dir / "robustness_drop.png"))

    print("\n" + "=" * 65)
    print(f"  All done! Results in: {output_dir}")
    print("=" * 65)

# ═══════════════════════════════════════════════════════════════
# 5. 命令行入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate robustness of augmented fraud test sets with D-STAR")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", choices=list(AUGMENTED_FILES.keys()), default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--use_kg", action="store_true")
    parser.add_argument("--batch_size", type=int, default=MODEL_CONFIG["batch_size"])
    args = parser.parse_args()
    MODEL_CONFIG["batch_size"] = args.batch_size

    if args.data_dir:
        args.data_dir = Path(args.data_dir).resolve()
    else:
        for _d in [SCRIPT_DIR, SCRIPT_DIR / "code1_model_raw", SCRIPT_DIR.parent]:
            if any((_d / f).exists() for f in AUGMENTED_FILES.values()):
                args.data_dir = _d
                break
        else:
            args.data_dir = SCRIPT_DIR
    run_evaluation(args)