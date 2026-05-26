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
matplotlib.use("Agg")           # 无 GUI 环境下使用
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
# 优先：脚本同目录下找模块（云端 /workspace/）
# 其次：父目录下找模块（本地 code1_model_raw/evaluate_augmented.py）
for _p in [str(SCRIPT_DIR), str(SCRIPT_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from dstar_model import create_dstar_model, DSTARModel
    from dataset import FraudCallDataset, clean_text
except ModuleNotFoundError:
    # 云端场景：脚本在 /workspace/，模块在 /workspace/code1_model_raw/
    _CLOUD_MODULES = str(SCRIPT_DIR / "code1_model_raw")
    sys.path.insert(0, _CLOUD_MODULES)
    from dstar_model import create_dstar_model, DSTARModel
    from dataset import FraudCallDataset, clean_text

# ── 中文字体（Windows 环境）───────────────────────────────────────
def _setup_chinese_font():
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]
    for name in candidates:
        if any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None

FONT_NAME = _setup_chinese_font()


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

# 增强测试集文件名 → 中文策略名
AUGMENTED_FILES = {
    "original":    "增强测试集_original.csv",
    "credibility": "增强测试集_credibility.csv",
    "urgency":     "增强测试集_urgency.csv",
    "emotional":   "增强测试集_emotional.csv",
    "combined":    "增强测试集_combined.csv",
}

STRATEGY_LABELS = {
    "original":    "原始（对照）",
    "credibility": "建立信任",
    "urgency":     "制造紧迫感",
    "emotional":   "情感操纵",
    "combined":    "组合策略",
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
    # PyTorch 2.6 起 weights_only 默认为 True；
    # checkpoint 中含有 numpy scalar 等自定义全局时需显式设为 False（仅信任自己训练的文件）
    try:
        import numpy as np
        import torch.serialization as _ts
        # 方法一：通过 safe_globals 白名单（更安全，推荐）
        with _ts.safe_globals([np.core.multiarray.scalar,
                               np.ndarray,
                               np.dtype]):
            checkpoint = torch.load(model_path, map_location=device,
                                    weights_only=True)
    except Exception:
        # 方法二：回退到 weights_only=False（兼容旧版 PyTorch 或更复杂的 checkpoint）
        checkpoint = torch.load(model_path, map_location=device,
                                weights_only=False)
    # 兼容直接存 state_dict 或包含 'model_state_dict' 的 checkpoint
    state_dict = checkpoint.get("model_state_dict", checkpoint) \
                 if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"  [√] 模型已加载: {model_path}")
    return model


def preprocess_df(df: pd.DataFrame, use_kg: bool = False, kg_mode: str = "append") -> pd.DataFrame:
    """对 DataFrame 的 specific_dialogue_content 列做预处理"""
    df = df.copy()
    df = df.dropna(subset=["is_fraud"]).reset_index(drop=True)
    df["is_fraud"] = df["is_fraud"].astype(int)

    if use_kg:
        from knowledge_graph import KnowledgeGraphProcessor
        kg = KnowledgeGraphProcessor()
        df["processed_text"] = [
            kg.enrich_text_with_kg(t, kg_mode)
            for t in df["specific_dialogue_content"]
        ]
    else:
        df["processed_text"] = df["specific_dialogue_content"].apply(clean_text)

    df["processed_text"] = df["processed_text"].replace("", "[空文本]")
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
# 3. 可视化
# ═══════════════════════════════════════════════════════════════

def plot_metrics_bar(results: dict, output_path: str):
    """绘制各策略在四个指标上的柱状图对比"""
    keys    = list(results.keys())
    labels  = [STRATEGY_LABELS.get(k, k) for k in keys]
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    titles  = ["Accuracy（准确率）", "Precision（精确率）", "Recall（召回率）", "F1-Score"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("增强测试集鲁棒性评估 — D-STAR 模型指标对比", fontsize=14, fontweight="bold")

    for ax, metric, color, title in zip(axes.flatten(), metrics, colors, titles):
        values = [results[k][metric] for k in keys]
        bars   = ax.bar(labels, values, color=color, alpha=0.85, edgecolor="white", width=0.55)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("分值")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        # 标注数值
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.012, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8.5)
        # 原始集参考线
        ref = results.get("original", {}).get(metric, None)
        if ref is not None:
            ax.axhline(ref, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [√] 指标对比图已保存: {output_path}")


def plot_confusion_matrices(results: dict, output_path: str):
    """绘制各策略的混淆矩阵（1×5 布局）"""
    keys   = list(results.keys())
    n      = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    fig.suptitle("增强测试集混淆矩阵对比（D-STAR）", fontsize=13, fontweight="bold")

    for ax, key in zip(axes, keys):
        cm = np.array(results[key]["confusion_matrix"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["非欺诈", "欺诈"],
            yticklabels=["非欺诈", "欺诈"],
            linewidths=0.5, linecolor="white",
            cbar=False,
        )
        ax.set_title(STRATEGY_LABELS.get(key, key), fontsize=10)
        ax.set_xlabel("预测标签")
        ax.set_ylabel("真实标签")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [√] 混淆矩阵图已保存: {output_path}")


def plot_robustness_drop(results: dict, output_path: str):
    """绘制各策略相对于原始集的指标下降折线图"""
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
    ax.set_ylabel("△ 相对原始集（正/负值表示提升/下降）")
    ax.set_title("各增强策略对模型性能的影响（相对原始集变化量）", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [√] 鲁棒性变化图已保存: {output_path}")


# ═══════════════════════════════════════════════════════════════
# 4. 主流程
# ═══════════════════════════════════════════════════════════════

def run_evaluation(args):
    print("\n" + "=" * 65)
    print("  增强测试集鲁棒性评估 — D-STAR 模型")
    print("=" * 65)

    # ── 设备 ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[设备] {device}")
    if torch.cuda.is_available():
        print(f"       GPU: {torch.cuda.get_device_name(0)}")

    # ── 模型路径 ───────────────────────────────────────────────
    model_path = args.model_path
    if model_path is None:
        model_path = str(SCRIPT_DIR / "outputs" / "best_model.pt")
    if not os.path.exists(model_path):
        print(f"\n[错误] 找不到模型文件: {model_path}")
        print("       请先运行 train_dstar.py 完成训练，或用 --model_path 指定路径。")
        sys.exit(1)

    # ── 加载模型 & Tokenizer ───────────────────────────────────
    print(f"\n[Step 1] 加载模型: {model_path}")
    model = load_model(model_path, MODEL_CONFIG, device)

    print(f"\n[Step 2] 加载 Tokenizer: {MODEL_CONFIG['model_name']}")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_CONFIG["model_name"])
    except Exception:
        tokenizer = DistilBertTokenizer.from_pretrained(
            MODEL_CONFIG["model_name"], local_files_only=True
        )
    print(f"  [√] vocab_size={tokenizer.vocab_size}")

    # ── 输出目录 ───────────────────────────────────────────────
    output_dir = SCRIPT_DIR / "augmented_eval_results"
    output_dir.mkdir(exist_ok=True)

    # ── 确定要评估的数据集 ─────────────────────────────────────
    target_keys = args.datasets if args.datasets else list(AUGMENTED_FILES.keys())

    results     = {}
    report_rows = []

    print(f"\n[Step 3] 开始逐一评估（共 {len(target_keys)} 个数据集）")
    print("-" * 65)

    for key in target_keys:
        if key not in AUGMENTED_FILES:
            print(f"  [!] 未知数据集 key: {key}，跳过")
            continue

        csv_file = args.data_dir / AUGMENTED_FILES[key]
        if not csv_file.exists():
            print(f"  [!] 文件不存在: {csv_file}，跳过")
            continue

        print(f"\n  ▶ [{STRATEGY_LABELS.get(key, key)}]  {csv_file.name}")

        # 加载 & 预处理
        df = pd.read_csv(str(csv_file), encoding="utf-8-sig")
        df = preprocess_df(df, use_kg=args.use_kg, kg_mode="append")
        print(f"    样本数: {len(df)}  |  欺诈比例: {df['is_fraud'].mean():.1%}")

        # DataLoader
        dataset = FraudCallDataset(
            texts      = df["processed_text"].tolist(),
            labels     = df["is_fraud"].tolist(),
            tokenizer  = tokenizer,
            max_length = MODEL_CONFIG["max_seq_len"],
        )
        loader = DataLoader(
            dataset,
            batch_size  = MODEL_CONFIG["batch_size"],
            shuffle     = False,
            num_workers = 0,
            pin_memory  = torch.cuda.is_available(),
        )

        # 推理
        metrics = evaluate_loader(model, loader, device)
        results[key] = metrics

        # 打印
        print(f"    Accuracy : {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall   : {metrics['recall']:.4f}")
        print(f"    F1-Score : {metrics['f1']:.4f}")
        print(f"    ROC-AUC  : {metrics['roc_auc']:.4f}")

        # 详细分类报告
        print("\n    分类报告:")
        cr = classification_report(
            metrics["y_true"], metrics["y_pred"],
            target_names=["非欺诈", "欺诈"], digits=4
        )
        for line in cr.strip().split("\n"):
            print(f"      {line}")

        report_rows.append({
            "数据集":    STRATEGY_LABELS.get(key, key),
            "增强策略":  key,
            "样本数":    len(df),
            "Accuracy":  round(metrics["accuracy"],  4),
            "Precision": round(metrics["precision"], 4),
            "Recall":    round(metrics["recall"],    4),
            "F1-Score":  round(metrics["f1"],        4),
            "ROC-AUC":   round(metrics["roc_auc"],   4) if not np.isnan(metrics["roc_auc"]) else "N/A",
        })

    # ── 汇总表格 ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  汇总对比表")
    print("=" * 65)
    summary_df = pd.DataFrame(report_rows)
    print(summary_df.to_string(index=False))

    # 保存 CSV
    summary_csv = output_dir / "summary_metrics.csv"
    summary_df.to_csv(str(summary_csv), index=False, encoding="utf-8-sig")
    print(f"\n  [√] 汇总表已保存: {summary_csv}")

    # 保存 JSON
    json_results = {
        k: {m: (float(v) if isinstance(v, (float, np.floating)) else v)
            for m, v in metrics.items()
            if m not in ("y_true", "y_pred")}
        for k, metrics in results.items()
    }
    json_path = output_dir / "eval_results.json"
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    print(f"  [√] 详细结果已保存: {json_path}")

    # ── 可视化 ─────────────────────────────────────────────────
    if len(results) >= 2:
        print("\n[Step 4] 生成可视化图表...")
        plot_metrics_bar(
            results,
            str(output_dir / "metrics_comparison.png")
        )
        plot_confusion_matrices(
            results,
            str(output_dir / "confusion_matrices.png")
        )
        if "original" in results:
            plot_robustness_drop(
                results,
                str(output_dir / "robustness_drop.png")
            )
    else:
        print("\n[注] 只评估了 1 个数据集，跳过对比可视化")

    print("\n" + "=" * 65)
    print(f"  全部完成！结果保存至: {output_dir}")
    print("=" * 65)


# ═══════════════════════════════════════════════════════════════
# 5. 命令行入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="用训练好的 D-STAR 模型评估五个增强版测试集的鲁棒性"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="训练好的模型 checkpoint 路径（默认: outputs/best_model.pt）"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(AUGMENTED_FILES.keys()),
        default=None,
        help="指定要评估的数据集（默认全部）。可选: original credibility urgency emotional combined"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="增强测试集 CSV 所在目录（默认: 脚本所在目录）"
    )
    parser.add_argument(
        "--use_kg", action="store_true",
        help="使用知识图谱预处理（与训练时保持一致时加上此参数；会显著增加耗时）"
    )
    parser.add_argument(
        "--batch_size", type=int, default=MODEL_CONFIG["batch_size"],
        help=f"推理 batch size（默认 {MODEL_CONFIG['batch_size']}）"
    )

    args = parser.parse_args()
    MODEL_CONFIG["batch_size"] = args.batch_size
    # 数据目录：CLI 指定 > 脚本同目录 > 父目录
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
