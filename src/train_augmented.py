"""
基于组合策略增强训练集的 D-STAR 继续训练脚本
==============================================
在已有 best_model.pt 的基础上，用"组合增强训练集"做微调：

数据流:
  原始训练集 (../训练集结果.csv)
    └── 组合增强 (augment_dialogue_combined) → 增强训练集
  原始验证集 (从训练集 20% 划分，不增强，与原始训练一致)
  原始测试集 (../测试集结果.csv，不增强，公平评估)

训练策略:
  - 从 best_model.pt 热启动（load & continue）
  - 较小学习率 (5e-6) 防止遗忘
  - 层次化 lr：DistilBERT 层更小，D-STAR 头部更大
  - 更短的 epoch（默认 10）+ Early stopping

使用方法:
  # 基本用法（自动找 best_model.pt，无 KG）
  python train_augmented.py --no_kg

  # 完整用法
  python train_augmented.py \
      --pretrain_path ./best_model.pt \
      --epochs 10 \
      --lr 5e-6 \
      --batch_size 16 \
      --no_kg

  # 使用知识图谱预处理（与原始训练保持一致，但较慢）
  python train_augmented.py --pretrain_path ./best_model.pt

输出:
  outputs_augmented/
    ├── best_model_augmented.pt   最优 checkpoint
    ├── results_augmented.json    完整指标 JSON
    └── training_log.csv          每轮 loss/metric 记录
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple

from torch.optim import AdamW
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

# ── 将当前目录加入 sys.path（支持云端 /workspace/ 与本地两种结构）─────
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in [str(SCRIPT_DIR), str(SCRIPT_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dstar_model import create_dstar_model, DSTARModel
from dataset import FraudCallDataset, clean_text

# augment_dataset.py 在父目录（本地）或同目录（云端复制时）
try:
    from augment_dataset import augment_dialogue_combined
except ModuleNotFoundError:
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    from augment_dataset import augment_dialogue_combined


# ═══════════════════════════════════════════════════════════════
# 1. 默认配置
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    # 模型参数（与原始训练保持一致）
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

    # 微调训练参数
    "batch_size":         16,
    "max_epochs":         10,
    "learning_rate":      5e-6,      # 远小于原始 2e-5，防止遗忘
    "weight_decay":       0.01,
    "warmup_ratio":       0.1,
    "max_seq_len":        512,
    "val_ratio":          0.2,       # 与原始训练保持一致
    "random_seed":        42,

    # Early stopping
    "patience":           4,
    "min_delta":          0.001,

    # 数据
    "use_knowledge_graph": False,    # 默认关闭，加快速度；与原训练一致则打开
    "kg_mode":            "append",

    # 输出
    "output_dir":         "outputs_augmented",
    "log_interval":       10,

    # 预训练模型路径（相对于脚本目录）
    "pretrain_path":      "best_model.pt",
}


# ═══════════════════════════════════════════════════════════════
# 2. 工具函数
# ═══════════════════════════════════════════════════════════════

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience: int = 4, min_delta: float = 0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = None

    def __call__(self, value: float) -> bool:
        if self.best is None or value > self.best + self.min_delta:
            self.best    = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def compute_metrics(labels: np.ndarray, preds: np.ndarray,
                    probs: np.ndarray = None) -> Dict:
    m = {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall":    recall_score(labels, preds, average="binary", zero_division=0),
        "f1":        f1_score(labels, preds, average="binary", zero_division=0),
    }
    cm = confusion_matrix(labels, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        m["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        m["tp"], m["tn"], m["fp"], m["fn"] = int(tp), int(tn), int(fp), int(fn)
    if probs is not None and len(np.unique(labels)) > 1:
        try:
            m["roc_auc"] = roc_auc_score(labels, probs[:, 1])
        except Exception:
            m["roc_auc"] = 0.0
    return m


def safe_load_checkpoint(path: str, device: torch.device) -> dict:
    """兼容 PyTorch 2.6 weights_only 变更"""
    try:
        import torch.serialization as _ts
        with _ts.safe_globals([np.core.multiarray.scalar, np.ndarray, np.dtype]):
            return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


# ═══════════════════════════════════════════════════════════════
# 3. 数据加载（增强训练集 + 原始验证/测试集）
# ═══════════════════════════════════════════════════════════════

def _find_csv(script_dir: Path, filename: str) -> Path:
    """从脚本目录向上最多两层查找 CSV 文件"""
    for base in [script_dir, script_dir.parent, script_dir.parent.parent]:
        p = base / filename
        if p.exists():
            return p
    raise FileNotFoundError(
        f"找不到 {filename}，搜索路径：{script_dir}、{script_dir.parent}、{script_dir.parent.parent}"
    )


def load_augmented_data(
    config: Dict,
    script_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载并构建增强训练集：
      - 原始训练集 → 组合增强 → 增强训练集（含原始 + 增强，约 2× 数据量）
      - 验证集来自增强训练集的 20%（不增强，只做 clean_text）
      - 测试集原始不动
    """
    train_csv = _find_csv(script_dir, "训练集结果.csv")
    test_csv  = _find_csv(script_dir, "测试集结果.csv")

    print(f"  训练集路径: {train_csv}")
    print(f"  测试集路径: {test_csv}")

    # ── 加载 ──────────────────────────────────────────────────
    try:
        raw_train = pd.read_csv(str(train_csv), encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw_train = pd.read_csv(str(train_csv), encoding="utf-8")

    try:
        raw_test = pd.read_csv(str(test_csv), encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw_test = pd.read_csv(str(test_csv), encoding="utf-8")

    raw_train = raw_train.dropna(subset=["is_fraud"]).reset_index(drop=True)
    raw_test  = raw_test.dropna(subset=["is_fraud"]).reset_index(drop=True)
    raw_train["is_fraud"] = raw_train["is_fraud"].astype(int)
    raw_test["is_fraud"]  = raw_test["is_fraud"].astype(int)

    print(f"  原始训练集: {len(raw_train)} 条")
    print(f"  原始测试集: {len(raw_test)} 条")

    # ── 从原始训练集划分验证集（与原始训练保持一致）─────────────
    train_base, val_base = train_test_split(
        raw_train,
        test_size=config["val_ratio"],
        random_state=config["random_seed"],
        stratify=raw_train["is_fraud"],
    )
    train_base = train_base.reset_index(drop=True)
    val_base   = val_base.reset_index(drop=True)

    # ── 知识图谱 / 文本清洗 ───────────────────────────────────
    if config["use_knowledge_graph"]:
        from knowledge_graph import KnowledgeGraphProcessor
        kg = KnowledgeGraphProcessor()
        preprocess = lambda t: kg.enrich_text_with_kg(t, config["kg_mode"])
    else:
        preprocess = clean_text

    # ── 组合增强训练集 ─────────────────────────────────────────
    print("\n  正在生成组合增强训练集（applied: credibility + urgency + emotional）...")
    aug_train = train_base.copy()
    aug_train["specific_dialogue_content"] = aug_train[
        "specific_dialogue_content"
    ].apply(augment_dialogue_combined)
    aug_train["augmentation_strategy"] = "Combined_Strategy"

    # 合并：原始 + 增强（翻倍，标签不变）
    combined_train = pd.concat(
        [train_base, aug_train], ignore_index=True
    ).sample(frac=1, random_state=config["random_seed"]).reset_index(drop=True)

    print(f"  增强后训练集: {len(combined_train)} 条（原始 {len(train_base)} + 增强 {len(aug_train)}）")

    # ── 预处理文本 ─────────────────────────────────────────────
    print("  正在预处理文本...")

    combined_train["processed_text"] = combined_train[
        "specific_dialogue_content"
    ].apply(preprocess)
    val_base["processed_text"] = val_base[
        "specific_dialogue_content"
    ].apply(preprocess)
    raw_test["processed_text"] = raw_test[
        "specific_dialogue_content"
    ].apply(preprocess)

    # 空文本补位
    for df in [combined_train, val_base, raw_test]:
        df["processed_text"] = df["processed_text"].replace("", "[空文本]")

    # ── 标签分布 ──────────────────────────────────────────────
    for name, df in [("训练集（增强后）", combined_train),
                     ("验证集",         val_base),
                     ("测试集",         raw_test)]:
        fc = df["is_fraud"].sum()
        print(f"  {name}: {len(df)} 条  欺诈 {fc}({fc/len(df):.1%})")

    return combined_train, val_base, raw_test


# ═══════════════════════════════════════════════════════════════
# 4. 训练 & 评估
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler, device, log_interval=10) -> Dict:
    model.train()
    total_loss = ce_total = sparse_total = 0.0
    all_preds, all_labels = [], []
    t0 = time.time()

    for i, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        out    = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss   = out["loss"]
        ce_l   = out.get("ce_loss", loss)
        sp_l   = out.get("sparsity_loss", torch.tensor(0.0))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss   += loss.item()
        ce_total     += ce_l.item()
        sparse_total += sp_l.item()

        preds = torch.argmax(out["logits"], dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if (i + 1) % log_interval == 0:
            print(f"    Batch [{i+1}/{len(loader)}]  "
                  f"loss={loss.item():.4f}  ce={ce_l.item():.4f}  "
                  f"sparse={sp_l.item():.4f}  t={time.time()-t0:.1f}s")

    m = compute_metrics(np.array(all_labels), np.array(all_preds))
    m["loss"]         = total_loss / len(loader)
    m["ce_loss"]      = ce_total   / len(loader)
    m["sparse_loss"]  = sparse_total / len(loader)
    return m


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        out   = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if "loss" in out:
            total_loss += out["loss"].item()

        probs = torch.softmax(out["logits"], dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_probs = np.array(all_probs)

    m = compute_metrics(y_true, y_pred, y_probs)
    m["loss"] = total_loss / max(len(loader), 1)
    return m, y_true, y_pred, y_probs


# ═══════════════════════════════════════════════════════════════
# 5. 主流程
# ═══════════════════════════════════════════════════════════════

def main(config: Dict):
    set_seed(config["random_seed"])

    # ── 设备 ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[设备] {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── 输出目录 ───────────────────────────────────────────────
    output_dir = SCRIPT_DIR / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  D-STAR 增强微调训练（组合策略）")
    print("=" * 65)

    # ── Step 1: 加载数据 ───────────────────────────────────────
    print("\n[Step 1] 加载数据并生成增强训练集...")
    train_df, val_df, test_df = load_augmented_data(config, SCRIPT_DIR)

    # ── Step 2: Tokenizer ─────────────────────────────────────
    print(f"\n[Step 2] 加载 Tokenizer: {config['model_name']}")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(config["model_name"])
    except Exception:
        tokenizer = DistilBertTokenizer.from_pretrained(
            config["model_name"], local_files_only=True
        )
    print(f"  vocab_size={tokenizer.vocab_size}")

    # ── Step 3: DataLoader ────────────────────────────────────
    from torch.utils.data import DataLoader

    def _make_loader(df, shuffle):
        ds = FraudCallDataset(
            texts     = df["processed_text"].tolist(),
            labels    = df["is_fraud"].tolist(),
            tokenizer = tokenizer,
            max_length= config["max_seq_len"],
        )
        return DataLoader(
            ds,
            batch_size  = config["batch_size"],
            shuffle     = shuffle,
            num_workers = 0,
            pin_memory  = torch.cuda.is_available(),
        )

    train_loader = _make_loader(train_df, shuffle=True)
    val_loader   = _make_loader(val_df,   shuffle=False)
    test_loader  = _make_loader(test_df,  shuffle=False)
    print(f"\n[Step 3] DataLoader 完成")
    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")
    print(f"  测试批次: {len(test_loader)}")

    # ── Step 4: 构建模型 ──────────────────────────────────────
    print(f"\n[Step 4] 初始化 D-STAR 模型结构...")
    model = create_dstar_model(
        pretrained_model   = config["model_name"],
        num_labels         = config["num_labels"],
        freeze_distilbert  = config["freeze_distilbert"],
        num_encoder_layers = config["num_encoder_layers"],
        num_heads          = config["num_heads"],
        hidden_dim         = config["hidden_dim"],
        ffn_dim            = config["ffn_dim"],
        alpha              = config["alpha"],
        dropout_rate       = config["dropout_rate"],
        lambda_sparse      = config["lambda_sparse"],
    )

    # ── Step 5: 加载预训练权重 ────────────────────────────────
    pretrain_path = Path(config["pretrain_path"])
    if not pretrain_path.is_absolute():
        pretrain_path = SCRIPT_DIR / pretrain_path

    if pretrain_path.exists():
        print(f"\n[Step 5] 从 checkpoint 热启动: {pretrain_path}")
        ckpt = safe_load_checkpoint(str(pretrain_path), device)
        state_dict = ckpt.get("model_state_dict", ckpt) \
                     if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  缺少的 key（共 {len(missing)} 个）: {missing[:5]}...")
        if unexpected:
            print(f"  多余的 key（共 {len(unexpected)} 个）: {unexpected[:5]}...")
        print(f"  [√] 权重加载完成")

        # 打印原始模型在验证集的基线指标
        model.to(device)
        print("  计算原始模型基线（验证集）...")
        base_val, _, _, _ = evaluate(model, val_loader, device)
        print(f"  基线 → Acc={base_val['accuracy']:.4f}  "
              f"P={base_val['precision']:.4f}  "
              f"R={base_val['recall']:.4f}  "
              f"F1={base_val['f1']:.4f}")
    else:
        print(f"\n[Step 5] 未找到 checkpoint ({pretrain_path})，从头训练")
        model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}  可训练: {trainable_params:,}")

    # ── Step 6: 优化器 ────────────────────────────────────────
    lr  = config["learning_rate"]
    wd  = config["weight_decay"]
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_params = [
        # DistilBERT 层：小 lr，带/不带 weight_decay
        {
            "params": [p for n, p in model.distilbert.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "lr": lr,
            "weight_decay": wd,
        },
        {
            "params": [p for n, p in model.distilbert.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "lr": lr,
            "weight_decay": 0.0,
        },
        # D-STAR 头部：大 lr（5×）
        {
            "params": [p for n, p in model.named_parameters()
                       if "distilbert" not in n and
                       not any(nd in n for nd in no_decay)],
            "lr": lr * 5,
            "weight_decay": wd,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if "distilbert" not in n and
                       any(nd in n for nd in no_decay)],
            "lr": lr * 5,
            "weight_decay": 0.0,
        },
    ]

    optimizer    = AdamW(optimizer_params)
    total_steps  = len(train_loader) * config["max_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    print(f"\n[Step 6] 优化器准备完成")
    print(f"  DistilBERT lr={lr:.1e}  D-STAR head lr={lr*5:.1e}")
    print(f"  总步数={total_steps}  预热步数={warmup_steps}")

    # ── Step 7: 训练循环 ──────────────────────────────────────
    print(f"\n[Step 7] 开始训练（max_epochs={config['max_epochs']}）")
    print("=" * 65)

    early_stop    = EarlyStopping(config["patience"], config["min_delta"])
    best_f1       = 0.0
    best_ckpt     = str(output_dir / "best_model_augmented.pt")
    history       = []

    for epoch in range(1, config["max_epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['max_epochs']}")
        print("-" * 40)

        # 训练
        tr = train_epoch(model, train_loader, optimizer, scheduler,
                         device, config["log_interval"])
        # 验证
        val, _, _, _ = evaluate(model, val_loader, device)

        row = {
            "epoch":       epoch,
            "train_loss":  round(tr["loss"],       4),
            "train_acc":   round(tr["accuracy"],   4),
            "train_f1":    round(tr["f1"],         4),
            "val_loss":    round(val["loss"],      4),
            "val_acc":     round(val["accuracy"],  4),
            "val_prec":    round(val["precision"], 4),
            "val_recall":  round(val["recall"],    4),
            "val_f1":      round(val["f1"],        4),
            "val_roc_auc": round(val.get("roc_auc", 0.0), 4),
        }
        history.append(row)

        print(f"  训练集: loss={tr['loss']:.4f}  acc={tr['accuracy']:.4f}  f1={tr['f1']:.4f}")
        print(f"  验证集: loss={val['loss']:.4f}  acc={val['accuracy']:.4f}  "
              f"P={val['precision']:.4f}  R={val['recall']:.4f}  F1={val['f1']:.4f}")
        if "roc_auc" in val:
            print(f"  验证集: ROC-AUC={val['roc_auc']:.4f}  "
                  f"Specificity={val.get('specificity', 0.0):.4f}")

        # 保存最优
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics":      val,
                    "config":           config,
                },
                best_ckpt,
            )
            print(f"  >> [√] 保存最优模型  val_f1={best_f1:.4f}  → {best_ckpt}")

        # Early stopping
        if early_stop(val["f1"]):
            print(f"\n  Early stopping 触发（连续 {config['patience']} 轮无提升）")
            break

    # 保存训练日志
    log_path = str(output_dir / "training_log.csv")
    pd.DataFrame(history).to_csv(log_path, index=False, encoding="utf-8-sig")
    print(f"\n  [√] 训练日志已保存: {log_path}")

    # ── Step 8: 加载最优模型，测试集评估 ──────────────────────
    print("\n" + "=" * 65)
    print("[Step 8] 加载最优模型，测试集评估...")

    if os.path.exists(best_ckpt):
        ckpt = safe_load_checkpoint(best_ckpt, device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  加载第 {ckpt['epoch']} 轮权重")

    test_m, test_labels, test_preds, test_probs = evaluate(model, test_loader, device)

    print("\n" + "=" * 65)
    print("  测试集最终结果")
    print("=" * 65)
    print(f"  Accuracy :  {test_m['accuracy']:.4f}")
    print(f"  Precision:  {test_m['precision']:.4f}")
    print(f"  Recall   :  {test_m['recall']:.4f}")
    print(f"  F1-Score :  {test_m['f1']:.4f}")
    if "roc_auc" in test_m:
        print(f"  ROC-AUC  :  {test_m['roc_auc']:.4f}")
    if "specificity" in test_m:
        print(f"  Specificity: {test_m['specificity']:.4f}")

    print("\n  分类报告:")
    print(classification_report(test_labels, test_preds,
                                target_names=["非欺诈", "欺诈"]))

    cm = confusion_matrix(test_labels, test_preds)
    print(f"  混淆矩阵:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    # ── Step 9: 保存结果 JSON ─────────────────────────────────
    def _to_native(obj):
        if isinstance(obj, (np.int64, np.int32, np.int_)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_native(x) for x in obj]
        return obj

    results = _to_native({
        "test_metrics":     test_m,
        "training_history": history,
        "config":           config,
        "best_val_f1":      best_f1,
    })

    json_path = str(output_dir / "results_augmented.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  [√] 结果已保存: {json_path}")

    print("\n" + "=" * 65)
    print(f"  训练完成！所有输出位于: {output_dir}")
    print("=" * 65)

    return model, results


# ═══════════════════════════════════════════════════════════════
# 6. 命令行入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="D-STAR 增强微调：在 best_model.pt 基础上用组合策略增强训练集继续训练"
    )
    parser.add_argument(
        "--pretrain_path", type=str, default=CONFIG["pretrain_path"],
        help=f"预训练 checkpoint 路径（相对于脚本目录，默认: {CONFIG['pretrain_path']}）"
    )
    parser.add_argument(
        "--epochs", type=int, default=CONFIG["max_epochs"],
        help=f"训练轮数（默认 {CONFIG['max_epochs']}）"
    )
    parser.add_argument(
        "--lr", type=float, default=CONFIG["learning_rate"],
        help=f"学习率（默认 {CONFIG['learning_rate']}）"
    )
    parser.add_argument(
        "--batch_size", type=int, default=CONFIG["batch_size"],
        help=f"Batch size（默认 {CONFIG['batch_size']}）"
    )
    parser.add_argument(
        "--patience", type=int, default=CONFIG["patience"],
        help=f"Early stopping patience（默认 {CONFIG['patience']}）"
    )
    parser.add_argument(
        "--no_kg", action="store_true",
        help="不使用知识图谱预处理（默认不使用，加快速度）"
    )
    parser.add_argument(
        "--use_kg", action="store_true",
        help="启用知识图谱预处理（与原始训练保持一致时使用）"
    )
    parser.add_argument(
        "--freeze_distilbert", action="store_true",
        help="冻结 DistilBERT 底层，只训练 D-STAR 头部"
    )
    parser.add_argument(
        "--output_dir", type=str, default=CONFIG["output_dir"],
        help=f"输出目录（默认 {CONFIG['output_dir']}）"
    )

    args = parser.parse_args()

    config = CONFIG.copy()
    config["pretrain_path"]      = args.pretrain_path
    config["max_epochs"]         = args.epochs
    config["learning_rate"]      = args.lr
    config["batch_size"]         = args.batch_size
    config["patience"]           = args.patience
    config["use_knowledge_graph"]= args.use_kg and not args.no_kg
    config["freeze_distilbert"]  = args.freeze_distilbert
    config["output_dir"]         = args.output_dir

    print("\n========== 训练配置 ==========")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 30)

    main(config)
