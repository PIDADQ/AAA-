"""
D-STAR 模型训练脚本
论文: Classifying Scam Calls Through Content Analysis With Dynamic Sparsity Top-k Attention Regularization

论文实验配置:
- batch_size = 16
- epochs = 20
- lr = 2e-5 (Adam optimizer)
- lambda_sparse = 0.01 (L1 正则化)
- max_seq_len = 5 (最优实验值)
- 2 encoder layers, 8 attention heads, hidden_dim=768
- 80/20 train-test split
- Early stopping

达到论文指标:
- Accuracy: 94% (94.74% exact)
- Recall: 91.67%
- F1-score: 84.98%
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dstar_model import DSTARModel, create_dstar_model
from dataset import load_and_preprocess_data, create_dataloaders


# ============================================================
# 配置参数
# ============================================================

CONFIG = {
    # 模型参数（论文配置）
    "model_name": "distilbert-base-multilingual-cased",
    "num_encoder_layers": 2,
    "num_heads": 8,
    "hidden_dim": 768,
    "ffn_dim": 3072,
    "alpha": 0.5,           # Top-k 稀疏率（保留 50% tokens）
    "lambda_sparse": 0.01,  # L1 正则化强度
    "dropout_rate": 0.1,
    "num_labels": 2,
    "freeze_distilbert": False,
    
    # 训练参数（论文配置）
    "batch_size": 16,
    "max_epochs": 20,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_seq_len": 512,     # tokenize 时使用 512，模型内部用 alpha 控制稀疏
    "val_ratio": 0.2,
    
    # Early stopping
    "patience": 5,          # 提前停止耐心值
    "min_delta": 0.001,     # 最小改善量
    
    # 数据预处理
    "use_knowledge_graph": True,
    "kg_mode": "append",   # "append" / "filter" / "original"
    "random_seed": 42,
    
    # 输出
    "output_dir": "outputs",
    "save_best_model": True,
    "log_interval": 10,     # 每多少 batch 打印一次
}


# ============================================================
# 训练工具函数
# ============================================================

def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """提前停止（论文使用了 early stopping）"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray = None) -> Dict:
    """计算论文中的所有评估指标"""
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='binary', zero_division=0),
        'recall': recall_score(labels, preds, average='binary', zero_division=0),
        'f1': f1_score(labels, preds, average='binary', zero_division=0),
    }
    
    # 混淆矩阵
    cm = confusion_matrix(labels, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        metrics['tp'] = int(tp)
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
    
    # ROC AUC
    if probs is not None and len(np.unique(labels)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(labels, probs[:, 1])
        except:
            metrics['roc_auc'] = 0.0
    
    return metrics


# ============================================================
# 训练和评估函数
# ============================================================

def train_epoch(
    model: DSTARModel,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    log_interval: int = 10,
) -> Dict:
    """训练一个 epoch"""
    model.train()
    
    total_loss = 0.0
    total_ce_loss = 0.0
    total_sparse_loss = 0.0
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs['loss']
        ce_loss = outputs.get('ce_loss', loss)
        sparse_loss = outputs.get('sparsity_loss', torch.tensor(0.0))
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # 记录
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_sparse_loss += sparse_loss.item()
        
        logits = outputs['logits']
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"  Batch [{batch_idx+1}/{len(loader)}] "
                f"Loss={loss.item():.4f} "
                f"CE={ce_loss.item():.4f} "
                f"Sparse={sparse_loss.item():.4f} "
                f"Time={elapsed:.1f}s"
            )
    
    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics['loss'] = avg_loss
    metrics['ce_loss'] = total_ce_loss / len(loader)
    metrics['sparse_loss'] = total_sparse_loss / len(loader)
    
    return metrics


@torch.no_grad()
def evaluate(
    model: DSTARModel,
    loader,
    device: torch.device,
) -> Dict:
    """评估模型"""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        if 'loss' in outputs:
            total_loss += outputs['loss'].item()
        
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
    
    all_probs = np.array(all_probs)
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        all_probs,
    )
    metrics['loss'] = total_loss / max(len(loader), 1)
    
    return metrics, np.array(all_labels), np.array(all_preds), all_probs


# ============================================================
# 主训练流程
# ============================================================

def train(config: Dict = None):
    """D-STAR 模型完整训练流程"""
    
    if config is None:
        config = CONFIG
    
    # 设置随机种子
    set_seed(config['random_seed'])
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("D-STAR 模型训练")
    print("论文: Classifying Scam Calls Through Content Analysis")
    print("With Dynamic Sparsity Top-k Attention Regularization")
    print("="*60)
    
    # ====== Step 1: 加载数据 ======
    print("\n[Step 1] 加载并预处理数据集...")
    train_df, val_df, test_df = load_and_preprocess_data(
        val_ratio=config['val_ratio'],
        use_knowledge_graph=config['use_knowledge_graph'],
        kg_mode=config['kg_mode'],
        random_seed=config['random_seed'],
    )
    
    # ====== Step 2: 初始化 Tokenizer ======
    print(f"\n[Step 2] 加载 DistilBERT Tokenizer: {config['model_name']}")
    
    # 尝试使用镜像源
    hf_endpoint = os.environ.get('HF_ENDPOINT', '')
    if hf_endpoint:
        print(f"  使用镜像源: {hf_endpoint}")
    
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(config['model_name'])
        print(f"  Tokenizer 加载成功 (vocab_size={tokenizer.vocab_size})")
    except Exception as e:
        print(f"  在线加载失败: {e}")
        print("  尝试使用缓存...")
        tokenizer = DistilBertTokenizer.from_pretrained(
            config['model_name'],
            local_files_only=True
        )
    
    # ====== Step 3: 创建 DataLoader ======
    print(f"\n[Step 3] 创建 DataLoader (batch_size={config['batch_size']})")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_seq_len'],
        num_workers=0,
    )
    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")
    print(f"  测试批次: {len(test_loader)}")
    
    # ====== Step 4: 创建模型 ======
    print(f"\n[Step 4] 初始化 D-STAR 模型...")
    model = create_dstar_model(
        pretrained_model=config['model_name'],
        num_labels=config['num_labels'],
        freeze_distilbert=config['freeze_distilbert'],
        num_encoder_layers=config['num_encoder_layers'],
        num_heads=config['num_heads'],
        hidden_dim=config['hidden_dim'],
        ffn_dim=config['ffn_dim'],
        alpha=config['alpha'],
        dropout_rate=config['dropout_rate'],
        lambda_sparse=config['lambda_sparse'],
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    # ====== Step 5: 优化器和调度器 ======
    print(f"\n[Step 5] 设置优化器 (lr={config['learning_rate']}, Adam)")
    
    # 分层学习率：DistilBERT 用更小的 lr
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.distilbert.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            'lr': config['learning_rate'],
            'weight_decay': config['weight_decay'],
        },
        {
            'params': [p for n, p in model.distilbert.named_parameters()
                      if any(nd in n for nd in no_decay)],
            'lr': config['learning_rate'],
            'weight_decay': 0.0,
        },
        {
            'params': [p for n, p in model.named_parameters()
                      if 'distilbert' not in n and not any(nd in n for nd in no_decay)],
            'lr': config['learning_rate'] * 5,  # D-STAR 层用更大的 lr
            'weight_decay': config['weight_decay'],
        },
        {
            'params': [p for n, p in model.named_parameters()
                      if 'distilbert' not in n and any(nd in n for nd in no_decay)],
            'lr': config['learning_rate'] * 5,
            'weight_decay': 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters)
    
    total_steps = len(train_loader) * config['max_epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"  总训练步数: {total_steps}, 预热步数: {warmup_steps}")
    
    # ====== Step 6: 训练循环 ======
    print(f"\n[Step 6] 开始训练 (max_epochs={config['max_epochs']})")
    print("="*60)
    
    early_stopping = EarlyStopping(
        patience=config['patience'],
        min_delta=config['min_delta'],
        mode='max',
    )
    
    best_val_f1 = 0.0
    best_model_path = os.path.join(output_dir, 'best_model.pt')
    history = []
    
    for epoch in range(1, config['max_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['max_epochs']}")
        print("-"*40)
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, config['log_interval']
        )
        
        # 验证
        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        
        # 记录
        epoch_record = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'train_f1': train_metrics['f1'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_specificity': val_metrics.get('specificity', 0.0),
            'val_roc_auc': val_metrics.get('roc_auc', 0.0),
        }
        history.append(epoch_record)
        
        print(f"  训练集: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f}, F1={train_metrics['f1']:.4f}")
        print(f"  验证集: Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.4f}, "
              f"P={val_metrics['precision']:.4f}, R={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}")
        
        if 'roc_auc' in val_metrics:
            print(f"  验证集: ROC-AUC={val_metrics['roc_auc']:.4f}, Specificity={val_metrics.get('specificity', 0.0):.4f}")
        
        # 保存最优模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            if config['save_best_model']:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'config': config,
                }, best_model_path)
                print(f"  >> 保存最优模型 (val_f1={best_val_f1:.4f})")
        
        # Early Stopping
        if early_stopping(val_metrics['f1']):
            print(f"\n  Early stopping 触发 (patience={config['patience']})")
            break
    
    # ====== Step 7: 加载最优模型并测试 ======
    print("\n" + "="*60)
    print("[Step 7] 加载最优模型，在测试集上评估...")
    
    if config['save_best_model'] and os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  加载第 {checkpoint['epoch']} 轮最优模型")
    
    test_metrics, test_labels, test_preds, test_probs = evaluate(model, test_loader, device)
    
    print("\n" + "="*60)
    print("测试集最终结果（对比论文指标）")
    print("="*60)
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}  (论文: 0.9400)")
    print(f"  Recall:      {test_metrics['recall']:.4f}  (论文: 0.9167)")
    print(f"  F1-Score:    {test_metrics['f1']:.4f}  (论文: 0.8498)")
    print(f"  Precision:   {test_metrics['precision']:.4f}  (论文: 0.7920)")
    if 'specificity' in test_metrics:
        print(f"  Specificity: {test_metrics['specificity']:.4f}  (论文: 0.9534)")
    if 'npv' in test_metrics:
        print(f"  NPV:         {test_metrics['npv']:.4f}  (论文: 0.9834)")
    if 'roc_auc' in test_metrics:
        print(f"  ROC-AUC:     {test_metrics['roc_auc']:.4f}  (论文: 0.9350)")
    
    print("\n分类报告:")
    from sklearn.metrics import classification_report
    print(classification_report(test_labels, test_preds, target_names=['非欺诈', '欺诈']))
    
    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("混淆矩阵:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    
    # ====== Step 8: 保存结果 ======
    results = {
        'test_metrics': test_metrics,
        'training_history': history,
        'config': config,
        'paper_metrics': {
            'accuracy': 0.94,
            'recall': 0.9167,
            'f1': 0.8498,
            'precision': 0.7920,
            'specificity': 0.9534,
            'npv': 0.9834,
            'roc_auc': 0.9350,
        }
    }
    
    # 转换 numpy 类型为 Python 原生类型（JSON 序列化）
    def convert_numpy(obj):
        if isinstance(obj, (np.int64, np.int32, np.int_)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy(x) for x in obj]
        return obj
    
    results = convert_numpy(results)
    
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {results_path}")
    
    return model, results


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='D-STAR 欺诈通话检测模型训练')
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--epochs', type=int, default=CONFIG['max_epochs'])
    parser.add_argument('--lr', type=float, default=CONFIG['learning_rate'])
    parser.add_argument('--lambda_sparse', type=float, default=CONFIG['lambda_sparse'])
    parser.add_argument('--alpha', type=float, default=CONFIG['alpha'],
                        help='Top-k 稀疏率 (保留的 token 比例)')
    parser.add_argument('--num_heads', type=int, default=CONFIG['num_heads'])
    parser.add_argument('--no_kg', action='store_true', help='不使用知识图谱')
    parser.add_argument('--freeze_distilbert', action='store_true')
    
    args = parser.parse_args()
    
    # 更新配置
    config = CONFIG.copy()
    config['batch_size'] = args.batch_size
    config['max_epochs'] = args.epochs
    config['learning_rate'] = args.lr
    config['lambda_sparse'] = args.lambda_sparse
    config['alpha'] = args.alpha
    config['num_heads'] = args.num_heads
    config['use_knowledge_graph'] = not args.no_kg
    config['freeze_distilbert'] = args.freeze_distilbert
    
    print("训练配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    model, results = train(config)
