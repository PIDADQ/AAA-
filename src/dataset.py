"""
数据集加载与预处理模块
针对欺诈通话检测数据集

适配论文 D-STAR 的数据预处理流程:
1. 文本清洗 (lowercase, 去符号, 词干化)
2. 知识图谱过滤与增强
3. DistilBERT tokenization
4. 80/20 训练/测试划分

数据文件:
- ../code1/训练集结果.csv  (含验证集，从中划分20%)
- ../code1/测试集结果.csv
"""

import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split

from knowledge_graph import KnowledgeGraphProcessor


# ============================================================
# 1. 文本预处理函数（复现论文预处理流程）
# ============================================================

def clean_text(text: str) -> str:
    """
    论文描述的文本预处理:
    1. 转小写 (converting all text to lowercase)
    2. 移除无关符号和数字 (removing irrelevant symbols and numerical characters)
    3. 移除常用对话套语 (removing common conversational phrases)
    4. 词干化 (stemming using Porter Stemmer)
    
    本版本针对中文文本做了调整
    """
    if not isinstance(text, str):
        return ""
    
    # 移除音频标记格式
    text = re.sub(r'音频内容[：:]?', '', text)
    text = re.sub(r'left\s*[:：]\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'right\s*[:：]\s*', '', text, flags=re.IGNORECASE)
    
    # 移除方括号和其中内容（通常是说话人标注）
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'【.*?】', '', text)
    
    # 移除数字（可选，保留中文数字）
    text = re.sub(r'\d+', '', text)
    
    # 移除常用问候语和无意义套话
    stopwords = [
        '你好', '喂', '嗯', '啊', '哦', '噢', '是的', '对',
        '好的', '谢谢', '再见', '不客气',
    ]
    for sw in stopwords:
        text = text.replace(sw, '')
    
    # 保留中文字符、中文标点和空格
    text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\s]', ' ', text)
    
    # 合并多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_and_preprocess_data(
    train_path: str = None,
    test_path: str = None,
    val_ratio: float = 0.2,
    use_knowledge_graph: bool = True,
    kg_mode: str = "append",
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载并预处理数据集
    
    Args:
        train_path: 训练集CSV路径（默认自动推断）
        test_path: 测试集CSV路径（默认自动推断）
        val_ratio: 从训练集中划分的验证集比例
        use_knowledge_graph: 是否使用知识图谱增强
        kg_mode: 知识图谱模式 ("append"/"filter"/"original")
        random_seed: 随机种子
    
    Returns:
        train_df, val_df, test_df
    """
    # 自动推断路径
    if train_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_path = os.path.join(script_dir, "..", "code1", "训练集结果.csv")
    if test_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_path = os.path.join(script_dir, "..", "code1", "测试集结果.csv")
    
    print(f"加载训练集: {train_path}")
    print(f"加载测试集: {test_path}")
    
    # 加载数据
    train_df = pd.read_csv(train_path, encoding='utf-8')
    test_df = pd.read_csv(test_path, encoding='utf-8')
    
    print(f"原始训练集大小: {len(train_df)}")
    print(f"原始测试集大小: {len(test_df)}")
    
    # 处理缺失值
    for df_name, df in [("训练集", train_df), ("测试集", test_df)]:
        missing = df['is_fraud'].isnull().sum()
        if missing > 0:
            print(f"  {df_name} 中 is_fraud 有 {missing} 个缺失值，已删除")
    
    train_df = train_df.dropna(subset=['is_fraud']).copy()
    test_df = test_df.dropna(subset=['is_fraud']).copy()
    
    # 转换标签为整数
    train_df['is_fraud'] = train_df['is_fraud'].astype(int)
    test_df['is_fraud'] = test_df['is_fraud'].astype(int)
    
    # 文本预处理
    print("\n正在进行文本预处理...")
    
    if use_knowledge_graph:
        print(f"使用知识图谱增强 (mode={kg_mode})")
        kg_processor = KnowledgeGraphProcessor()
        
        train_df['processed_text'] = [
            kg_processor.enrich_text_with_kg(text, kg_mode)
            for text in train_df['specific_dialogue_content']
        ]
        test_df['processed_text'] = [
            kg_processor.enrich_text_with_kg(text, kg_mode)
            for text in test_df['specific_dialogue_content']
        ]
    else:
        train_df['processed_text'] = train_df['specific_dialogue_content'].apply(clean_text)
        test_df['processed_text'] = test_df['specific_dialogue_content'].apply(clean_text)
    
    # 处理空文本
    empty_train = (train_df['processed_text'] == '').sum()
    empty_test = (test_df['processed_text'] == '').sum()
    if empty_train > 0:
        print(f"  训练集中有 {empty_train} 条空文本，已填充为空格")
        train_df['processed_text'] = train_df['processed_text'].replace('', '[空文本]')
    if empty_test > 0:
        print(f"  测试集中有 {empty_test} 条空文本，已填充为空格")
        test_df['processed_text'] = test_df['processed_text'].replace('', '[空文本]')
    
    # 从训练集划分验证集
    train_df_final, val_df = train_test_split(
        train_df,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=train_df['is_fraud'],
    )
    
    print(f"\n数据集划分结果:")
    print(f"  训练集: {len(train_df_final)} 条")
    print(f"  验证集: {len(val_df)} 条")
    print(f"  测试集: {len(test_df)} 条")
    
    # 打印标签分布
    for name, df in [("训练集", train_df_final), ("验证集", val_df), ("测试集", test_df)]:
        fraud_count = df['is_fraud'].sum()
        total = len(df)
        print(f"  {name}欺诈比例: {fraud_count}/{total} = {fraud_count/total:.1%}")
    
    return train_df_final.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ============================================================
# 2. PyTorch Dataset
# ============================================================

class FraudCallDataset(Dataset):
    """
    欺诈通话检测数据集
    
    论文实验设置:
    - max_seq_len=5 (最优配置，论文实验结果)
    - 但训练时建议用更长的序列以获取更多上下文
    - tokenizer: DistilBERT
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: DistilBertTokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: DistilBertTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建 DataLoader
    
    Args:
        train_df, val_df, test_df: 数据集 DataFrame
        tokenizer: DistilBERT tokenizer
        batch_size: 论文使用 16
        max_length: 最大序列长度（论文实验用 5，但实际训练建议 512）
        num_workers: 数据加载线程数
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = FraudCallDataset(
        texts=train_df['processed_text'].tolist(),
        labels=train_df['is_fraud'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    val_dataset = FraudCallDataset(
        texts=val_df['processed_text'].tolist(),
        labels=val_df['is_fraud'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    test_dataset = FraudCallDataset(
        texts=test_df['processed_text'].tolist(),
        labels=test_df['is_fraud'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    return train_loader, val_loader, test_loader


# ============================================================
# 3. 测试
# ============================================================

if __name__ == "__main__":
    # 测试数据加载
    train_df, val_df, test_df = load_and_preprocess_data(
        use_knowledge_graph=True,
        kg_mode="append",
    )
    
    print("\n样本示例:")
    print(f"文本: {train_df['processed_text'].iloc[0][:100]}...")
    print(f"标签: {train_df['is_fraud'].iloc[0]}")
    
    print("\n数据加载模块测试完成!")
