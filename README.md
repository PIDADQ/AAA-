# Fraud-R1-DSTAR: 基于动态稀疏注意力与社工策略增强的中文欺诈通话检测

本项目复现并扩展了 D-STAR 模型，结合 Fraud-R1 的社会工程学攻击增强框架，在中文欺诈通话数据集上完成模型训练、鲁棒性评测与防御微调。

---

## 一、项目简介

电信诈骗是当前社会危害最严重的犯罪类型之一。2024年全国公安机关破获电信网络诈骗案件约56.9万起，涉案金额超过3000亿元，受害者涵盖各年龄段，老年人与高学历人群均难以幸免。诈骗分子利用社会工程学手段，通过话术包装降低受害者的警惕性，传统基于关键词匹配或浅层机器学习的检测方法难以应对这种隐蔽性攻击。

D-STAR 提出动态稀疏注意力机制，通过 Top-k 选择和 L1 正则化让模型自动聚焦对话中最具区分度的 token，忽略无意义套话。Fraud-R1 则将社工攻击抽象为建立信任、制造紧迫感、情感操纵三种策略模板，用于评测目标模型的鲁棒性。

本项目在中文欺诈通话数据集上复现 D-STAR，并用 Fraud-R1 框架生成增强测试集，评估模型对社工诱导的脆弱性，最后通过组合策略数据增强与热启动微调显著提升防御能力。

---

## 二、文件结构

```
Fraud-R1-DSTAR/
├── src/                              # 核心代码
│   ├── dstar_model.py                # D-STAR 模型架构（DSA + Top-k + L1 正则）
│   ├── dataset.py                    # 数据加载与预处理（含知识图谱增强）
│   ├── knowledge_graph.py            # 中文欺诈知识图谱构建
│   ├── train_dstar.py                # 原始模型训练脚本
│   ├── augment_dataset.py            # Fraud-R1 策略增强（三种单策略 + 组合策略）
│   ├── evaluate_augmented.py         # 增强测试集批量评估
│   ├── train_augmented.py            # 组合策略增强后热启动微调
│   ├── ablation_and_visualization.py # 消融实验与可视化
│   ├── generate_figures.py           # 论文图表复现
│   └── requirements.txt              # Python 依赖
│
├── data/                             # 数据集与增强结果
│   ├── summary_metrics.csv           # 实验二原始模型评估汇总
│   ├── 增强测试集_original.csv        # 原始测试集
│   ├── 增强测试集_credibility.csv     # 建立信任增强测试集
│   ├── 增强测试集_urgency.csv         # 制造紧迫感增强测试集
│   ├── 增强测试集_emotional.csv       # 情感操纵增强测试集
│   └── 增强测试集_combined.csv        # 组合策略增强测试集
│
├── results/                          # 实验结果与可视化
│   ├── summary_metrics微调前测试集.csv
│   ├── summary_metrics微调后测试集.csv
│   ├── confusion_matrices_english.png
│   ├── metrics_comparison_english.png
│   ├── robustness_drop_english.png
│   ├── training_curves_english.png
│   └── figures/
│       ├── model_architecture.png    # 模型架构图
│       ├── formula_visualization.png # Top-k 与 L1 训练曲线
│       ├── performance_comparison.png# 性能对比与消融实验
│       └── sensitivity_analysis.png  # 敏感性分析
│
└── README.md                         # 本文件
```

---

## 三、环境配置

```bash
# 创建虚拟环境
conda create -n fraud-r1 python=3.10
conda activate fraud-r1

# 安装依赖
cd src
pip install -r requirements.txt
```

依赖包：torch>=1.12.0、transformers>=4.20.0、scikit-learn>=1.0.0、pandas>=1.3.0、numpy>=1.21.0、matplotlib>=3.5.0、jieba>=0.42.1、tqdm>=4.62.0

---

## 四、快速开始

### 4.1 训练原始 D-STAR 模型

```bash
cd src
python train_dstar.py --no_kg
```

训练完成后生成 `best_model.pt`，同时输出训练集和测试集的评估指标。

### 4.2 生成增强测试集

```bash
python augment_dataset.py
```

在 `data/` 目录下生成 5 个增强测试集 CSV 文件。

### 4.3 评估原始模型鲁棒性

```bash
python evaluate_augmented.py --no_kg
```

输出原始模型在 5 个测试集上的 Accuracy、Precision、Recall、F1、ROC-AUC，并生成混淆矩阵图。

### 4.4 组合策略增强微调

```bash
python train_augmented.py --no_kg
```

在 `best_model.pt` 基础上热启动微调，学习率降至 5e-6，训练 10 个 epoch。

### 4.5 评估微调后模型

```bash
python evaluate_augmented.py --no_kg --model_path outputs/finetuned_model.pt
```

---

## 五、实验结果

### 5.1 实验一：原始模型性能

| 指标 | 本实验结果 | 论文英文结果 | 提升幅度 |
|------|-----------|-------------|---------|
| Accuracy | 99.88% | 94.74% | +5.14% |
| Precision | 99.78% | 79.20% | +20.58% |
| Recall | 100.00% | 91.67% | +8.33% |
| F1-Score | 99.89% | 84.98% | +14.91% |
| ROC-AUC | 99.9985% | 93.50% | +6.50% |

### 5.2 实验二：原始模型鲁棒性

| 数据集 | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|---------|-----------|--------|---------|---------|
| 原始对照 | 99.84% | 99.71% | 100% | 99.86% | 100.00% |
| 建立信任 | 76.81% | 70.12% | 100% | 82.44% | 98.92% |
| 制造紧迫感 | 99.57% | 99.21% | 100% | 99.61% | 100.00% |
| 情感操纵 | 99.65% | 99.36% | 100% | 99.68% | 100.00% |
| 组合策略 | 74.29% | 67.92% | 100% | 80.90% | 98.51% |

建立信任策略导致 Accuracy 骤降 23.03 个百分点，组合策略降幅达 25.55 个百分点，三种策略存在协同干扰效应。

### 5.3 微调后防御能力

| 测试集 | 微调前 Acc | 微调后 Acc | 微调前 F1 | 微调后 F1 | 变化 |
|--------|-----------|-----------|----------|----------|------|
| 原始测试集 | 99.84% | 99.88% | 99.86% | 99.89% | 基本持平 |
| 建立信任增强集 | 76.81% | 99.65% | 82.44% | 99.68% | 大幅上升 |
| 制造紧迫感增强集 | 99.57% | 99.80% | 99.61% | 99.82% | 小幅上升 |
| 情感操纵增强集 | 99.65% | 99.84% | 99.68% | 99.86% | 小幅上升 |
| 组合策略增强集 | 74.29% | 99.45% | 80.90% | 99.50% | 大幅上升 |

组合策略数据增强与热启动微调使建立信任测试集 Accuracy 提升 22.84 个百分点，组合策略测试集提升 25.16 个百分点，近乎完全恢复。

---

## 六、模型架构

D-STAR 模型数据流分为五个阶段：

1. **知识图谱预处理**：jieba 分词匹配诈骗关键词，追加类型标签
2. **DistilBERT Tokenizer**：最大序列长度 512，生成 token 嵌入向量
3. **两层 D-STAR Encoder**：每层含 DSA 模块、两个 LayerNorm、一个前馈网络
4. **CLS 位置平均池化**：取 CLS 隐藏状态做平均
5. **线性分类器 768→2**：输出欺诈/非欺诈概率

核心公式：

- 标准自注意力：Attention(Q, K, V) = softmax(QK^T / √d_k) V
- Top-k 阈值：k = floor(α · n)，α 为稀疏率控制参数，取 0.5
- L1 稀疏正则：L_sparsity = λ · Σ|A_sparse(i, j)|，λ 取 0.01
- 总损失：L_total = L_CE + L_sparsity

---

## 七、训练配置

| 参数 | 实验一原始训练 | 实验二微调 |
|------|--------------|-----------|
| batch_size | 16 | 16 |
| 最大 epoch | 20 | 10 |
| 实际 epoch | 8（早停） | 约 6（早停） |
| 基础学习率 | 2e-5 | 5e-6 |
| D-STAR 头部学习率 | 1e-4 | 2.5e-5 |
| 优化器 | AdamW | AdamW |
| 权重衰减 | 0.01 | 0.01 |
| 预热比例 | 0.1 | 0.1 |
| 梯度裁剪 max_norm | 1.0 | 1.0 |
| 早停 patience | 5 | 4 |
| 早停监控指标 | 验证集 F1 | 验证集 F1 |
| L1 正则化 λ | 0.01 | 0.01 |
| 稀疏率 α | 0.5 | 0.5 |
| 预训练模型 | distilbert-base-multilingual-cased | 同上 |

---

## 八、数据集说明

本项目使用深圳大学机器学习初步课程提供的中文虚假通话对话数据集。

- 训练集：约 8000 条对话，欺诈样本占比约 54%
- 测试集：2548 条对话，欺诈样本占比约 54%
- 每条样本包含 dialogue_id、text、label 三个字段
- label=1 表示欺诈通话，label=0 表示正常通话

增强后的训练集将原始样本与组合策略增强样本合并，数据量约为原来的两倍。

---

## 九、核心代码说明

| 文件 | 功能 |
|------|------|
| dstar_model.py | DynamicSparseAttention、DSTAREncoderLayer、DSTARModel 三个核心类 |
| dataset.py | 数据加载、文本清洗、知识图谱增强、80/20 划分、DataLoader 构建 |
| knowledge_graph.py | 中文诈骗知识图谱，覆盖 9 大类诈骗类型 |
| train_dstar.py | 原始模型训练，含分层学习率、早停、梯度裁剪 |
| augment_dataset.py | 三种单策略 + 组合策略模板，生成增强测试集 |
| evaluate_augmented.py | 批量评估 5 个测试集，输出指标表格与混淆矩阵 |
| train_augmented.py | 热启动微调，从 best_model.pt 继续训练 |
| ablation_and_visualization.py | 消融实验与敏感性分析 |
| generate_figures.py | 论文图表复现，不依赖 transformers |

---

## 十、参考文献

1. D-STAR: Classifying Scam Calls Through Content Analysis With Dynamic Sparsity Top-k Attention Regularization. IEEE Access, 2025.
2. Fraud-R1: A Multi-Round Benchmark for Assessing the Robustness of LLM Against Augmented Fraud and Phishing Inducements. ACL Findings, 2025.
3. Sanh V, Debut L, Chaumond J, et al. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019.
4. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. Advances in Neural Information Processing Systems, 2017: 5998-6008.
5. 课程提供的中文虚假通话对话数据集. 深圳大学机器学习初步课程, 2025.
