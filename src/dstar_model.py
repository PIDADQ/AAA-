"""
D-STAR (Dynamic Sparsity Top-k Attention Regularization) 模型实现
论文: Classifying Scam Calls Through Content Analysis With Dynamic Sparsity Top-k Attention Regularization
IEEE ACCESS 2025

核心创新点:
1. Dynamic Sparse Attention (DSA): 动态稀疏注意力机制
2. Top-k Selection: 动态 Top-k 选择，只保留最重要的 k 个 token
3. Sparsity Regularization: L1 稀疏正则化，鼓励注意力权重稀疏化
4. Knowledge Graph Preprocessing: 知识图谱辅助预处理

论文关键参数:
- 2 个 encoder layers
- 8 个 attention heads, hidden_dim=768
- max_seq_len=5 tokens (最优实验结果)
- batch_size=16, epochs=20
- lr=2e-5, L1 regularization lambda=0.01
- 80/20 train-test split
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple
from transformers import DistilBertModel, DistilBertTokenizer


# ============================================================
# 1. Dynamic Sparse Attention with Top-k Selection (DSA)
# ============================================================

class DynamicSparseAttention(nn.Module):
    """
    动态稀疏注意力机制 (DSA) + Top-k 选择
    
    标准注意力:  A = softmax(QK^T / sqrt(d_k)) V
    D-STAR注意力:
        1. 计算全量注意力分数 Z = QK^T / sqrt(d_k)
        2. Top-k 选择: 保留每行得分最高的 k 个位置
        3. 稀疏 softmax: 只对 Top-k 位置归一化
        4. 稀疏矩阵乘法: SpMM
    
    Top-k 动态计算:
        k = floor(alpha * n)
        其中 n 为序列长度, alpha 为稀疏率控制参数
    
    论文公式 (2): k = floor(alpha * n)
    论文公式 (3): A_sparse(i,j) = A(i,j) if A(i,j) >= threshold(k), else 0
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        alpha: float = 0.5,          # 稀疏率控制参数，保留 50% 的 token
        dropout_rate: float = 0.1,
        lambda_sparse: float = 0.01,  # L1 稀疏正则化强度
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.alpha = alpha
        self.lambda_sparse = lambda_sparse
        self.scale = math.sqrt(self.head_dim)
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 用于记录稀疏正则化损失
        self.sparsity_loss = torch.tensor(0.0)
    
    def _compute_topk_mask(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """
        计算 Top-k 掩码
        
        Args:
            attn_scores: (batch, heads, seq_len, seq_len)
        Returns:
            mask: (batch, heads, seq_len, seq_len) bool mask, True = 保留
        """
        batch, heads, n, _ = attn_scores.shape
        
        # 动态计算 k: k = floor(alpha * n)，论文公式 (2)
        k = max(1, int(math.floor(self.alpha * n)))
        k = min(k, n)  # 不超过序列长度
        
        # 对每个 query token，找到得分最高的 k 个 key token
        # attn_scores: (batch, heads, n, n)
        # topk_indices: (batch, heads, n, k)
        _, topk_indices = attn_scores.topk(k, dim=-1, largest=True, sorted=False)
        
        # 构建掩码
        mask = torch.zeros_like(attn_scores, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        
        return mask, k
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len), 1=有效 0=padding
        Returns:
            output: (batch, seq_len, hidden_dim)
            sparsity_loss: 稀疏正则化损失标量
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # === Step 1: 线性投影 ===
        Q = self.q_proj(hidden_states)  # (batch, seq, hidden)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        
        # === Step 2: 拆分多头 ===
        def split_heads(x):
            # (batch, seq, hidden) -> (batch, heads, seq, head_dim)
            b, s, h = x.shape
            x = x.view(b, s, self.num_heads, self.head_dim)
            return x.transpose(1, 2)
        
        Q = split_heads(Q)  # (batch, heads, seq, head_dim)
        K = split_heads(K)
        V = split_heads(V)
        
        # === Step 3: 计算全量注意力分数 (SDDMM 对应步骤) ===
        # Z = QK^T / sqrt(d_k), 论文公式 (1)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale  # (batch, heads, seq, seq)
        
        # 处理 padding mask
        if attention_mask is not None:
            # attention_mask: (batch, seq) -> (batch, 1, 1, seq)
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = extended_mask.float()
            extended_mask = (1.0 - extended_mask) * -1e9
            attn_scores = attn_scores + extended_mask
        
        # === Step 4: Top-k 选择 + 稀疏化 (论文公式 2, 3) ===
        topk_mask, k = self._compute_topk_mask(attn_scores)
        
        # 将非 Top-k 位置的分数设为 -inf，使其 softmax 后为 0
        sparse_scores = attn_scores.masked_fill(~topk_mask, float('-inf'))
        
        # === Step 5: 稀疏 Softmax (Sparse softmax) ===
        A_sparse = F.softmax(sparse_scores, dim=-1)
        
        # 处理全为 -inf 的行（全 padding 情况）
        A_sparse = torch.nan_to_num(A_sparse, nan=0.0)
        
        A_sparse = self.dropout(A_sparse)
        
        # === Step 6: 稀疏矩阵-向量乘法 (SpMM) ===
        output = torch.matmul(A_sparse, V)  # (batch, heads, seq, head_dim)
        
        # === Step 7: 合并多头 ===
        output = output.transpose(1, 2).contiguous()  # (batch, seq, heads, head_dim)
        output = output.view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)
        
        # === Step 8: L1 稀疏正则化损失 (论文公式 4) ===
        # L_sparsity = lambda * sum(|A_sparse(i,j)|)
        sparsity_loss = self.lambda_sparse * A_sparse.abs().sum()
        
        return output, sparsity_loss


# ============================================================
# 2. D-STAR Transformer Encoder Layer
# ============================================================

class DSTAREncoderLayer(nn.Module):
    """
    D-STAR Transformer Encoder 层
    
    结构:
    - DSA (Dynamic Sparse Attention)
    - LayerNorm + 残差连接
    - Feed-Forward Network
    - LayerNorm + 残差连接
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        ffn_dim: int = 3072,
        alpha: float = 0.5,
        dropout_rate: float = 0.1,
        lambda_sparse: float = 0.01,
    ):
        super().__init__()
        
        self.attention = DynamicSparseAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            alpha=alpha,
            dropout_rate=dropout_rate,
            lambda_sparse=lambda_sparse,
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-Attention + 残差
        attn_output, sparsity_loss = self.attention(hidden_states, attention_mask)
        hidden_states = self.norm1(hidden_states + attn_output)
        
        # FFN + 残差
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.norm2(hidden_states + ffn_output)
        
        return hidden_states, sparsity_loss


# ============================================================
# 3. D-STAR 完整模型
# ============================================================

class DSTARModel(nn.Module):
    """
    D-STAR 完整模型架构
    
    论文架构:
    Input Text
        |
    DistilBERT Tokenizer & Embedding (冻结或微调)
        |
    Knowledge Graph Filter (预处理阶段完成)
        |
    2x D-STAR Encoder Layers (DSA + Top-k + Sparsity Reg)
        |
    Pooling (CLS token or mean pooling)
        |
    Classifier (Binary)
        |
    Output: Scam / Non-Scam
    
    论文配置:
    - 2 encoder layers
    - 8 heads, hidden_dim=768
    - max_seq_len=5 (实验最优)
    - batch=16, epochs=20, lr=2e-5
    - lambda_sparse=0.01
    """
    
    def __init__(
        self,
        pretrained_model: str = "distilbert-base-multilingual-cased",
        num_labels: int = 2,
        num_encoder_layers: int = 2,
        num_heads: int = 8,
        hidden_dim: int = 768,
        ffn_dim: int = 3072,
        alpha: float = 0.5,
        dropout_rate: float = 0.1,
        lambda_sparse: float = 0.01,
        freeze_distilbert: bool = False,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.lambda_sparse = lambda_sparse
        
        # DistilBERT 作为 embedding 提取器
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model)
        distilbert_hidden = self.distilbert.config.hidden_size
        
        if freeze_distilbert:
            for param in self.distilbert.parameters():
                param.requires_grad = False
        
        # 投影层：将 DistilBERT 隐层维度映射到 D-STAR 维度
        if distilbert_hidden != hidden_dim:
            self.input_projection = nn.Linear(distilbert_hidden, hidden_dim)
        else:
            self.input_projection = nn.Identity()
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout_rate, max_len=512)
        
        # D-STAR Encoder Layers (论文: 2 layers)
        self.encoder_layers = nn.ModuleList([
            DSTAREncoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                alpha=alpha,
                dropout_rate=dropout_rate,
                lambda_sparse=lambda_sparse,
            )
            for _ in range(num_encoder_layers)
        ])
        
        # 分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        
        # 初始化分类头权重
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: (batch,) optional
        
        Returns:
            dict with 'logits', 'loss' (if labels provided), 'sparsity_loss'
        """
        # Step 1: DistilBERT embedding
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # (batch, seq, distilbert_hidden)
        hidden_states = distilbert_output.last_hidden_state
        
        # Step 2: 投影到 D-STAR 维度
        hidden_states = self.input_projection(hidden_states)
        
        # Step 3: 位置编码
        hidden_states = self.pos_encoding(hidden_states)
        
        # Step 4: D-STAR Encoder Layers
        total_sparsity_loss = torch.tensor(0.0, device=hidden_states.device)
        for layer in self.encoder_layers:
            hidden_states, sparsity_loss = layer(hidden_states, attention_mask)
            total_sparsity_loss = total_sparsity_loss + sparsity_loss
        
        # Step 5: Pooling — 使用 CLS token (位置 0)
        pooled = hidden_states[:, 0, :]  # (batch, hidden_dim)
        pooled = self.dropout(pooled)
        
        # Step 6: 分类
        logits = self.classifier(pooled)  # (batch, num_labels)
        
        result = {
            'logits': logits,
            'sparsity_loss': total_sparsity_loss,
        }
        
        if labels is not None:
            # Binary cross-entropy loss (论文使用的损失函数)
            ce_loss = F.cross_entropy(logits, labels)
            # 总损失 = CE + L1 稀疏正则化
            total_loss = ce_loss + total_sparsity_loss
            result['loss'] = total_loss
            result['ce_loss'] = ce_loss
        
        return result


# ============================================================
# 4. 位置编码
# ============================================================

class PositionalEncoding(nn.Module):
    """标准 Transformer 位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================
# 5. 模型工厂函数
# ============================================================

def create_dstar_model(
    pretrained_model: str = "distilbert-base-multilingual-cased",
    num_labels: int = 2,
    freeze_distilbert: bool = False,
    **kwargs
) -> DSTARModel:
    """
    创建 D-STAR 模型
    
    论文最优配置:
    - 2 encoder layers
    - 8 attention heads
    - hidden_dim=768
    - alpha=0.5 (保留 50% token)
    - lambda_sparse=0.01
    - max_seq_len 在 tokenizer 中设置为 512 (训练), 推理时用 5
    """
    model = DSTARModel(
        pretrained_model=pretrained_model,
        num_labels=num_labels,
        num_encoder_layers=kwargs.get('num_encoder_layers', 2),
        num_heads=kwargs.get('num_heads', 8),
        hidden_dim=kwargs.get('hidden_dim', 768),
        ffn_dim=kwargs.get('ffn_dim', 3072),
        alpha=kwargs.get('alpha', 0.5),
        dropout_rate=kwargs.get('dropout_rate', 0.1),
        lambda_sparse=kwargs.get('lambda_sparse', 0.01),
        freeze_distilbert=freeze_distilbert,
    )
    return model


if __name__ == "__main__":
    # 快速测试
    print("Testing D-STAR model components...")
    
    # 测试 DSA
    dsa = DynamicSparseAttention(hidden_dim=64, num_heads=4, alpha=0.5)
    x = torch.randn(2, 10, 64)
    mask = torch.ones(2, 10)
    out, loss = dsa(x, mask)
    print(f"DSA output shape: {out.shape}, sparsity_loss: {loss.item():.4f}")
    
    # 测试 Encoder Layer
    encoder = DSTAREncoderLayer(hidden_dim=64, num_heads=4, ffn_dim=256, alpha=0.5)
    out, loss = encoder(x, mask)
    print(f"Encoder output shape: {out.shape}, sparsity_loss: {loss.item():.4f}")
    
    print("D-STAR model components test passed!")
