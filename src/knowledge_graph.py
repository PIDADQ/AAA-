"""
知识图谱预处理模块 (Knowledge Graph Preprocessing)
论文: D-STAR (Dynamic Sparsity Top-k Attention Regularization)

论文描述:
- 知识图谱是一个分类体系结构 (taxonomy-based structure)
- 手工构建，包含诈骗类型及其关联关键词
- 作为过滤器，帮助模型专注于诈骗相关特征
- 定期更新以涵盖新兴诈骗手法

原论文使用英文数据集，本复现版本针对中文欺诈通话数据集构建了对应的中文知识图谱。

知识图谱结构:
ScamTypes {
    诈骗类型1: [关键词列表],
    诈骗类型2: [关键词列表],
    ...
}
"""

import re
import jieba
import jieba.analyse
from typing import Dict, List, Optional, Set


# ============================================================
# 1. 欺诈通话知识图谱（中文版）
# ============================================================

FRAUD_KNOWLEDGE_GRAPH: Dict[str, List[str]] = {
    
    # === 网络购物诈骗 ===
    "购物诈骗": [
        "退款", "赔偿", "商品", "订单", "快递", "物流",
        "客服", "平台", "店铺", "评价", "刷单", "佣金",
        "优惠券", "返现", "积分", "会员", "购买", "交易",
        "买家", "卖家", "淘宝", "京东", "拼多多", "二维码",
    ],
    
    # === 金融诈骗 ===
    "金融诈骗": [
        "贷款", "借款", "利息", "征信", "信用", "额度",
        "刷流水", "开户", "银行卡", "转账", "汇款", "手续费",
        "保证金", "押金", "风控", "审核", "解冻", "冻结",
        "投资", "理财", "收益", "回报", "股票", "基金",
        "期货", "虚拟货币", "比特币", "资产", "资金",
    ],
    
    # === 冒充公检法诈骗 ===
    "冒充公检法": [
        "警察", "公安", "检察院", "法院", "案件", "涉案",
        "犯罪", "洗钱", "协助调查", "不得告知", "秘密账户",
        "保密", "传票", "拘留", "逮捕", "刑事", "立案",
        "缉毒", "反恐", "国家安全", "取款", "配合",
    ],
    
    # === 冒充企业/机构诈骗 ===
    "冒充企业机构": [
        "官方客服", "工作人员", "认证", "授权", "总部",
        "品牌方", "合作", "营业执照", "资质", "正规",
        "签合同", "合同", "协议", "法律", "维权",
    ],
    
    # === 兼职刷单诈骗 ===
    "兼职刷单": [
        "兼职", "刷单", "刷好评", "任务", "提成", "佣金",
        "在家做", "赚钱", "日结", "高薪", "零门槛",
        "微信群", "QQ群", "接单", "垫付", "返还",
    ],
    
    # === 情感/婚恋诈骗 ===
    "情感诈骗": [
        "交友", "相亲", "恋爱", "男朋友", "女朋友",
        "感情", "喜欢", "爱", "见面", "约会", "礼物",
        "红包", "转账", "困难", "急需", "帮忙",
    ],
    
    # === 虚假中奖诈骗 ===
    "中奖诈骗": [
        "中奖", "获奖", "大奖", "彩票", "抽奖",
        "恭喜", "幸运", "奖金", "领奖", "手续费",
        "个人所得税", "激活费", "公证费",
    ],
    
    # === 诱导性话术特征词 ===
    "诱导话术": [
        "立即", "马上", "紧急", "限时", "现在",
        "不要告诉别人", "保密", "按照我说的做",
        "不要挂电话", "一直保持通话", "帮你解决",
        "保证", "确保", "绝对安全", "放心",
        "操作账户", "转到安全账户",
    ],
    
    # === 账号安全类诈骗 ===
    "账号安全": [
        "账号", "密码", "验证码", "短信", "登录",
        "被盗", "异常", "风险", "安全验证", "实名认证",
        "绑定手机", "解绑", "注销", "申诉",
    ],
    
    # === 通用诈骗信号词 ===
    "通用信号": [
        "钱", "汇", "打款", "付款", "交钱", "缴费",
        "不要声张", "私下解决", "配合", "信任",
        "受害者", "损失", "挽回", "赔偿",
    ],
}

# 合法/正常通话的特征词（非诈骗）
NORMAL_CALL_KEYWORDS: List[str] = [
    "普通咨询", "售后服务", "物流查询", "账单查询",
    "预约挂号", "候诊", "体检", "学校通知",
    "家庭", "朋友", "同事", "日常", "正常业务",
]


# ============================================================
# 2. 知识图谱处理器
# ============================================================

class KnowledgeGraphProcessor:
    """
    知识图谱预处理器
    
    功能:
    1. 过滤文本中的诈骗相关关键词
    2. 标记诈骗类型
    3. 丰富文本的诈骗语义特征
    
    论文描述:
    "the knowledge graph enriches the preprocessing phase by filtering out
    irrelevant content and enriching the input data with scam-related context"
    """
    
    def __init__(self, knowledge_graph: Dict[str, List[str]] = None):
        self.kg = knowledge_graph or FRAUD_KNOWLEDGE_GRAPH
        
        # 构建关键词到诈骗类型的映射
        self.keyword_to_type: Dict[str, str] = {}
        self.all_keywords: Set[str] = set()
        
        for scam_type, keywords in self.kg.items():
            for kw in keywords:
                self.keyword_to_type[kw] = scam_type
                self.all_keywords.add(kw)
        
        # 加载 jieba 自定义词典
        self._load_custom_words()
    
    def _load_custom_words(self):
        """将知识图谱关键词加入 jieba 分词词典"""
        for kw in self.all_keywords:
            jieba.add_word(kw, freq=100, tag='fraud')
    
    def extract_fraud_features(self, text: str) -> Dict:
        """
        从文本中提取诈骗特征
        
        Args:
            text: 原始对话文本
        Returns:
            dict: {
                'keywords_found': 找到的关键词列表,
                'scam_types': 涉及的诈骗类型列表,
                'fraud_score': 诈骗关键词密度得分,
                'filtered_text': 经过知识图谱过滤后的文本
            }
        """
        # 清洗文本
        cleaned = self._clean_text(text)
        
        # 分词
        words = jieba.lcut(cleaned)
        
        # 匹配关键词
        found_keywords = []
        found_types = set()
        
        for word in words:
            if word in self.keyword_to_type:
                found_keywords.append(word)
                found_types.add(self.keyword_to_type[word])
        
        # 计算诈骗关键词密度
        total_words = max(len(words), 1)
        fraud_score = len(found_keywords) / total_words
        
        # 生成过滤后的文本（保留诈骗相关内容）
        filtered_text = ' '.join(found_keywords) if found_keywords else cleaned
        
        return {
            'keywords_found': found_keywords,
            'scam_types': list(found_types),
            'fraud_score': fraud_score,
            'filtered_text': filtered_text,
            'original_text': cleaned,
        }
    
    def _clean_text(self, text: str) -> str:
        """
        文本清洗 (论文描述的预处理步骤)
        
        步骤:
        1. 移除音频标记 (left:/right: 等)
        2. 转为小写（英文部分）
        3. 移除特殊符号
        4. 移除数字（可选）
        """
        if not isinstance(text, str):
            return ""
        
        # 移除音频标记
        text = re.sub(r'音频内容[：:]?', '', text)
        text = re.sub(r'left\s*[:：]\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'right\s*[:：]\s*', '', text, flags=re.IGNORECASE)
        
        # 移除特殊格式
        text = re.sub(r'\[.*?\]', '', text)  # 移除方括号内容
        text = re.sub(r'【.*?】', '', text)  # 移除中文方括号内容
        
        # 移除无意义的符号（保留中文标点）
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\w\s，。！？、；：""''（）]', ' ', text)
        
        # 英文转小写
        text = text.lower()
        
        # 合并多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def enrich_text_with_kg(self, text: str, mode: str = "append") -> str:
        """
        用知识图谱丰富文本
        
        Args:
            text: 原始文本
            mode: 
                "append" - 在原文后追加找到的关键词
                "filter" - 只保留知识图谱中的关键词
                "original" - 原样返回（仅清洗）
        Returns:
            处理后的文本
        """
        features = self.extract_fraud_features(text)
        cleaned_text = features['original_text']
        
        if mode == "append" and features['keywords_found']:
            # 在原文后追加发现的诈骗关键词（增强语义）
            kg_context = ' '.join(set(features['keywords_found']))
            return f"{cleaned_text} {kg_context}"
        
        elif mode == "filter" and features['keywords_found']:
            # 只保留与诈骗相关的内容
            return features['filtered_text']
        
        return cleaned_text
    
    def batch_process(self, texts: List[str], mode: str = "append") -> List[str]:
        """批量处理文本"""
        return [self.enrich_text_with_kg(text, mode) for text in texts]
    
    def get_stats(self, texts: List[str]) -> Dict:
        """获取文本集合的诈骗特征统计"""
        all_types = []
        scores = []
        
        for text in texts:
            feat = self.extract_fraud_features(text)
            all_types.extend(feat['scam_types'])
            scores.append(feat['fraud_score'])
        
        from collections import Counter
        type_counts = Counter(all_types)
        
        return {
            'avg_fraud_score': np.mean(scores) if scores else 0,
            'top_scam_types': type_counts.most_common(5),
            'total_samples': len(texts),
        }


# ============================================================
# 3. 测试
# ============================================================

if __name__ == "__main__":
    import numpy as np
    
    kg_processor = KnowledgeGraphProcessor()
    
    # 测试诈骗文本
    fraud_text = """
    喂，你好，我是某银行客服，您的账户出现了异常交易，
    需要您立即配合我们进行安全验证，请将您账户中的资金
    转到我们的安全账户，金额为5万元，这是保密操作，
    不要告诉家人。验证码是123456，请告诉我。
    """
    
    features = kg_processor.extract_fraud_features(fraud_text)
    print("=== 诈骗文本分析 ===")
    print(f"发现关键词: {features['keywords_found']}")
    print(f"涉及诈骗类型: {features['scam_types']}")
    print(f"诈骗特征得分: {features['fraud_score']:.4f}")
    print()
    
    # 测试正常文本
    normal_text = """
    你好，我想询问一下明天的天气情况，
    以及附近有没有好的餐厅推荐？
    """
    
    features2 = kg_processor.extract_fraud_features(normal_text)
    print("=== 正常文本分析 ===")
    print(f"发现关键词: {features2['keywords_found']}")
    print(f"涉及诈骗类型: {features2['scam_types']}")
    print(f"诈骗特征得分: {features2['fraud_score']:.4f}")
    
    print("\n知识图谱模块测试完成!")
