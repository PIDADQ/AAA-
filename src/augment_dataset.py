"""
基于Fraud-R1思想的数据集增强脚本
实现三种策略：建立信任、制造紧迫感、情感操纵
"""

import pandas as pd
import random
import copy
import re

# ==================== 增强策略模板 ====================

# 策略1：建立信任（Credibility Building）
CREDIBILITY_PREFIXES = [
    "您好，我是{org}官方认证的工作人员，工号{id}，本次来电已通过国家电信管理局实名认证。",
    "请放心，我们公司已通过ISO27001信息安全认证，所有通话均有录音备查，请您放心配合。",
    "您好，我这边是{org}的高级客服主管{name}，直接管理您的账户，您可以通过官网查询我的工号{id}。",
    "我们是受监管的正规金融机构，所有操作均在银保监会监督下进行，您的资金安全有国家背书保障。",
    "您好，您之前咨询过我们的服务，我是负责跟进您案件的专员，今天特地联系您确认最新进展。",
]

CREDIBILITY_SUFFIXES = [
    "我的工号是{id}，您可以通过官方网站或拨打{phone}核实我的身份。",
    "这次来电完全是为了维护您的权益，我们已经和相关部门取得联系，确认了您的情况。",
    "我们所有的操作流程都严格按照监管要求执行，每一步都有法律保障。",
    "我在这个岗位已经工作了{years}年，帮助过很多像您一样的客户成功解决了问题。",
]

# 策略2：制造紧迫感（Urgency Creating）
URGENCY_ADDITIONS = [
    "需要提醒您的是，今天是最后期限，系统将在{time}点整自动关闭您的处理窗口，请务必抓紧时间。",
    "由于名额非常有限，目前全市只剩最后{num}个处理资格，建议您立即办理以免错过机会。",
    "请注意，如果您在{time}分钟内没有完成操作，系统将自动冻结您的账户，届时解冻需要额外缴纳{fee}元手续费。",
    "我必须告诉您，这个优惠窗口今晚{time}点截止，过了时间就无法再申请了，现在办理是最明智的选择。",
    "系统显示您的账户正处于风险状态，如果不立即处理，将在{time}小时后被列入黑名单，影响您的信用记录。",
    "领导特批的绿色通道今天{time}就要关闭了，您现在是最后一位还没确认的客户，请您配合我们完成这个步骤。",
]

# 策略3：情感操纵（Emotional Appeal）
EMOTIONAL_ADDITIONS = [
    "我完全理解您的顾虑，说实话，我自己的家人当初也遇到过同样的问题，我是真心希望帮助您的。",
    "您知道吗？我今天已经联系了几十位客户，看到大家因为这个问题遭受损失，我真的很心疼，才坚持要帮您解决的。",
    "我作为一个老员工，看到这种情况真的很痛心。公司的政策可能有些死板，但我愿意为您破例申请特殊处理。",
    "您放心，我会把这个案件作为重点来跟进，我会用我自己的信誉来保证您的权益不会受到任何损害。",
    "说真的，我已经为这个案件加班好几天了，就是希望能在期限内帮您处理好，不让您受到任何影响。",
    "我知道您可能觉得陌生，但我真的是出于好意才联系您的。如果不是因为关心您的利益，我完全不必费这个功夫。",
]

# 信任建立的模板数据
ORG_NAMES = ["工商银行", "招商银行", "建设银行", "支付宝", "微信支付", "京东金融", "中国电信", "国家税务局"]
COMMON_IDS = ["KF20241015", "CL-88823", "ZB-996007", "CS-20240101"]
COMMON_PHONES = ["4008-xxx-xxx", "010-xxxxxxxx", "400-800-xxxx"]
MALE_NAMES = ["王明", "李强", "张磊", "刘伟", "陈志远"]
FEMALE_NAMES = ["李雪", "王芳", "张晓燕", "刘美玲", "陈晓慧"]


def get_credibility_prefix():
    """生成建立信任的前缀话术"""
    template = random.choice(CREDIBILITY_PREFIXES)
    return template.format(
        org=random.choice(ORG_NAMES),
        id=random.choice(COMMON_IDS),
        name=random.choice(MALE_NAMES + FEMALE_NAMES),
        phone=random.choice(COMMON_PHONES),
    )


def get_credibility_suffix():
    """生成建立信任的后缀话术"""
    template = random.choice(CREDIBILITY_SUFFIXES)
    return template.format(
        id=random.choice(COMMON_IDS),
        phone=random.choice(COMMON_PHONES),
        years=random.randint(3, 10),
    )


def get_urgency_addition():
    """生成紧迫感话术"""
    template = random.choice(URGENCY_ADDITIONS)
    hours = random.randint(1, 6)
    minutes = random.randint(10, 59)
    fee = random.choice([50, 100, 200, 500])
    num = random.randint(1, 5)
    time_str = f"{random.randint(14, 20)}:00"
    return template.format(
        time=time_str,
        num=num,
        fee=fee,
    )


def get_emotional_addition():
    """生成情感操纵话术"""
    return random.choice(EMOTIONAL_ADDITIONS)


def augment_dialogue_credibility(dialogue: str) -> str:
    """策略1：建立信任 - 在对话开头加入身份信息"""
    lines = dialogue.strip().split('\n')
    result_lines = []
    
    # 找到第一个left说话的位置，在其后插入信任建立话术
    first_left_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('left:'):
            first_left_idx = i
            break
    
    if first_left_idx >= 0:
        # 修改第一个left的内容，加入可信度信息
        original_line = lines[first_left_idx]
        prefix = get_credibility_prefix()
        suffix = get_credibility_suffix()
        
        # 提取原始内容
        content_match = re.match(r'(left:\s*)(.*)', original_line, re.DOTALL)
        if content_match:
            new_content = f"{content_match.group(1)}{prefix} {content_match.group(2)} {suffix}"
            lines[first_left_idx] = new_content
    
    return '\n'.join(lines)


def augment_dialogue_urgency(dialogue: str) -> str:
    """策略2：制造紧迫感 - 在关键节点加入时间压力"""
    lines = dialogue.strip().split('\n')
    
    # 找到适合插入紧迫感的位置（通常在对话中后段的left说话时）
    left_indices = [i for i, line in enumerate(lines) if line.strip().startswith('left:')]
    
    if len(left_indices) >= 2:
        # 在中后段的某个left发言中插入紧迫感
        insert_idx = left_indices[len(left_indices) // 2]
        urgency_text = get_urgency_addition()
        
        original_line = lines[insert_idx]
        content_match = re.match(r'(left:\s*)(.*)', original_line, re.DOTALL)
        if content_match:
            new_content = f"{content_match.group(1)}{content_match.group(2)} {urgency_text}"
            lines[insert_idx] = new_content
    
    return '\n'.join(lines)


def augment_dialogue_emotional(dialogue: str) -> str:
    """策略3：情感操纵 - 加入情感化语言"""
    lines = dialogue.strip().split('\n')
    
    # 找到最后一个left说话的位置
    left_indices = [i for i, line in enumerate(lines) if line.strip().startswith('left:')]
    
    if left_indices:
        # 在倒数第二个left发言中添加情感化语言
        target_idx = left_indices[-1] if len(left_indices) == 1 else left_indices[-2]
        emotional_text = get_emotional_addition()
        
        original_line = lines[target_idx]
        content_match = re.match(r'(left:\s*)(.*)', original_line, re.DOTALL)
        if content_match:
            new_content = f"{content_match.group(1)}{emotional_text} {content_match.group(2)}"
            lines[target_idx] = new_content
    
    return '\n'.join(lines)


def augment_dialogue_combined(dialogue: str) -> str:
    """组合策略：同时应用三种增强策略"""
    result = augment_dialogue_credibility(dialogue)
    result = augment_dialogue_urgency(result)
    result = augment_dialogue_emotional(result)
    return result


def build_augmented_datasets(test_df: pd.DataFrame, sample_size: int = None) -> dict:
    """
    构建增强测试集
    
    Args:
        test_df: 原始测试集
        sample_size: 采样数量，None表示全量
    
    Returns:
        dict: 包含各增强策略子测试集的字典
    """
    if sample_size and sample_size < len(test_df):
        df = test_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        df = test_df.copy()
    
    results = {}
    
    # 原始测试集
    results['original'] = df.copy()
    
    # 策略1：建立信任测试集
    df_credibility = df.copy()
    df_credibility['specific_dialogue_content'] = df_credibility['specific_dialogue_content'].apply(
        augment_dialogue_credibility
    )
    df_credibility['augmentation_strategy'] = 'Credibility_Building'
    results['credibility'] = df_credibility
    
    # 策略2：紧迫感测试集
    df_urgency = df.copy()
    df_urgency['specific_dialogue_content'] = df_urgency['specific_dialogue_content'].apply(
        augment_dialogue_urgency
    )
    df_urgency['augmentation_strategy'] = 'Urgency_Creating'
    results['urgency'] = df_urgency
    
    # 策略3：情感操纵测试集
    df_emotional = df.copy()
    df_emotional['specific_dialogue_content'] = df_emotional['specific_dialogue_content'].apply(
        augment_dialogue_emotional
    )
    df_emotional['augmentation_strategy'] = 'Emotional_Appeal'
    results['emotional'] = df_emotional
    
    # 组合策略测试集
    df_combined = df.copy()
    df_combined['specific_dialogue_content'] = df_combined['specific_dialogue_content'].apply(
        augment_dialogue_combined
    )
    df_combined['augmentation_strategy'] = 'Combined_Strategy'
    results['combined'] = df_combined
    
    return results


def print_comparison(original_dialogue: str, augmented_dialogue: str, strategy: str):
    """打印对比结果"""
    print(f"\n{'='*60}")
    print(f"策略：{strategy}")
    print(f"{'='*60}")
    print("\n【原始对话】")
    print(original_dialogue[:500] + "..." if len(original_dialogue) > 500 else original_dialogue)
    print("\n【增强后对话】")
    print(augmented_dialogue[:700] + "..." if len(augmented_dialogue) > 700 else augmented_dialogue)
    print()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("正在加载数据集...")
    test_df = pd.read_csv(r'e:/个人成长/大三下/机器学习/实验2/测试集结果.csv')
    print(f"测试集大小：{len(test_df)} 条")
    print(f"列名：{test_df.columns.tolist()}")
    print(f"\n诈骗类型分布：")
    print(test_df['fraud_type'].value_counts())
    
    print("\n\n========== 增强示例 ==========\n")
    
    # 选择一个诈骗样本作为示例
    sample = test_df[test_df['is_fraud'] == True].iloc[0]
    original = sample['specific_dialogue_content']
    
    # 策略1示例
    aug1 = augment_dialogue_credibility(original)
    print_comparison(original, aug1, "建立信任（Credibility Building）")
    
    # 策略2示例
    aug2 = augment_dialogue_urgency(original)
    print_comparison(original, aug2, "制造紧迫感（Urgency Creating）")
    
    # 策略3示例
    aug3 = augment_dialogue_emotional(original)
    print_comparison(original, aug3, "情感操纵（Emotional Appeal）")
    
    # 组合策略示例
    aug4 = augment_dialogue_combined(original)
    print_comparison(original, aug4, "组合策略（Combined Strategy）")
    
    # 构建所有增强数据集
    print("\n\n========== 构建增强数据集 ==========\n")
    augmented = build_augmented_datasets(test_df)
    
    # 保存各增强数据集
    output_dir = r'e:/个人成长/大三下/机器学习/实验2'
    for name, df in augmented.items():
        output_path = f"{output_dir}/增强测试集_{name}.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"已保存：增强测试集_{name}.csv ({len(df)} 条)")
    
    print("\n数据集构建完成！")
    print(f"\n各数据集说明：")
    print(f"  original   - 原始测试集（{len(augmented['original'])} 条）")
    print(f"  credibility - 建立信任策略增强（{len(augmented['credibility'])} 条）")
    print(f"  urgency    - 制造紧迫感策略增强（{len(augmented['urgency'])} 条）")
    print(f"  emotional  - 情感操纵策略增强（{len(augmented['emotional'])} 条）")
    print(f"  combined   - 组合策略增强（{len(augmented['combined'])} 条）")
