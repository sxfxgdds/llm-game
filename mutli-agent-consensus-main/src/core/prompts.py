import os
from typing import List, Dict, Any

def get_pd_reputation_prompt(
    agent_reputation: float,
    partner_reputation: float,
    history: List[Dict[str, Any]],
    history_length: int = 5
) -> str:
    """
    生成囚徒困境博弈提示词，包含声誉感知与历史记忆。
    
    Args:
        agent_reputation: 智能体自身的声誉得分 (0-1)
        partner_reputation: 对手的声誉得分 (0-1)
        history: 历史博弈记录
        history_length: 提示词中保留的历史深度
    """
    
    # 1. 基础规则说明 (Payoff Matrix)
    # 定义：T(Temptation)=5, R(Reward)=3, P(Punishment)=1, S(Sucker)=0
    base_prompt = (
        "你正在参加一场多次回合的社交博弈游戏。在每一轮中，你需要从两个选项中选择一个：\n"
        "- 选择 'C' (合作): 展现出团队协作精神。\n"
        "- 选择 'D' (背叛): 追求个人短期利益。\n\n"
        "收益规则如下：\n"
        "1. 如果双方都选 C：你们各获得 3 分 (双赢)。\n"
        "2. 如果你选 D 而对手选 C：你获得 5 分 (诱惑奖励)，对手得 0 分。\n"
        "3. 如果你选 C 而对手选 D：你获得 0 分 (被坑损失)，对手得 5 分。\n"
        "4. 如果双方都选 D：你们各获得 1 分 (两败俱伤)。\n\n"
    )

    # 2. 注入声誉信息 (Reputation Context)
    # 这里的描述会引导 LLM 注意到声誉与其行为的关联
    reputation_context = (
        "【当前社会声誉状态】\n"
        f"- 你的当前声誉值：{agent_reputation:.2f} / 1.00\n"
        f"- 你的对手声誉值：{partner_reputation:.2f} / 1.00\n"
        "（提示：声誉越高代表该个体在过去表现得更倾向于合作；声誉越低代表其更倾向于背叛。）\n\n"
    )

    # 3. 注入历史记忆 (History Trace)
    history_context = "【最近博弈历史】\n"
    if not history:
        history_context += "尚无历史记录，这是第一轮博弈。\n"
    else:
        # 只取最近几轮，防止 Prompt 过长超出窗口
        recent_trials = history[-history_length:]
        for i, trial in enumerate(recent_trials):
            # 提取历史动作和收益
            p_action = trial.get("partner_choice", "未知")
            my_reward = trial.get("reward", 0)
            history_context += f"- 往期第 {i+1} 轮：对手选择了 [{p_action}]，你获得了 {my_reward} 分。\n"
    
    # 4. 决策指令 (Action Instruction)
    instruction = (
        "\n你的目标是：在长期的社会互动中，通过策略性的选择来最大化你的总收益。\n"
        "请结合对方的声誉背景和过往表现，深思熟虑后做出本轮决策。\n"
        "只需输出一个字母 'C' 或 'D'，不要有任何多余的解释。"
    )

    return base_prompt + reputation_context + history_context + instruction