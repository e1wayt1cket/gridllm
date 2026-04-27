def get_insight(self, summary: dict) -> str:
    # 提取关键数据
    scenario = summary.get('scenario', '未知')
    welfare = summary.get('welfare_da', '?')
    re_rate = summary.get('re_rate', '?')
    load_sat = summary.get('load_sat', '?')
    avg_lmp = summary.get('avg_lmp', '?')
    min_lmp = summary.get('min_lmp', '?')
    max_lmp = summary.get('max_lmp', '?')
    avg_soc = summary.get('avg_soc', '?')
    min_soc = summary.get('min_soc', '?')
    max_soc = summary.get('max_soc', '?')
    total_ch = summary.get('total_ch', '?')
    total_dis = summary.get('total_dis', '?')
    total_buy = summary.get('total_buy', '?')
    total_sell = summary.get('total_sell', '?')
    storage_active = summary.get('storage_active', '?')

    prompt = f"""你是一个电力市场分析师。根据以下仿真数据，用中文简要解释曲线变化的原因，并给出1-2条针对性的调整建议。总字数控制在200字以内。

仿真场景：{scenario}
社会福利：{welfare} ¥，可再生能源消纳率：{re_rate}%，负荷满足率：{load_sat}%
节点电价（¥/MWh）：平均 {avg_lmp}，最低 {min_lmp}，最高 {max_lmp}
储能 SOC（%）：平均 {avg_soc}，最低 {min_soc}，最高 {max_soc}
储能总充电量：{total_ch} MWh，总放电量：{total_dis} MWh
总购电量：{total_buy} MWh，总售电量：{total_sell} MWh
储能是否动作：{storage_active}

请输出分析（原因 + 建议，≤200字）："""

    response = self._call_ollama(prompt, max_tokens=256)
    if response:
        return response.strip()[:200]
    # 降级规则分析
    return self._rule_insight(summary)

def _rule_insight(self, summary):
    # 基于规则的简单分析，用于 AI 不可用时
    re_rate = float(summary.get('re_rate', 0))
    lmp = float(summary.get('avg_lmp', 0))
    active = summary.get('storage_active', '否')
    if re_rate > 95:
        return "可再生消纳率很高，午间光伏过剩导致电价下降，部分节点出现负值。建议增加储能容量或引入灵活负荷以消纳更多绿电。"
    elif lmp < 0:
        return "节点电价整体为负，说明供远大于求，可能需要下调可再生出力或提高储能充电功率。"
    elif active == '否':
        return "储能没有动作，可能因为电价波动较小或套利空间不足。建议扩大电价峰谷差或调整储能报价策略。"
    return "系统运行平稳，各曲线符合预期，可进一步优化报价以提升社会福利。"