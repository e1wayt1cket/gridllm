# llm.py
"""
LLM Advisor - 纯 AI 决策版本（无关键词降级）
支持本地 Ollama 模型进行自然语言解析和曲线解释。
若 AI 不可用，直接返回默认配置。
"""

import json
import requests

class LLMAdvisor:
    def __init__(self, api_url="http://localhost:11434", model="gemma2:2b"):
        self.api_url = api_url
        self.model = model

    def _call_ollama(self, prompt, max_tokens=256):
        """调用 Ollama API，返回文本或空字符串"""
        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.1   # 降低随机性
                    }
                },
                timeout=60
            )
            if response.status_code != 200:
                print(f"Ollama 返回 {response.status_code}: {response.text}")
                return ""
            return response.json().get("response", "")
        except Exception as e:
            print(f"Ollama 调用失败: {e}")
            return ""

    # ------------------------------------------------------------------
    # 自然语言 → 仿真配置（纯 AI 版，无关键词降级）
    # ------------------------------------------------------------------
    def parse_natural_language_to_config(self, user_input: str) -> dict:
        """
        使用 Ollama 将自然语言转换为仿真配置字典。
        如果 AI 调用失败或返回内容无法解析，直接返回默认基线配置。
        """
        prompt = f"""你是一个电力市场仿真配置助手。根据用户输入输出一个 JSON 对象，包含以下字段：
- scenario_type: 场景类型，可选值：baseline, high_re, peak_load, congestion, re_ramp_drop, re_ramp_surge
- parameters: 一个字典，包含：
    - load_factor: 负荷系数（浮点数，1.0 为正常）
    - re_factor: 可再生能源系数（浮点数，1.0 为正常）
    - line_capacity_factor: 线路容量系数（浮点数，1.0 为正常，<1 表示阻塞）
    - T: 时段数（整数，默认 96）
    - strategy: 报价策略，可选值：random, best_response
    - use_ac_opf: 是否使用交流 OPF（布尔值，默认 false）

请根据用户意图推断合适的参数。如果用户未提及某个参数，使用以下默认值：
"scenario_type": "baseline",
"load_factor": 1.0, "re_factor": 1.0, "line_capacity_factor": 1.0,
"T": 96, "strategy": "random", "use_ac_opf": false

只输出 JSON，不要包含任何其他文字。如果无法理解用户意图，返回默认 JSON。

用户输入：{user_input}
"""
        response = self._call_ollama(prompt, max_tokens=256)
        if response:
            try:
                # 提取可能被 markdown 代码块包裹的 JSON
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0].strip()
                else:
                    json_str = response.strip()
                config = json.loads(json_str)
                if not isinstance(config.get("scenario_type"), str) or "parameters" not in config:
                    raise ValueError("缺少必要字段")
                # 确保 parameters 中包含所有键
                defaults = {"load_factor": 1.0, "re_factor": 1.0, "line_capacity_factor": 1.0,
                            "T": 96, "strategy": "random", "use_ac_opf": False}
                for k, v in defaults.items():
                    config["parameters"].setdefault(k, v)
                return config
            except Exception as e:
                print(f"AI JSON 解析失败: {e}")

        # 若 AI 完全失败，直接返回默认配置（无关键词降级）
        print("AI 解析失败，返回默认配置")
        return {
            "scenario_type": "baseline",
            "parameters": {
                "load_factor": 1.0,
                "re_factor": 1.0,
                "line_capacity_factor": 1.0,
                "T": 96,
                "strategy": "random",
                "use_ac_opf": False
            }
        }

    # ------------------------------------------------------------------
    # 曲线解释与建议（≤200字）
    # ------------------------------------------------------------------
    def get_insight(self, summary: dict) -> str:
        """
        根据仿真结果的关键指标，调用 Ollama 生成简短解释和建议。
        若 AI 不可用，使用内置规则生成分析。
        """
        prompt = f"""你是一个电力市场分析师。根据以下仿真数据，用中文简要解释曲线变化的原因，并给出1-2条针对性的调整建议。总字数控制在200字以内。

仿真场景：{summary.get('scenario', '未知')}
社会福利：{summary.get('welfare_da', '?')} ¥，可再生能源消纳率：{summary.get('re_rate', '?')}%，负荷满足率：{summary.get('load_sat', '?')}%
节点电价（¥/MWh）：平均 {summary.get('avg_lmp', '?')}，最低 {summary.get('min_lmp', '?')}，最高 {summary.get('max_lmp', '?')}
储能 SOC（%）：平均 {summary.get('avg_soc', '?')}，最低 {summary.get('min_soc', '?')}，最高 {summary.get('max_soc', '?')}
储能总充电量：{summary.get('total_ch', '?')} MWh，总放电量：{summary.get('total_dis', '?')} MWh
总购电量：{summary.get('total_buy', '?')} MWh，总售电量：{summary.get('total_sell', '?')} MWh
储能是否动作：{summary.get('storage_active', '?')}

请输出分析（原因 + 建议，≤200字）："""

        response = self._call_ollama(prompt, max_tokens=200)
        if response:
            return response.strip()[:200]

        # AI 不可用时的规则兜底
        return self._rule_insight(summary)

    def _rule_insight(self, summary: dict) -> str:
        """基于简单规则的分析（Ollama 不可用时的兜底）"""
        try:
            re_rate = float(summary.get('re_rate', 0))
            avg_lmp = float(summary.get('avg_lmp', 0))
            active = summary.get('storage_active', '否')
            load_sat = float(summary.get('load_sat', 100))
        except:
            return "系统运行平稳，可微调报价参数以提升社会福利。"

        if load_sat < 90:
            return "负荷满足率偏低，线路可能严重阻塞或发电能力不足。建议加强网架或增加分布式电源。"
        if re_rate > 95:
            return "可再生能源消纳率很高，午间光伏过剩导致电价下跌。可增加储能容量或引入灵活负荷消纳更多绿电。"
        if avg_lmp < 0:
            return "节点电价整体为负，供远大于求。建议降低可再生出力或提高储能充电功率以吸收过剩。"
        if active == '否':
            return "储能未动作，电价波动可能较小或套利空间不足。可扩大电价峰谷差或调整储能报价策略。"
        return "系统运行平稳，各曲线符合预期。可进一步优化报价提升社会福利。"