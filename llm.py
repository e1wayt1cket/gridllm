# llm.py
"""
LLM Advisor - natural language scenario parsing and AI insights.
Supports local Ollama models. Falls back to defaults if AI is unavailable.
"""

import json
import numpy as np
import requests
from typing import List, Tuple


class LLMAdvisor:
    def __init__(self, api_url="http://localhost:11434", model="gemma4:e2b"):
        self.api_url = api_url
        self.model = model

    def _call_ollama(self, prompt, max_tokens=256):
        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.1
                    }
                },
                timeout=60
            )
            if response.status_code != 200:
                print(f"Ollama returned {response.status_code}: {response.text}")
                return ""
            return response.json().get("response", "")
        except Exception as e:
            print(f"Ollama call failed: {e}")
            return ""

    # ------------------------------------------------------------------
    # Natural language -> simulation config (enriched schema)
    # ------------------------------------------------------------------
    def parse_natural_language_to_config(self, user_input: str) -> dict:
        """
        Convert natural language to a rich simulation configuration dict.
        Supports arbitrary scenario descriptions beyond the 6 presets.
        Falls back to baseline defaults on failure.
        """
        prompt = f"""You are a power market simulation config assistant. Based on user input, output ONLY a JSON object with this schema:

{{
  "base_scenario": "baseline|high_re|peak_load|congestion|re_ramp_drop|re_ramp_surge|custom",
  "description": "short summary of the scenario",
  "global_params": {{
    "T": 96,
    "strategy": "random|best_response",
    "opf_mode": "lindistflow|dc",
    "load_factor": 1.0,
    "line_capacity_factor": 1.0,
    "lambda_carbon": 100.0,
    "lambda_re": 100.0,
    "lambda_curtail": 30.0,
    "enable_multi_objective": true
  }},
  "agent_modifications": [
    {{
      "action": "add|modify|remove|add_storage",
      "bus": 20,
      "agent_type": "solar_farm|wind_farm|storage_only|load_only|prosumer",
      "params": {{
        "name": "Custom_Solar_Bus20",
        "capacity_mw": 5.0,
        "load_mw": 0.0,
        "storage_mwh": 0.0,
        "storage_power_mw": 0.0,
        "bid_value": 350.0,
        "offer_cost": 0.0
      }}
    }}
  ],
  "custom_loads": [
    {{"buses": [10, 11, 12], "factor": 2.0, "description": "double load on commercial buses"}}
  ],
  "renewable_params": {{
    "pv_factor": 1.0,
    "wind_factor": 1.0
  }}
}}

Rules:
- bus must be 0-32 (IEEE 33 bus, 0-indexed)
- "add" action: create new generation resource at the bus
- "modify" action: adjust existing agent at the bus (merge capacity)
- "remove" action: delete agent at the bus
- "add_storage" action: add battery storage to agent at the bus
- If user says "add 5MW solar at bus 20", use action="add", bus=20, agent_type="solar_farm", capacity_mw=5.0
- If user says "double load on buses 10-15", use custom_loads with factor=2.0
- If user says "increase carbon tax to 200", set lambda_carbon=200 in global_params
- For unspecified fields, use the defaults shown above
- Output ONLY the JSON, no markdown, no explanation

User input: {user_input}
"""
        response = self._call_ollama(prompt, max_tokens=512)
        if response:
            try:
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0].strip()
                else:
                    json_str = response.strip()
                config = json.loads(json_str)
                if "base_scenario" not in config:
                    raise ValueError("missing base_scenario")
                return self._validate_and_fill_config(config)
            except Exception as e:
                print(f"AI JSON parse failed: {e}")

        print("AI parse failed, returning default config")
        return self._default_config()

    def _default_config(self) -> dict:
        return {
            "base_scenario": "baseline",
            "description": "default baseline",
            "global_params": {
                "T": 96, "strategy": "random", "opf_mode": "lindistflow",
                "load_factor": 1.0, "line_capacity_factor": 1.0,
                "lambda_carbon": 100.0, "lambda_re": 100.0,
                "lambda_curtail": 30.0, "enable_multi_objective": True,
            },
            "agent_modifications": [],
            "custom_loads": [],
            "renewable_params": {"pv_factor": 1.0, "wind_factor": 1.0},
        }

    def _validate_and_fill_config(self, config: dict) -> dict:
        gp_defaults = {
            "T": 96, "strategy": "random", "opf_mode": "lindistflow",
            "load_factor": 1.0, "line_capacity_factor": 1.0,
            "lambda_carbon": 100.0, "lambda_re": 100.0,
            "lambda_curtail": 30.0, "enable_multi_objective": True,
        }
        config.setdefault("global_params", {})
        for k, v in gp_defaults.items():
            config["global_params"].setdefault(k, v)
        config.setdefault("agent_modifications", [])
        config.setdefault("custom_loads", [])
        config.setdefault("renewable_params", {"pv_factor": 1.0, "wind_factor": 1.0})
        config.setdefault("description", "")
        return config

    # ------------------------------------------------------------------
    # Apply LLM config to build MarketConfig and modify agents
    # ------------------------------------------------------------------
    def apply_llm_config_to_agents(self, llm_config: dict, base_agents: List, T: int) -> Tuple[List, object]:
        """
        Apply LLM-parsed configuration to create/modify agents.
        Returns (modified_agents, market_config).
        Agent merge policy: same-bus resources are merged into existing agent.
        """
        from models import MarketConfig, Agent, StorageSpec

        gp = llm_config.get("global_params", {})
        config = MarketConfig(
            opf_mode=gp.get("opf_mode", "lindistflow"),
            lambda_re=float(gp.get("lambda_re", 100.0)),
            lambda_curtail=float(gp.get("lambda_curtail", 30.0)),
            lambda_carbon=float(gp.get("lambda_carbon", 100.0)),
            enable_multi_objective=bool(gp.get("enable_multi_objective", True)),
            line_capacity_multiplier=3.0 * float(gp.get("line_capacity_factor", 1.0)),
        )

        agents = list(base_agents)

        # Global load scaling
        lf = float(gp.get("load_factor", 1.0))
        if lf != 1.0:
            for a in agents:
                a.load_forecast = a.load_forecast * lf
                a.load_real = a.load_real * lf

        # Renewable scaling
        rp = llm_config.get("renewable_params", {})
        pv_factor = float(rp.get("pv_factor", 1.0))
        wind_factor = float(rp.get("wind_factor", 1.0))
        if pv_factor != 1.0 or wind_factor != 1.0:
            for a in agents:
                if a.is_prosumer:
                    a.pv_forecast = a.pv_forecast * pv_factor
                    a.pv_real = a.pv_real * pv_factor
                if a.has_wind:
                    a.wind_forecast = a.wind_forecast * wind_factor
                    a.wind_real = a.wind_real * wind_factor

        # Custom per-bus load scaling
        for cl in llm_config.get("custom_loads", []):
            buses = cl.get("buses", [])
            factor = float(cl.get("factor", 1.0))
            for a in agents:
                if a.bus in buses:
                    a.load_forecast = a.load_forecast * factor
                    a.load_real = a.load_real * factor

        # Agent modifications
        for mod in llm_config.get("agent_modifications", []):
            action = mod.get("action", "")
            bus = int(mod.get("bus", 0))
            bus = max(0, min(32, bus))
            agent_type = mod.get("agent_type", "solar_farm")
            params = mod.get("params", {})

            if action == "remove":
                agents = [a for a in agents if a.bus != bus]
                continue

            if action == "add":
                self._add_agent(agents, bus, agent_type, params, T)
            elif action == "modify":
                self._merge_to_bus(agents, bus, agent_type, params, T)
            elif action == "add_storage":
                self._add_storage_to_bus(agents, bus, params)

        return agents, config

    def _add_agent(self, agents: List, bus: int, agent_type: str, params: dict, T: int):
        from models import Agent, StorageSpec
        name = params.get("name", f"{agent_type}_Bus{bus}")
        capacity = float(params.get("capacity_mw", 0.0))
        load_mw = float(params.get("load_mw", 0.0))
        hours = np.arange(T)

        pv_forecast = np.zeros(T)
        pv_real = np.zeros(T)
        wind_forecast = np.zeros(T)
        wind_real = np.zeros(T)
        is_prosumer = False
        has_wind = False

        if agent_type == "solar_farm":
            pv_forecast = np.clip(np.sin((hours - 6) / 24 * 2 * np.pi), 0, None) * capacity
            pv_real = pv_forecast * 0.95
            is_prosumer = True
        elif agent_type == "wind_farm":
            np.random.seed(bus + 42)
            base = 0.3 + 0.2 * np.sin(np.arange(T) * 2 * np.pi / 40)
            wind_forecast = np.clip(base + np.random.normal(0, 0.15, T), 0, None) * capacity
            wind_real = wind_forecast * 0.9
            has_wind = True
            is_prosumer = True
        elif agent_type == "prosumer":
            pv_forecast = np.clip(np.sin((hours - 6) / 24 * 2 * np.pi), 0, None) * capacity * 0.6
            pv_real = pv_forecast * 0.95
            is_prosumer = True

        bid = float(params.get("bid_value", 350.0 if is_prosumer else 500.0))
        offer = float(params.get("offer_cost", 0.0 if is_prosumer else 999.0))

        storage = None
        storage_mwh = float(params.get("storage_mwh", 0.0))
        if storage_mwh > 0:
            storage = StorageSpec(
                e_max=storage_mwh,
                p_ch_max=float(params.get("storage_power_mw", storage_mwh / 5)),
                p_dis_max=float(params.get("storage_power_mw", storage_mwh / 5)),
                eta_ch=0.93, eta_dis=0.93, soc0=0.5, soc_min=0.0, soc_max=1.0,
            )

        agent = Agent(
            name=name, bus=bus, is_prosumer=is_prosumer,
            load_forecast=np.full(T, load_mw), load_real=np.full(T, load_mw),
            pv_forecast=pv_forecast, pv_real=pv_real,
            wind_forecast=wind_forecast if has_wind else None,
            wind_real=wind_real if has_wind else None,
            bid_value=bid, offer_cost=offer,
            storage=storage, load_type=agent_type,
        )
        agents.append(agent)

    def _merge_to_bus(self, agents: List, bus: int, agent_type: str, params: dict, T: int):
        from models import StorageSpec
        capacity = float(params.get("capacity_mw", 0.0))
        target = next((a for a in agents if a.bus == bus), None)
        if target is None:
            self._add_agent(agents, bus, agent_type, params, T)
            return
        hours = np.arange(T)
        if agent_type in ("solar_farm", "prosumer"):
            new_pv = np.clip(np.sin((hours - 6) / 24 * 2 * np.pi), 0, None) * capacity
            target.pv_forecast = target.pv_forecast + new_pv
            target.pv_real = target.pv_real + new_pv * 0.95
            target.is_prosumer = True
        if agent_type == "wind_farm":
            np.random.seed(bus + 42)
            base = 0.3 + 0.2 * np.sin(np.arange(T) * 2 * np.pi / 40)
            new_wind = np.clip(base + np.random.normal(0, 0.15, T), 0, None) * capacity
            if target.has_wind:
                target.wind_forecast = target.wind_forecast + new_wind
                target.wind_real = target.wind_real + new_wind * 0.9
            else:
                target.wind_forecast = new_wind
                target.wind_real = new_wind * 0.9
            target.is_prosumer = True
        storage_mwh = float(params.get("storage_mwh", 0.0))
        if storage_mwh > 0:
            if target.storage is None:
                target.storage = StorageSpec(
                    e_max=storage_mwh,
                    p_ch_max=float(params.get("storage_power_mw", storage_mwh / 5)),
                    p_dis_max=float(params.get("storage_power_mw", storage_mwh / 5)),
                    eta_ch=0.93, eta_dis=0.93, soc0=0.5, soc_min=0.0, soc_max=1.0,
                )
            else:
                target.storage.e_max += storage_mwh
                target.storage.p_ch_max += float(params.get("storage_power_mw", storage_mwh / 5))
                target.storage.p_dis_max += float(params.get("storage_power_mw", storage_mwh / 5))

    def _add_storage_to_bus(self, agents: List, bus: int, params: dict):
        from models import StorageSpec
        target = next((a for a in agents if a.bus == bus), None)
        if target is None:
            return
        storage_mwh = float(params.get("storage_mwh", 10.0))
        storage_power = float(params.get("storage_power_mw", 2.0))
        if target.storage is None:
            target.storage = StorageSpec(
                e_max=storage_mwh, p_ch_max=storage_power, p_dis_max=storage_power,
                eta_ch=0.93, eta_dis=0.93, soc0=0.5, soc_min=0.0, soc_max=1.0,
            )
        else:
            target.storage.e_max += storage_mwh
            target.storage.p_ch_max += storage_power
            target.storage.p_dis_max += storage_power

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
碳排放：总量 {summary.get('carbon_emissions', '?')} tCO2，碳强度 {summary.get('carbon_intensity', '?')} tCO2/MWh
弃电量：{summary.get('curtailment', '?')} MWh
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