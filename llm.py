# llm.py
"""
LLM Advisor - natural language simulation config and AI insights.
Supports local Ollama models. Covers all configurable parameters:
  - MarketConfig fields (20+)
  - defaults.yaml overrides (40+)
  - scenario.yaml overrides
Falls back to keyword-based parsing when Ollama is unavailable.
"""

import dataclasses
import json
import re
import numpy as np
import requests
from typing import Any, Dict, List, Optional, Tuple

from models import MarketConfig
from grid import pv_profile, wind_profile


class LLMAdvisor:
    def __init__(self, api_url="http://localhost:11434", model="qwen2.5:7b"):
        self.api_url = api_url
        self.model = model

    # ------------------------------------------------------------------
    # LLM transport
    # ------------------------------------------------------------------
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
    # Current values catalog for prompt injection
    # ------------------------------------------------------------------
    def _collect_current_values(self) -> str:
        """Walk config_loader defaults + MarketConfig fields to build a
        compact reference block of all tunable parameters."""
        from config_loader import load_defaults, load_scenarios

        lines = ["CURRENT CONFIGURATION VALUES"]

        # defaults.yaml
        defaults = load_defaults()
        for top_key in ["network", "profiles", "price_curve"]:
            if top_key in defaults:
                self._flatten_dict(defaults[top_key], top_key, lines)

        for lt_name in sorted(defaults.get("load_types", {}).keys()):
            lt = defaults["load_types"][lt_name]
            for k, v in lt.items():
                if isinstance(v, list) and len(v) > 6:
                    v = f"[{v[0]},{v[1]},...,{v[-1]}]"
                lines.append(f"  load_types.{lt_name}.{k} = {v}")

        for ps_name in sorted(defaults.get("prosumers", {}).keys()):
            ps = defaults["prosumers"][ps_name]
            for k, v in ps.items():
                if k == "storage":
                    self._flatten_dict(v, f"prosumers.{ps_name}.storage", lines)
                elif isinstance(v, list):
                    lines.append(f"  prosumers.{ps_name}.{k} = {v}")
                else:
                    lines.append(f"  prosumers.{ps_name}.{k} = {v}")

        # scenarios
        scenarios = load_scenarios()
        for s_name in sorted(scenarios.get("scenarios", {}).keys()):
            sc = scenarios["scenarios"][s_name]
            items = []
            for k, v in sc.items():
                if k == "description":
                    continue
                items.append(f"{k}={v}")
            if items:
                lines.append(f"  SCENARIO: {s_name}  {', '.join(items)}")

        # MarketConfig defaults
        lines.append("  --- MARKET CONFIG FIELDS ---")
        for f in dataclasses.fields(MarketConfig):
            if f.name in ("verbose", "use_ac_opf", "opf_tolerance",
                          "opf_max_iter", "base_mva", "base_kv"):
                continue  # debug/internal fields
            val = f.default if f.default is not dataclasses.MISSING else "?"
            if f.name in ("bid_mult_range", "offer_adder_range"):
                val = list(val)
            lines.append(f"  {f.name} = {val}")

        return "\n".join(lines)

    @staticmethod
    def _flatten_dict(d: dict, prefix: str, lines: list) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                LLMAdvisor._flatten_dict(v, f"{prefix}.{k}", lines)
            elif isinstance(v, list):
                lines.append(f"  {prefix}.{k} = {v}")
            else:
                lines.append(f"  {prefix}.{k} = {v}")

    # ------------------------------------------------------------------
    # Natural language -> simulation config (enriched schema)
    # ------------------------------------------------------------------
    def parse_natural_language_to_config(self, user_input: str) -> dict:
        # Guard against empty/whitespace input that causes LLM hallucination
        if not user_input or not user_input.strip():
            return self._default_config()

        prompt = f"""你是一个电力市场仿真配置助手。根据用户输入输出JSON，只输出JSON，不要任何解释。

=== 可用场景 ===
baseline | high_re | peak_load | congestion | re_ramp_drop | re_ramp_surge

=== global_params（大部分参数放这里）===
字段名 — 含义（默认值）
  carbon_cap_tco2 — 碳排放上限tCO2(200)
  re_min_rate — 可再生最低消纳率%(95)
  lambda_carbon — 碳价格权重(50)
  lambda_re — 可再生激励权重(50)
  lambda_curtail — 弃电惩罚权重(15)
  load_factor — 全局负荷系数(1.0)
  line_capacity_factor — 线路容量系数(1.0)
  penalty_unserved — 未供应惩罚(800)
  opf_mode — OPF模式: lindistflow或dc
  pv_factor — 光伏出力系数(1.0)
  wind_factor — 风电出力系数(1.0)

=== defaults_overrides（仅config_loader点号路径，一般不碰）===
  price_curve.base — 基准电价(300)
  profiles.pv.amplitude — 光伏幅度(0.8)
  prosumers.residential.storage.capacity_factor — 储能容量系数(1.0)

=== agent_modifications ===
  操作: add/modify/remove/add_storage
  bus: 0-32
  agent_type: solar_farm/wind_farm/storage_only/load_only/prosumer
  add参数: capacity_mw(容量MW), load_mw(负荷MW), bid_value, offer_cost
  add_storage参数: storage_mwh(储能容量MWh), storage_power_mw(充放电功率MW)
  add_storage示例params: {{"storage_mwh":10,"storage_power_mw":2}}

=== 示例 ===
输入: 碳上限80吨
输出: {{"base_scenario":"baseline","global_params":{{"carbon_cap_tco2":80}}}}

输入: 负荷翻倍
输出: {{"base_scenario":"baseline","global_params":{{"load_factor":2.0}}}}

输入: 碳上限80吨，负荷翻倍
输出: {{"base_scenario":"baseline","global_params":{{"carbon_cap_tco2":80,"load_factor":2.0}}}}

输入: 节点8加5MW光伏
输出: {{"base_scenario":"baseline","agent_modifications":[{{"action":"add","bus":8,"agent_type":"solar_farm","params":{{"capacity_mw":5}}}}]}}

输入: 高可再生场景风电3倍
输出: {{"base_scenario":"high_re","scenario_overrides":{{"high_re":{{"multipliers.wind":3.0}}}}}}

=== 严格规则 ===
1. 输出必须是纯JSON，一行，无```标记，无解释文字
2. base_scenario必填，默认"baseline"
3. 只输出用户明确修改的字段，不要输出默认值
4. load_factor/碳/可再生/惩罚等标量参数放global_params，不要放defaults_overrides
5. defaults_overrides仅用于用户明确提到"基准电价"等config_loader路径时

用户输入: {user_input}
"""
        response = self._call_ollama(prompt, max_tokens=768)
        if response:
            try:
                config = self._extract_json(response)
                config.setdefault("base_scenario", "baseline")
                return self._validate_and_fill_config(config)
            except Exception as e:
                print(f"LLM JSON parse failed: {e}")

        print("LLM unavailable, trying rule-based fallback...")
        return self._rule_based_parse(user_input)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON object from LLM response (handles markdown fences)."""
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()
        return json.loads(text.strip())

    @staticmethod
    def _market_config_defaults() -> dict:
        """Extract default values from MarketConfig dataclass fields.
        Excludes internal/debug fields that users should not control via NL.
        """
        exclude = {"verbose", "use_ac_opf", "opf_tolerance",
                   "opf_max_iter", "base_mva", "base_kv"}
        result = {}
        for f in dataclasses.fields(MarketConfig):
            if f.name in exclude:
                continue
            val = f.default if f.default is not dataclasses.MISSING else None
            result[f.name] = val
        return result

    def _default_config(self) -> dict:
        mc = self._market_config_defaults()
        return {
            "base_scenario": "baseline",
            "description": "default baseline",
            "global_params": {
                "T": 96, "strategy": "random",
                "load_factor": 1.0, "line_capacity_factor": 1.0,
                **{k: v for k, v in mc.items() if v is not None},
            },
            "defaults_overrides": {},
            "scenario_overrides": {},
            "agent_modifications": [],
            "custom_loads": [],
            "renewable_params": {"pv_factor": 1.0, "wind_factor": 1.0},
        }

    def _validate_and_fill_config(self, config: dict) -> dict:
        mc = self._market_config_defaults()
        gp_defaults = {
            "T": 96, "strategy": "random",
            "load_factor": 1.0, "line_capacity_factor": 1.0,
            **{k: v for k, v in mc.items() if v is not None},
        }
        config.setdefault("global_params", {})
        for k, v in gp_defaults.items():
            config["global_params"].setdefault(k, v)
        config.setdefault("defaults_overrides", {})
        config.setdefault("scenario_overrides", {})
        config.setdefault("agent_modifications", [])
        config.setdefault("custom_loads", [])
        config.setdefault("renewable_params", {"pv_factor": 1.0, "wind_factor": 1.0})
        config.setdefault("description", "")
        return config

    # ------------------------------------------------------------------
    # Rule-based fallback parser (no LLM needed)
    # ------------------------------------------------------------------
    def _rule_based_parse(self, user_input: str) -> dict:
        """Keyword/regex fallback when Ollama is unavailable.
        Handles common patterns: set X to Y, double/half X, X by N times.
        """
        result = self._default_config()
        text = user_input.lower()
        modified = False

        # Numerical extractors
        def find_num(pattern, text, default=None):
            m = re.search(pattern, text)
            return float(m.group(1)) if m else default

        def find_multiplier(text):
            # "double" = 2, "triple" = 3, "half" = 0.5, "N倍" = N, "Nx" = N
            m = re.search(r'(\d+\.?\d*)\s*(倍|x|times|×)', text)
            if m: return float(m.group(1))
            if any(w in text for w in ['double', 'doubled', '翻倍', '加倍']): return 2.0
            if any(w in text for w in ['triple', '三倍', 'tripled']): return 3.0
            if any(w in text for w in ['half', '一半', '减半']): return 0.5
            return None

        # --- MarketConfig direct mappings ---
        market_keywords = {
            r'carbon\s*(cap|limit|上限)|碳\s*(上限|排放|限额)':
                ("carbon_cap_tco2", "value"),
            r're[_（\s]*(?:min|rate|消纳)|可再生|renewable':
                ("re_min_rate", "value"),
            r'lambda\s*carbon|碳\s*(价格|成本|税|价)|carbon\s*(price|cost|tax)':
                ("lambda_carbon", "value"),
            r'lambda\s*re|可再生\s*(激励|奖励)|re\s*incentive':
                ("lambda_re", "value"),
            r'lambda\s*curtail|弃[风电]\s*(惩罚|处罚)|curtailment\s*penalty':
                ("lambda_curtail", "value"),
            r'load|负荷|负载|load\s*factor':
                ("load_factor", "value"),
            r'line\s*capacity|线路\s*容量':
                ("line_capacity_factor", "value"),
            r'penalty\s*unserved|未供应\s*惩罚':
                ("penalty_unserved", "value"),
            r'emission\s*factor|排放\s*因子':
                ("emission_factor_grid", "value"),
            r'storage\s*charge\s*discount|储能\s*充电\s*折扣':
                ("storage_charge_discount", "value"),
            r'storage\s*discharge\s*premium|储能\s*放电\s*溢价':
                ("storage_discharge_premium", "value"),
            r'opf\s*mode|opf\s*模式':
                ("opf_mode", "text"),
        }

        for pattern, (field, mode) in market_keywords.items():
            if re.search(pattern, text):
                mult = find_multiplier(text)
                if mult is not None:
                    cur = result["global_params"].get(field, 1.0)
                    if mode == "value":
                        result["global_params"][field] = cur * mult
                        modified = True
                else:
                    val = find_num(r'(\d+\.?\d*)', text)
                    if val is not None:
                        result["global_params"][field] = val
                        modified = True

        # --- defaults.yaml overrides ---
        defaults_keywords = {
            r'pv\s*amplitude|太阳能\s*幅度|光伏\s*幅度|solar\s*amplitude':
                ("profiles.pv.amplitude", "value"),
            r'wind\s*seed|风电\s*种子':
                ("profiles.wind.seed", "value"),
            r'load\s*forecast\s*noise|负荷\s*预测\s*噪声':
                ("profiles.load.forecast_noise_sigma", "value"),
            r'price\s*base|电价\s*(基|底)|基准\s*电价':
                ("price_curve.base", "value"),
            r'price\s*noise|电价\s*噪声':
                ("price_curve.noise_sigma", "value"),
            r'storage\s*capacity|储能\s*容量|电池\s*容量':
                ("prosumers.residential.storage.capacity_factor", "value"),
            r'pv\s*capacity\s*factor|光伏\s*容量\s*因子':
                ("prosumers.residential.pv_capacity_factor", "value"),
            r'wind\s*capacity\s*factor|风电\s*容量\s*因子':
                ("prosumers.industrial.wind_capacity_factor", "value"),
            r'storage\s*efficiency|储能\s*效率|电池\s*效率':
                ("prosumers.residential.storage.eta_ch", "value"),
        }

        for pattern, (path, mode) in defaults_keywords.items():
            if re.search(pattern, text):
                mult = find_multiplier(text)
                if mult is not None:
                    from config_loader import get_default
                    cur = get_default(path, 1.0)
                    result["defaults_overrides"][path] = cur * mult
                    modified = True
                else:
                    val = find_num(r'(\d+\.?\d+)', text)
                    if val is not None:
                        result["defaults_overrides"][path] = val
                        modified = True

        # Catch-all: "set X to Y", "X = Y"
        if not modified:
            m = re.search(r'(set|change|调整|设为|设置)\s+(\w[\w\s]+?)\s+(to|为|到|=)\s*(\d+\.?\d*)', text)
            if m:
                keyword = m.group(2).strip()
                val = float(m.group(4))
                for pat, (field, _) in market_keywords.items():
                    if re.search(pat, keyword):
                        result["global_params"][field] = val
                        modified = True
                        break
                if not modified:
                    result["description"] = f"rule-parsed: set {keyword} to {val}"

        # Agent modifications: detect add/modify/remove instructions
        agent_actions = self._parse_agent_modifications(user_input)
        if agent_actions:
            result["agent_modifications"] = agent_actions
            modified = True

        if modified:
            result["description"] = f"rule-parsed from: {user_input[:80]}"
        return result

    def _parse_agent_modifications(self, text: str) -> list:
        """Detect agent add/modify/remove patterns from NL text."""
        mods = []
        # Extract bus numbers from text
        bus_match = re.search(r'(?:bus|node|节点|母线)\s*(\d+)', text, re.IGNORECASE)
        bus = int(bus_match.group(1)) if bus_match else 0
        bus = max(0, min(32, bus))

        capacity_match = re.search(r'(\d+\.?\d*)\s*(?:MW|mw)', text)
        capacity = float(capacity_match.group(1)) if capacity_match else 5.0

        storage_match = re.search(r'(\d+\.?\d*)\s*(?:MWh|mwh)', text)
        storage_mwh = float(storage_match.group(1)) if storage_match else 0.0

        # Detect action
        if re.search(r'(add|添加|增加|新增)', text, re.IGNORECASE):
            if re.search(r'(storage|储能|电池)', text, re.IGNORECASE):
                mods.append({"action": "add_storage", "bus": bus, "agent_type": "storage_only",
                             "params": {"storage_mwh": storage_mwh or 10.0,
                                       "storage_power_mw": capacity or 2.0}})
            elif re.search(r'(solar|pv|光伏|太阳能)', text, re.IGNORECASE):
                mods.append({"action": "add", "bus": bus, "agent_type": "solar_farm",
                             "params": {"capacity_mw": capacity, "bid_value": 350.0}})
            elif re.search(r'(wind|风电|风力)', text, re.IGNORECASE):
                mods.append({"action": "add", "bus": bus, "agent_type": "wind_farm",
                             "params": {"capacity_mw": capacity, "bid_value": 350.0}})
            elif re.search(r'(load|负荷)', text, re.IGNORECASE):
                mods.append({"action": "add", "bus": bus, "agent_type": "load_only",
                             "params": {"load_mw": capacity}})
            else:
                mods.append({"action": "add", "bus": bus, "agent_type": "prosumer",
                             "params": {"capacity_mw": capacity}})
        elif re.search(r'(remove|删除|移除)', text, re.IGNORECASE):
            mods.append({"action": "remove", "bus": bus, "agent_type": "prosumer", "params": {}})
        elif re.search(r'(modify|修改|调整)', text, re.IGNORECASE):
            if re.search(r'(storage|储能|电池)', text, re.IGNORECASE):
                mods.append({"action": "add_storage", "bus": bus, "agent_type": "storage_only",
                             "params": {"storage_mwh": storage_mwh or 10.0}})
            else:
                mods.append({"action": "modify", "bus": bus, "agent_type": "prosumer",
                             "params": {"capacity_mw": capacity}})

        return mods

    # ------------------------------------------------------------------
    # Apply overrides to config_loader caches
    # ------------------------------------------------------------------
    def apply_defaults_overrides(self, overrides: Dict[str, Any]) -> None:
        """Write dotted-key overrides into the config_loader defaults cache.
        Call BEFORE get_scenario() so agents pick up the modified config.
        """
        if not overrides:
            return
        from config_loader import set_default
        for key_path, value in overrides.items():
            set_default(key_path, value)

    def apply_scenario_overrides(self, overrides: Dict[str, Dict[str, Any]]) -> None:
        """Write per-scenario parameter overrides into the config_loader cache."""
        if not overrides:
            return
        from config_loader import set_scenario_param
        for scenario_name, params in overrides.items():
            for key_path, value in params.items():
                set_scenario_param(scenario_name, key_path, value)

    # ------------------------------------------------------------------
    # Apply LLM config to build MarketConfig and modify agents
    # ------------------------------------------------------------------
    def apply_llm_config_to_agents(self, llm_config: dict, base_agents: List,
                                   T: int) -> Tuple[List, object]:
        from models import Agent, StorageSpec

        gp = llm_config.get("global_params", {})

        def _float(key, default, min_val=None, max_val=None):
            v = float(gp.get(key, default))
            if min_val is not None:
                v = max(min_val, v)
            if max_val is not None:
                v = min(max_val, v)
            return v

        def _bool(key, default):
            return bool(gp.get(key, default))

        def _int(key, default, min_val=None, max_val=None):
            v = int(gp.get(key, default))
            if min_val is not None:
                v = max(min_val, v)
            if max_val is not None:
                v = min(max_val, v)
            return v

        def _tuple_float(key, default):
            v = gp.get(key, default)
            if isinstance(v, (list, tuple)):
                return (float(v[0]), float(v[1]))
            return default

        config = MarketConfig(
            opf_mode=str(gp.get("opf_mode", "lindistflow")),
            lambda_re=_float("lambda_re", 50.0, min_val=0, max_val=10000),
            lambda_curtail=_float("lambda_curtail", 15.0, min_val=0, max_val=10000),
            lambda_carbon=_float("lambda_carbon", 50.0, min_val=0, max_val=10000),
            enable_multi_objective=_bool("enable_multi_objective", True),
            use_constraint_multi_obj=_bool("use_constraint_multi_obj", True),
            carbon_cap_tco2=max(0.0, float(gp.get("carbon_cap_tco2", 200.0))),
            re_min_rate=max(0.0, min(100.0, float(gp.get("re_min_rate", 95.0)))),
            line_capacity_multiplier=3.0 * _float("line_capacity_factor", 1.0, min_val=0.1, max_val=100),
            penalty_unserved=_float("penalty_unserved", 800.0, min_val=0, max_val=1e6),
            emission_factor_grid=_float("emission_factor_grid", 0.58, min_val=0, max_val=100),
            storage_charge_discount=_float("storage_charge_discount", 0.85, min_val=0, max_val=1),
            storage_discharge_premium=_float("storage_discharge_premium", 1.15, min_val=1, max_val=100),
            storage_soc_buffer=_float("storage_soc_buffer", 0.02, min_val=0, max_val=1),
            rt_horizon=_int("rt_horizon", 4, min_val=1, max_val=96),
            rt_step=_int("rt_step", 1, min_val=1, max_val=96),
            bid_mult_range=_tuple_float("bid_mult_range", (0.8, 1.2)),
            offer_adder_range=_tuple_float("offer_adder_range", (0.0, 50.0)),
            default_bid_mult=_float("default_bid_mult", 1.0),
            default_offer_adder=_float("default_offer_adder", 0.0),
        )

        agents = list(base_agents)

        # Global load scaling
        lf = max(0.1, min(10.0, _float("load_factor", 1.0)))
        if lf != 1.0:
            for a in agents:
                a.load_forecast = a.load_forecast * lf
                a.load_real = a.load_real * lf

        # Renewable scaling
        rp = llm_config.get("renewable_params", {})
        pv_factor = max(0.01, min(10.0, float(rp.get("pv_factor", 1.0))))
        wind_factor = max(0.01, min(10.0, float(rp.get("wind_factor", 1.0))))
        if pv_factor != 1.0 or wind_factor != 1.0:
            for a in agents:
                if a.is_prosumer:
                    a.pv_forecast = a.pv_forecast * pv_factor
                    a.pv_real = a.pv_real * pv_factor
                if a.has_wind:
                    a.wind_forecast = a.wind_forecast * wind_factor       # type: ignore
                    a.wind_real = a.wind_real * wind_factor               # type: ignore

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
            bus = max(0, min(32, int(mod.get("bus", 0))))
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

    # ------------------------------------------------------------------
    # Agent CRUD helpers (unchanged from original)
    # ------------------------------------------------------------------
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
            pv_forecast = pv_profile(hours, capacity)
            pv_real = pv_forecast * 0.95
            is_prosumer = True
        elif agent_type == "wind_farm":
            wind_forecast = wind_profile(hours, bus + 42) * capacity
            wind_real = wind_forecast * 0.9
            has_wind = True
            is_prosumer = True
        elif agent_type == "prosumer":
            pv_forecast = pv_profile(hours, capacity * 0.6)
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
            new_pv = pv_profile(hours, capacity)
            target.pv_forecast = target.pv_forecast + new_pv
            target.pv_real = target.pv_real + new_pv * 0.95
            target.is_prosumer = True
        if agent_type == "wind_farm":
            new_wind = wind_profile(hours, bus + 42) * capacity
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
        # Normalize LLM-invented field names to canonical ones
        storage_mwh = float(params.get("storage_mwh",
                            params.get("capacity_mwh",
                            params.get("capacity", 10.0))))
        storage_power = float(params.get("storage_power_mw",
                              params.get("power_mw",
                              params.get("p_ch_max", 2.0))))
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
    # Insight generation (unchanged)
    # ------------------------------------------------------------------
    def get_insight(self, summary: dict) -> str:
        prompt = f"""你是一个电力市场仿真分析顾问。用户可能是发电商、储能运营商、负荷聚合商、大用户、电网运营商或监管机构中的任何一方。基于仿真数据提供客观分析，按以下四段输出：

1. 市场概况：电价水平、RE消纳、碳排放是否正常。
2. 各参与方表现：发电侧、储能侧、负荷侧的收益与效率差异。
3. 风险与瓶颈：弃电、阻塞、电价异常发生的节点或时段。
4. 策略建议：按参与方类型分别给出参数调整方向。

=== 参考基准 ===
节点电价正常范围: 200-600 ¥/MWh，低于0为异常，高于800为紧张。
RE消纳率: >90%良好，<80%需关注。
碳强度: <0.4 tCO2/MWh较清洁，>0.6需优化。
储能利用率: SOC在20-80%之间波动为正常，长期不动或频繁触顶/触底为低效。
负荷满足率: >95%正常，<90%存在供电缺口。

=== 仿真数据 ===
场景: {summary.get('scenario', '未知')}
社会福利: {summary.get('welfare_da', '?')} ¥
RE消纳率: {summary.get('re_rate', '?')}%
负荷满足率: {summary.get('load_sat', '?')}%
节点电价（¥/MWh）: 平均{summary.get('avg_lmp', '?')}，最低{summary.get('min_lmp', '?')}，最高{summary.get('max_lmp', '?')}
碳排放: 总量{summary.get('carbon_emissions', '?')} tCO2，碳强度{summary.get('carbon_intensity', '?')} tCO2/MWh
弃电量: {summary.get('curtailment', '?')} MWh
储能SOC（%）: 平均{summary.get('avg_soc', '?')}，最低{summary.get('min_soc', '?')}，最高{summary.get('max_soc', '?')}
储能总充电: {summary.get('total_ch', '?')} MWh，总放电: {summary.get('total_dis', '?')} MWh
总购电量: {summary.get('total_buy', '?')} MWh，总售电量: {summary.get('total_sell', '?')} MWh
储能动作: {summary.get('storage_active', '?')}

按四段格式输出分析，总字数控制在500字以内，不要使用markdown，不要说明你的身份："""

        response = self._call_ollama(prompt, max_tokens=1024)
        if response:
            return response.strip()
        return self._rule_insight(summary)

    def _rule_insight(self, summary: dict) -> str:
        try:
            re_rate = float(summary.get('re_rate', 0))
            avg_lmp = float(summary.get('avg_lmp', 0))
            active = summary.get('storage_active', '否')
            load_sat = float(summary.get('load_sat', 100))
        except Exception:
            return "系统运行平稳，可微调报价参数以提升社会福利。"

        parts = []

        if load_sat < 90:
            parts.append("负荷满足率偏低，线路可能严重阻塞或发电能力不足，建议加强网架或增加分布式电源。")
        if re_rate > 95:
            parts.append("可再生能源消纳率高，午间光伏过剩可能导致电价下跌，建议增加储能容量或引入灵活负荷消纳更多绿电。")
        elif re_rate < 80:
            parts.append("可再生能源消纳率不足，弃风弃光严重，考虑降低可再生出力或提高储能充电功率以吸收过剩。")
        if avg_lmp < 0:
            parts.append("节点电价整体为负，供远大于求，建议降低可再生出力或提高储能充电功率以吸收过剩。")
        if active == '否':
            parts.append("储能未动作，电价波动可能较小或套利空间不足，可扩大电价峰谷差或调整储能报价策略。")

        if not parts:
            parts.append("系统运行平稳，各曲线符合预期，可进一步优化报价参数以提升社会福利。")

        return "。\n".join(parts) + "。"
