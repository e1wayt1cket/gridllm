"""
LLM Advisor for Multi-Agent Energy Trading Simulation
Integrates Large Language Model to provide strategic advice and analysis
"""
import json
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np

# 更新导入以使用当前项目结构
from models import MarketConfig
from market import clear_market, adaptive_bidding
from scenarios import get_scenario


@dataclass
class AdviceRequest:
    """
    Request structure for LLM advisor
    """
    scenario_name: str
    agents_data: Dict[str, Any]
    market_results: Dict[str, Any]
    query: str
    use_ac_opf: bool = False  # 使用DC OPF作为默认


class LLMAdvisor:
    """
    Interface to LLM for strategic advice in energy trading
    """
    def __init__(self, api_url: str = "http://localhost:11434/api/generate", model: str = "llama2"):
        self.api_url = api_url
        self.model = model
        self.default_config = MarketConfig(use_ac_opf=False)  # 使用DC OPF作为默认

    def get_strategic_advice(self, req: AdviceRequest) -> str:
        """
        Get strategic advice based on market simulation results
        """
        try:
            # Construct prompt
            prompt = self._construct_prompt(req)
            
            # Call LLM API
            response = self._call_llm_api(prompt)
            
            return response
        except Exception as e:
            print(f"LLM advisor error: {e}")
            return self._fallback_advice(req.query)

    def _construct_prompt(self, req: AdviceRequest) -> str:
        """
        Construct prompt for LLM
        """
        # Extract key metrics
        avg_price = req.market_results['price'].mean() if isinstance(req.market_results['price'], np.ndarray) else req.market_results['price']
        re_consumption_rate = req.market_results['re_consumption_rate']
        welfare = req.market_results['welfare']
        
        # Count storage agents
        storage_count = sum(1 for agent_name in req.market_results['schedules'] 
                           if req.agents_data[agent_name].get('has_storage', False))
        
        prompt = f"""
        You are an expert advisor for energy market simulations. 
        Analyze the following scenario: '{req.scenario_name}'
        
        Market Results:
        - Average Market Price: {avg_price:.2f} ¥/MWh
        - Renewable Consumption Rate: {re_consumption_rate:.2f}%
        - Social Welfare: {welfare:.2f}
        - Number of Storage Agents: {storage_count}
        - OPF Type Used: {"AC" if req.use_ac_opf else "DC"}
        
        User Query: {req.query}
        
        Provide strategic advice based on these results. Focus on:
        1. Economic implications of the results
        2. Efficiency of renewable integration
        3. Performance of storage assets
        4. Recommendations for improving market outcomes
        5. Specific suggestions for agent strategies
        """
        
        return prompt

    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "No response from LLM")
        except requests.exceptions.RequestException as e:
            print(f"Failed to connect to LLM API: {e}")
            return f"API connection failed: {e}"
        except Exception as e:
            print(f"Unexpected error when calling LLM API: {e}")
            return f"Unexpected error: {e}"

    def _fallback_advice(self, query: str) -> str:
        """
        Fallback advice if LLM is not available
        """
        return f"""
        [LLM Connection Failed]
        
        Based on your query: "{query}"
        
        General advice:
        1. Check if Ollama is running and the model is loaded
        2. Verify the API endpoint configuration
        3. Consider running the simulation with different parameters
        
        Current Configuration:
        - Using AC OPF: {self.default_config.use_ac_opf}
        """

    def parse_natural_language_to_config(self, user_input: str) -> Dict[str, Any]:
        """
        将自然语言指令转换为结构化配置。
        简化版：基于关键词规则映射，无需调用外部 LLM。
        """
        user_input_lower = user_input.lower()

        config = {
            "scenario_type": "baseline",
            "parameters": {
                "load_factor": 1.0,
                "re_factor": 1.0,
                "line_capacity_factor": 1.0,
                "T": 96,
                "strategy": "random",
                "use_ac_opf": False,
                "w_re_consume": 0.0,
            },
            "description": "自定义场景"
        }

        # 场景关键词映射
        if any(kw in user_input_lower for kw in ["低负荷", "春节", "节假日", "深夜", "low load"]):
            config["parameters"]["load_factor"] = 0.3
            config["scenario_type"] = "low_load_high_re"
        elif any(kw in user_input_lower for kw in ["高峰负荷", "极端天气", "peak load", "high load"]):
            config["parameters"]["load_factor"] = 1.8
            config["scenario_type"] = "peak_load"

        if any(kw in user_input_lower for kw in ["高光伏", "高风电", "高可再生", "high re", "rich"]):
            config["parameters"]["re_factor"] = 2.0
            if config["scenario_type"] == "low_load_high_re":
                config["parameters"]["re_factor"] = 3.0
        elif any(kw in user_input_lower for kw in ["低光伏", "无风", "low re", "匮乏"]):
            config["parameters"]["re_factor"] = 0.2
            config["scenario_type"] = "high_load_low_re"

        if any(kw in user_input_lower for kw in ["阻塞", "瓶颈", "congestion", "bottleneck"]):
            config["parameters"]["line_capacity_factor"] = 0.5
            if config["scenario_type"] == "peak_load":
                config["scenario_type"] = "peak_congestion"
            else:
                config["scenario_type"] = "congestion"

        if any(kw in user_input_lower for kw in ["骤降", "drop", "sudden drop"]):
            config["scenario_type"] = "re_ramp_drop"
        elif any(kw in user_input_lower for kw in ["骤升", "surge", "sudden surge"]):
            config["scenario_type"] = "re_ramp_surge"

        if any(kw in user_input_lower for kw in ["纳什", "nash", "博弈", "game"]):
            config["parameters"]["strategy"] = "nash"
        elif any(kw in user_input_lower for kw in ["lmp", "边际电价", "价格", "price"]):
            config["parameters"]["strategy"] = "lmp_based"
        elif any(kw in user_input_lower for kw in ["最佳响应", "best response", "响应"]):
            config["parameters"]["strategy"] = "best_response"
        elif any(kw in user_input_lower for kw in ["随机", "random"]):
            config["parameters"]["strategy"] = "random"

        if any(kw in user_input_lower for kw in ["ac", "交流"]):
            config["parameters"]["use_ac_opf"] = True

        if any(kw in user_input_lower for kw in ["24小时", "24h", "hour"]):
            config["parameters"]["T"] = 24
        elif any(kw in user_input_lower for kw in ["48小时", "48h"]):
            config["parameters"]["T"] = 48

        config["description"] = f"解析: {user_input[:30]}..."
        return config


def analyze_storage_performance(market_results: Dict, agents_data: Dict) -> str:
    """
    Analyze storage performance based on market results
    """
    storage_analysis = []
    
    for agent_name, sched in market_results['schedules'].items():
        # 跳过特殊键
        if agent_name == 'GRID':
            continue
            
        agent_info = agents_data.get(agent_name, {})
        
        if agent_info.get('has_storage', False):
            # Calculate storage metrics
            total_charge = np.sum(sched['p_ch'])
            total_discharge = np.sum(sched['p_dis'])
            avg_soc = np.mean(sched['soc'])
            
            # Count active periods
            charge_periods = np.sum(sched['p_ch'] > 0.001)
            discharge_periods = np.sum(sched['p_dis'] > 0.001)
            
            analysis = f"""
            Storage Agent: {agent_name}
            - Total Charge: {total_charge * 1000:.2f} kWh
            - Total Discharge: {total_discharge * 1000:.2f} kWh
            - Average SOC: {avg_soc * 100:.1f}%
            - Active Charge Periods: {charge_periods}
            - Active Discharge Periods: {discharge_periods}
            """
            
            storage_analysis.append(analysis)
    
    return "\n".join(storage_analysis) if storage_analysis else "No storage agents found in this scenario."


def run_strategic_analysis(scenario_name: str, strategy: str = "random", T: int = 96) -> Dict[str, Any]:
    """Run a complete strategic analysis"""
    agents, wholesale = get_scenario(scenario_name, T=T)
    config = MarketConfig(use_ac_opf=False)
    
    # 使用指定的策略
    actions = adaptive_bidding(agents, config, strategy=strategy)
    
    # 运行市场出清
    market_results = clear_market(agents, T, 'DA', actions, config)

    agents_data = {}
    for agent in agents:
        agents_data[agent.name] = {
            'has_storage': agent.storage is not None,
            'has_pv': agent.is_prosumer and np.sum(agent.pv_forecast) > 0,
            'has_wind': agent.has_wind,
            'load_type': getattr(agent, 'load_type', 'unknown'),
            'is_prosumer': agent.is_prosumer
        }

    advisor = LLMAdvisor()
    queries = [
        "How efficient is the renewable energy consumption in this scenario?",
        "What is the economic performance of storage assets?",
        "How could agents improve their market strategies?"
    ]

    analysis_results = {
        'market_results': market_results,
        'agents_data': agents_data,
        'advice': []
    }

    for query in queries:
        req = AdviceRequest(
            scenario_name=scenario_name,
            agents_data=agents_data,
            market_results=market_results,
            query=query,
            use_ac_opf=config.use_ac_opf
        )
        advice = advisor.get_strategic_advice(req)
        analysis_results['advice'].append({'query': query, 'advice': advice})

    storage_analysis = analyze_storage_performance(market_results, agents_data)
    analysis_results['storage_analysis'] = storage_analysis
    return analysis_results


def run_llm_analysis_for_dashboard(scenario_name: str, strategy: str = "random", T: int = 96) -> Dict[str, Any]:
    """
    专为dashboard设计的LLM分析函数
    """
    try:
        results = run_strategic_analysis(scenario_name, strategy, T)
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error running LLM analysis: {str(e)}"
        }


if __name__ == "__main__":
    # 示例用法
    print("LLM Advisor Module - Testing")
    advisor = LLMAdvisor()
    
    # 解析自然语言示例
    user_input = "我想看高可再生渗透率场景下的市场表现"
    parsed_config = advisor.parse_natural_language_to_config(user_input)
    print(f"Parsed Config: {parsed_config}")
    
    # 运行战略分析示例
    analysis = run_strategic_analysis("high_re", strategy="random", T=24)
    print(f"Analysis completed for {len(analysis['advice'])} queries")
    print(f"Storage analysis: {analysis['storage_analysis'][:100]}...")