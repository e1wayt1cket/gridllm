
"""
Web API for serving agent trading status and real-time data
Uses FastAPI as backend. Interfaces with agent_trading.py
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import importlib.util
import sys
from pathlib import Path

# 动态加载 agent_trading
file_path = Path(__file__).parent / "agent_trading.py"
spec = importlib.util.spec_from_file_location("agent_trading", file_path)
agent_trading = importlib.util.module_from_spec(spec)
sys.modules["agent_trading"] = agent_trading
spec.loader.exec_module(agent_trading)

app = FastAPI()


@app.get("/status")
def get_status():
    # 仅供演示，实际需agent_trading暴露数据方法
    return {
        "msg": "OK",
        "type": getattr(
            agent_trading,
            "__doc__",
            "agent_trading")}


@app.get("/agents")
def get_agents():
    # 演示：返回所有 agent 状态
    if hasattr(agent_trading, "get_agent_states"):
        return agent_trading.get_agent_states()
    return {"msg": "No get_agent_states implemented"}


@app.get("/realtime")
def get_realtime():
    # 实时数据接口
    if hasattr(agent_trading, "get_realtime_data"):
        return agent_trading.get_realtime_data()
    return {"msg": "No get_realtime_data implemented"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8086)
