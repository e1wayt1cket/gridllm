
"""
Linear programming/multi-objective problem scanner for agent_trading.py
"""
import ast
with open("agent_trading.py", "r") as f:
    tree = ast.parse(f.read())
objectives = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and hasattr(node.func, 'attr'):
        if node.func.attr == "solve":
            # 认为调用优化器（如cvxpy/scipy等）
            objectives.append(ast.dump(node))
if __name__ == "__main__":
    print("Objective functions solve calls in agent_trading.py:")
    for i, obj in enumerate(objectives):
        print(i, obj)
