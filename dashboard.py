
import numpy as np
import dash
from dash import html, dcc
import plotly.graph_objs as go
import datetime


def generate_demo_data():
    steps = 96
    times = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 15, 30, 45]]
    rng = np.random.default_rng(42)
    # 场景/代理/指标维度
    price_1 = 400 + 30 * \
        np.sin(np.linspace(0, 4 * np.pi, steps)) + rng.normal(0, 12, steps)
    price_2 = 420 + 20 * \
        np.cos(np.linspace(0, 2 * np.pi, steps)) + rng.normal(0, 13, steps)
    vol_1 = 20 + 8 * \
        np.abs(np.sin(np.linspace(0, 4 * np.pi, steps))) + rng.normal(1, 2, steps)
    vol_2 = 22 + 6 * \
        np.abs(np.cos(np.linspace(0, 3 * np.pi, steps))) + rng.normal(1, 2, steps)
    stack_a = np.clip(
        8 +
        6 *
        np.sin(
            np.linspace(
                0,
                2 *
                np.pi,
                steps)),
        2,
        None)
    stack_b = np.clip(
        6 +
        6 *
        np.cos(
            np.linspace(
                0,
                2 *
                np.pi,
                steps)),
        2,
        None)
    stack_c = np.clip(
        4 +
        2 *
        np.sin(
            np.linspace(
                0,
                6 *
                np.pi,
                steps)),
        2,
        None)
    # 统计项
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    stats = {
        "最高价": int(np.max([price_1, price_2])),
        "最低价": int(np.min([price_1, price_2])),
        "均价": f"{np.mean([price_1, price_2]):.1f}",
        "总成交(MWh)": int(np.sum(vol_1) + np.sum(vol_2)),
        "最新时间": f"{today} {times[-1]}",
        "当前价": f"{(float(price_1[-1]) + float(price_2[-1])) / 2:.1f}",
    }
    return {
        "times": times,
        "price_1": price_1,
        "price_2": price_2,
        "vol_1": vol_1,
        "vol_2": vol_2,
        "stack_a": stack_a,
        "stack_b": stack_b,
        "stack_c": stack_c,
        "stats": stats,
    }


def serve_app():
    data = generate_demo_data()
    app = dash.Dash(__name__)
    app.title = '电力市场大屏演示'
    layout = html.Div([
        # 顶部大标题与LOGO/时间
        html.Div([
            html.Div([
                html.Span(
                    "中国电力交易·大屏演示界面",
                    style={
                        "fontSize": 38,
                        "color": "#13325e",
                        "fontWeight": "bold",
                        "paddingRight": "18px"}),
                html.Span(
                    "Electricity Market Demo Display",
                    style={
                        "fontSize": 23,
                        "color": "#497acf",
                        "fontWeight": "normal"})
            ], style={"display": "inline-block", "verticalAlign": "bottom"}),
            html.Div([
                html.Span(f'{data["stats"]["最新时间"]}', style={
                          "fontSize": 16, "color": "#234e8c", "paddingRight": "16px"}),
                html.Img(
                    src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/State_Grid_Corporation_of_China_logo.svg/120px-State_Grid_Corporation_of_China_logo.svg.png",
                    height="48px"),
            ], style={"float": "right", "display": "inline-block", "verticalAlign": "top", "paddingRight": "38px"})
        ], style={"paddingTop": "24px", "paddingBottom": "6px", "marginLeft": "38px", "marginRight": "18px", "borderBottom": "2px solid #edf1fc"}),
        html.Div([
            html.Label("选择场景: ", style={"fontSize": 18, "marginRight": 10}),
            dcc.Dropdown(id='scene-dropdown',
                         options=[
                             {'label': '场景1', 'value': '1'},
                             {'label': '场景2', 'value': '2'}
                         ],
                         value='1',
                         style={"width": "200px", "display": "inline-block"})
        ], style={"margin": "20px 0", "paddingLeft": "38px"}),
        dcc.Graph(id='line-graph'),
    ])

    @app.callback(dash.Output('line-graph', 'figure'),
                  [dash.Input('scene-dropdown', 'value')])
    def update_graph(scene):
        scene_key = f"price_{scene}"
        figure = go.Figure(data=[
            go.Scatter(x=data['times'], y=data[scene_key], mode='lines+markers', name=f'场景{scene}')
        ])
        return figure

    app.layout = layout
    app.run(debug=True, port=8050, host="127.0.0.1")


if __name__ == "__main__":
    serve_app()
