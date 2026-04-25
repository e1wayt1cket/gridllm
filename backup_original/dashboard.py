import numpy as np
import dash
from dash import html, dcc
import plotly.graph_objs as go
import datetime


def generate_demo_data():
    steps = 96
    times = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 15, 30, 45]]
    rng = np.random.default_rng(42)

    price_1 = 400 + 30 * \
        np.sin(np.linspace(0, 4 * np.pi, steps)) + rng.normal(0, 12, steps)
    price_2 = 420 + 20 * \
        np.cos(np.linspace(0, 2 * np.pi, steps)) + rng.normal(0, 13, steps)
    vol_1 = 20 + 8 * \
        np.abs(np.sin(np.linspace(0, 4 * np.pi, steps))) + rng.normal(1, 2, steps)
    vol_2 = 22 + 6 * \
        np.abs(np.cos(np.linspace(0, 3 * np.pi, steps))) + rng.normal(1, 2, steps)

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
        "stats": stats,
    }


def serve_app():
    data = generate_demo_data()
    app = dash.Dash(__name__)
    app.title = '电力市场大屏演示'

    layout = html.Div([
        html.Div([
            html.Div([
                html.H1("电力市场实时监控", style={"color": "#13325e"}),
                html.P(f"最后刷新时间: {data['stats']['最新时间']}", style={"fontSize": 16, "color": "#234e8c"}),
            ], style={"marginBottom": "20px"}),

            html.Div([
                html.Div([
                    html.H3("实时电价", style={"color": "#13325e"}),
                    html.P(f"{data['stats']['当前价']} 元/MWh", style={"fontSize": 20}),
                ], style={"display": "inline-block", "width": "48%"}),

                html.Div([
                    html.H3("总交易量", style={"color": "#13325e"}),
                    html.P(f"{data['stats']['总成交(MWh)']} MWh", style={"fontSize": 20}),
                ], style={"display": "inline-block", "width": "48%"}),
            ], style={"display": "flex", "justifyContent": "space-between"}),
        ], style={"padding": "20px", "backgroundColor": "#f3f6fc"}),

        dcc.Graph(id="price-volume-graph", figure={
            "data": [
                go.Scatter(x=data['times'], y=data['price_1'], mode='lines', name='电价1'),
                go.Scatter(x=data['times'], y=data['price_2'], mode='lines', name='电价2'),
                go.Bar(x=data['times'], y=data['vol_1'], name='交易量1', opacity=0.5),
                go.Bar(x=data['times'], y=data['vol_2'], name='交易量2', opacity=0.5),
            ],
            "layout": go.Layout(
                title="实时电价与交易量",
                xaxis={"title": "时间"},
                yaxis={"title": "价格 (元/MWh)"},
                barmode='overlay',
            )
        }),
    ])

    app.layout = layout
    app.run(debug=True, port=8060, host="127.0.0.1")


if __name__ == "__main__":
    serve_app()
