# ui/theme.py
"""Shared visual constants and the app's injected CSS.

Colour names and the stylesheet are carried over from the original
single-page dashboard unchanged, so moving to a multi-page shell does not
restyle anything.
"""

from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

# -- 颜色面板 (保留原有设计) --
C_BG       = "#f0f2f5"
C_SURFACE  = "#ffffff"
C_PRIMARY  = "#4f46e5"
C_ACCENT   = "#7c3aed"
C_TEXT     = "#1e293b"
C_MUTED    = "#64748b"
C_BORDER   = "#e2e8f0"
C_SUCCESS  = "#10b981"
C_WARNING  = "#f59e0b"
C_DANGER   = "#ef4444"
C_INFO     = "#3b82f6"

# -- 占位辅助函数 --
def _placeholder_fig(text="等待仿真结果..."):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white", paper_bgcolor=C_SURFACE,
        plot_bgcolor='#f8fafc', height=420,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.add_annotation(text=text, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color=C_MUTED, size=15))
    return fig

def _placeholder_kpi():
    return dbc.Alert(
        "输入场景描述并点击「运行仿真」查看结果",
        color='light', className='text-center',
        style={'color': C_MUTED, 'borderRadius': '14px'}
    )

def _placeholder_text(text="暂无数据"):
    return html.Div(text, style={'color': C_MUTED, 'textAlign': 'center',
                                  'padding': '24px', 'fontSize': '14px'})


INDEX_STRING = '''
<!DOCTYPE html>
<html>
    <head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
        <style>
            *, *::before, *::after { box-sizing: border-box; }
            body {
                background-color: #f0f2f5; margin: 0;
                font-family: "Inter", "Segoe UI", "PingFang SC", Arial, sans-serif;
                -webkit-font-smoothing: antialiased;
            }

            .chart-card {
                transition: box-shadow 0.2s ease;
            }
            .chart-card:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04) !important;
            }

            .kpi-card {
                background: #ffffff;
                padding: 18px 20px;
                border-radius: 14px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.06);
                text-align: center;
                min-width: 140px;
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }
            .kpi-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(0,0,0,0.10);
            }

            #nl-input:focus, #nl-input:focus-within {
                border-color: #4f46e5 !important;
                box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
            }

            #parse-btn:hover {
                background-color: #4338ca !important;
                box-shadow: 0 4px 12px rgba(79,70,229,0.35);
            }
            #parse-btn:active {
                transform: scale(0.97);
            }

            ::-webkit-scrollbar { width: 6px; height: 6px; }
            ::-webkit-scrollbar-track { background: transparent; }
            ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
            ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

            .modebar { opacity: 0.3; transition: opacity 0.2s; }
            .chart-card:hover .modebar { opacity: 1; }
        </style>
    </head>
    <body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>
'''
