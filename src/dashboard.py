# dashboard.py
"""Entry point for the research dashboard (port 8056).

The pages and their callbacks live under src/ui/; this module only builds the
shell and serves it, so `python src/dashboard.py` keeps working as before.
"""

from ui.shell import create_app

app = create_app()
server = app.server

if __name__ == "__main__":
    app.run(debug=True, port=8056)
