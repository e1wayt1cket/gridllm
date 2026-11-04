# Full default test suite

The default suite (`pytest.ini`: `-m "not slow"`) was run in two parts because
one pre-existing test exceeds a 60-second per-test timeout on this Windows
machine while completing normally:

| command | result |
|---|---|
| `python -m pytest tests/ -q --ignore=tests/test_eval_cli_api_consistency.py --ignore=tests/test_ui_pages.py` | **1361 passed**, 36 deselected, 0 failed (769 s) |
| `python -m pytest tests/test_eval_cli_api_consistency.py -q` | **4 passed** (80 s) |

`tests/test_ui_pages.py` carries `pytestmark = pytest.mark.slow` and is
deselected by the default configuration; it is not part of the default run.

Combined default-suite result: **1365 passed, 0 failed, 36 slow tests
deselected**. The earlier `--timeout=60` run interrupted
`test_eval_cli_api_consistency` during its 80-second evaluation; that was a
timeout-limit artifact, not a test failure. Logs:
`full_test_suite_core.log`, `test_eval_cli_api_consistency.log`.
