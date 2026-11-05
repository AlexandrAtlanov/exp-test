"""Подключает submission/, tools/ и training/ к sys.path для импортов в тестах.

Импортируется тестами отдельным оператором import: побочный эффект —
добавление путей (koz3.*, make_gt и другие инструменты, settings и
annotations). Отдельный модуль нужен, чтобы в тестах не было кода между
import-блоками (E402 в части версий ruff включён, в части нет; import-модуль
проходит оба дефолта).
"""

import sys
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parents[2]
for _sub in ("training", "tools", "submission"):
    _path = str(_REPO_DIR / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)
