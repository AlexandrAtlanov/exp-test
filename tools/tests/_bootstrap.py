"""Подключает каталог submission/ к sys.path для импортов koz3.* в тестах.

Импортируется тестами отдельным оператором import: побочный эффект —
добавление пути к пакету. Отдельный модуль нужен, чтобы в тестах не было
кода между import-блоками (E402 в части версий ruff включён, в части нет;
import-модуль проходит оба дефолта).
"""

import sys
from pathlib import Path

_SUBMISSION_DIR = Path(__file__).resolve().parents[2] / "submission"
if str(_SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBMISSION_DIR))