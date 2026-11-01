"""Тестовая настройка: доступ к пакету koz3 из каталога submission/."""

import sys
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parents[2] / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))