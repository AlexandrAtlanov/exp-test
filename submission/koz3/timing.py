"""Бюджет времени прогона: единые монотонные часы для планирования."""

import time


class Budget:
    """Монотонные часы бюджета прогона

    Создаётся в solution.py один раз на прогон; детекторы спрашивают
    ``can_afford`` перед обработкой очередного файла.
    """

    def __init__(self, total_s: float) -> None:
        """Бюджет на ``total_s`` секунд; отсчёт с момента создания."""
        self._total_s = float(total_s)
        self._start = time.monotonic()

    def elapsed(self) -> float:
        """Прошло секунд с создания бюджета."""
        return time.monotonic() - self._start

    def remaining(self) -> float:
        """Осталось секунд бюджета; результат неотрицателен."""
        return max(0.0, self._total_s - self.elapsed())

    def can_afford(self, seconds: float) -> bool:
        """Хватает ли остатка бюджета на ``seconds`` секунд."""
        return self.remaining() >= seconds
