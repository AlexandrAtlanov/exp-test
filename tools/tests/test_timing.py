"""Тесты koz3.timing.Budget."""

import time
import unittest

import _bootstrap  # noqa: F401  # добавляет submission/ в sys.path

from koz3.timing import Budget


class BudgetTest(unittest.TestCase):
    """Проверки бюджета времени."""

    def test_cannot_afford_over_budget(self) -> None:
        budget = Budget(10)
        self.assertFalse(budget.can_afford(20))

    def test_can_afford_within_budget(self) -> None:
        budget = Budget(10)
        self.assertTrue(budget.can_afford(5))

    def test_elapsed_and_remaining_at_start(self) -> None:
        budget = Budget(10)
        self.assertGreaterEqual(budget.elapsed(), 0.0)
        self.assertLessEqual(budget.remaining(), 10.0)
        self.assertGreaterEqual(budget.remaining(), 9.9)

    def test_elapsed_is_monotonic(self) -> None:
        budget = Budget(10)
        first = budget.elapsed()
        time.sleep(0.01)
        self.assertGreaterEqual(budget.elapsed(), first)


if __name__ == "__main__":
    unittest.main()
