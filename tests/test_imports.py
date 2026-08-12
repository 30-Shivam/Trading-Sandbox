"""Cheap smoke coverage: every top-level script imports cleanly. Catches
syntax/import regressions (the kind this session hit more than once)
within seconds, in CI, before a scheduled run ever has the chance to fail
on them in production.
"""
import importlib

import pytest

TOP_LEVEL_MODULES = [
    "dip_buy_analyzer",
    "ingest",
    "daily_run",
    "optimize",
    "benchmark_random_entry",
    "run_backtest",
    "settle_trades",
    "confirm_fill",
    "evaluate_config",
    "llm_agent",
    "ai_context",
    "market_data",
    "notifications",
    "config_loader",
]


@pytest.mark.parametrize("module_name", TOP_LEVEL_MODULES)
def test_module_imports_cleanly(module_name):
    importlib.import_module(module_name)
