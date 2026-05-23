"""Shared pytest fixtures and configuration."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --integration flag to run network-requiring tests."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests that download models from HuggingFace Hub",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration tests unless --integration is passed."""
    if config.getoption("--integration"):
        return
    skip_integration = pytest.mark.skip(reason="Pass --integration to run HF model scans")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
