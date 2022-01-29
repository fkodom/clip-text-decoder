import pytest


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow to run")


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--slow")
    skip_slow = pytest.mark.skip(reason="need --slow or --very_slow option to run")

    for item in items:
        if ("slow" in item.keywords) and (not run_slow):
            item.add_marker(skip_slow)
