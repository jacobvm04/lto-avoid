from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


def pytest_configure(config):
    ARTIFACTS_DIR.mkdir(exist_ok=True)
