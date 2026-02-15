from pathlib import Path

from dotenv import load_dotenv

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


def pytest_configure(config):
    load_dotenv()
    ARTIFACTS_DIR.mkdir(exist_ok=True)
