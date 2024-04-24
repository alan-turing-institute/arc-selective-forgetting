from typing import Iterator
from unittest.mock import patch

import pytest


@pytest.fixture(scope="session", autouse=True)
def default_session_fixture() -> Iterator[None]:
    print("Patching core.feature.service")
    with patch("core.feature.service.Utility"):
        yield
    print("Patching complete. Unpatching")
