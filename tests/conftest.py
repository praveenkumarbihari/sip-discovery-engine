import pytest

from sipscout.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture()
def conn():
    from sipscout import db
    c = db.connect(":memory:")
    yield c
    c.close()
