import pytest

from app import app
from identity import UserRecord, require_admin, require_approved_user, require_user

_TEST_USER = UserRecord(
    id="test-user",
    email="tester@example.com",
    role="user",
    status="approved",
    created_at=0.0,
    updated_at=0.0,
)


@pytest.fixture(autouse=True)
def _default_approved_user():
    """Most API tests exercise chat/history without standing up real sessions."""
    app.dependency_overrides[require_user] = lambda: _TEST_USER
    app.dependency_overrides[require_approved_user] = lambda: _TEST_USER
    yield
    app.dependency_overrides.pop(require_user, None)
    app.dependency_overrides.pop(require_approved_user, None)
    app.dependency_overrides.pop(require_admin, None)
