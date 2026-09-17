import pytest

from app import app
from identity import AuthStore, UserRecord, require_admin, require_approved_user, require_user

_TEST_USER = UserRecord(
    id="test-user",
    email="tester@example.com",
    role="user",
    status="approved",
    created_at=0.0,
    updated_at=0.0,
)


@pytest.fixture(autouse=True)
def _default_approved_user(monkeypatch):
    """Most API tests exercise chat/history without standing up real sessions."""
    app.dependency_overrides[require_user] = lambda: _TEST_USER
    app.dependency_overrides[require_approved_user] = lambda: _TEST_USER

    # Fake auth user is not in sqlite; billing still runs after each chat turn.
    original_consume = AuthStore.consume_cloud_usage

    def consume(self, user_id, *, llm=0, embed=0, rerank=0):
        if user_id == _TEST_USER.id:
            return _TEST_USER
        return original_consume(self, user_id, llm=llm, embed=embed, rerank=rerank)

    monkeypatch.setattr(AuthStore, "consume_cloud_usage", consume)
    yield
    app.dependency_overrides.pop(require_user, None)
    app.dependency_overrides.pop(require_approved_user, None)
    app.dependency_overrides.pop(require_admin, None)
