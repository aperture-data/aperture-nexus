"""Native error statuses must not be hidden by legacy message matching."""
import pytest

from aperture_nexus.memory import _check_response
from aperture_nexus.exceptions import NexusStorageError


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('status', [-1, 1, 3, 4])
def test_error_text_cannot_override_failure_status(wrapped, status):
    body = {'status': status, 'info': 'Index already exists with a different definition'}
    response = [{'CreateIndex': body}] if wrapped else body
    with pytest.raises(NexusStorageError):
        _check_response(response, 'CreateIndex')


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('status', [0, 2])
def test_explicit_idempotent_status_is_supported(wrapped, status):
    body = {'status': status, 'info': 'conditional object exists'}
    _check_response([{'AddEntity': body}] if wrapped else body, 'AddEntity')
