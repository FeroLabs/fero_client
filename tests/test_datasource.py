"""A module to test `DataSource` and related classes."""

import pytest
from fero import FeroError
from fero.datasource import UploadedFileStatus


@pytest.fixture
def file_status_data():
    """Get sample data matching the `UploadedFileStatus` class structure."""
    return {
        "uuid": "5351ab61-a50b-428d-adbb-81c927a8f14c",
        "status": "D",
        "error_notices": {"global_notices": [], "parsing_notices": []},
    }


def test_get_upload_status_makes_correct_calls(patched_fero_client, file_status_data):
    """Test that `get_upload_status` makes the correct call to the server."""
    file_uuid = file_status_data["uuid"]
    file_status = UploadedFileStatus(patched_fero_client, file_uuid)
    patched_fero_client.get.return_value = file_status_data

    status = file_status.get_upload_status()
    status["uuid"] = str(status["uuid"])
    assert status == file_status_data
    patched_fero_client.get.assert_called_with(
        f"/api/v2/uploaded_files/{file_uuid}/", allow_404=True
    )


def test_wait_until_complete_returns_good(patched_fero_client, file_status_data):
    """Test that `wait_until_complete` returns the instance when complete."""
    file_uuid = file_status_data["uuid"]
    file_status = UploadedFileStatus(patched_fero_client, file_uuid)
    patched_fero_client.get.return_value = file_status_data

    assert isinstance(file_status.wait_until_complete(), UploadedFileStatus)


def test_wait_until_complete_raises_error(patched_fero_client, file_status_data):
    """Test that `wait_until_complete` raises the returned error."""
    file_uuid = file_status_data["uuid"]
    file_status = UploadedFileStatus(patched_fero_client, file_uuid)
    file_status_data["error_notices"]["parsing_notices"] = [{"kind": "test_error"}]
    file_status_data["status"] = "E"
    patched_fero_client.get.return_value = file_status_data
    with pytest.raises(FeroError) as e:
        file_status.wait_until_complete()

    assert (
        e.value.message
        == f'Unable to upload file.  The following error(s) occurred: "{str({"kind": "test_error"})}"'
    )
