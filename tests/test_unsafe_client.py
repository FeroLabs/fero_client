"""Test the `UnsafeFeroForScripting` class."""

import pytest
from requests import Response
from unittest import mock
from unittest.mock import call
from fero.client import UnsafeFeroForScripting
from fero.workspace import Workspace
from fero.datasource import DataSource
from fero.process import Process
from fero.analysis import Analysis


@pytest.fixture
def patched_unsafe_fero_client():
    """Get fero client object using mocked access."""
    with mock.patch.object(UnsafeFeroForScripting, "post"):
        with mock.patch.object(UnsafeFeroForScripting, "get"):
            yield UnsafeFeroForScripting(fero_token="fakeToken")


@pytest.fixture
def workspace_fixture(workspace_data, patched_fero_client):
    """Get a sample `Workspace` object."""
    return Workspace(patched_fero_client, workspace_data)


@pytest.fixture
def patch_unsafe_fero_get():
    """Create a mocked GET call from the unsafe fero client."""
    with mock.patch.object(UnsafeFeroForScripting, "get") as mock_fero_get:
        yield mock_fero_get


def test_get_workspace_success(patch_unsafe_fero_get, workspace_data):
    """Test that a workspace is returned by `get_workspace`."""
    patch_unsafe_fero_get.return_value = workspace_data
    client = UnsafeFeroForScripting(fero_token="fakeToken", hostname="http://test.com")
    ws = client.get_workspace("9f79206e-94fc-4834-8f52-84008b12df86")
    assert isinstance(ws, Workspace)
    assert ws.name == workspace_data["name"]
    assert len(ws.datasources) == 2
    assert len(ws.processes) == 1
    assert len(ws.analyses) == 1
    patch_unsafe_fero_get.assert_called_with(
        "/api/workspaces/9f79206e-94fc-4834-8f52-84008b12df86/"
    )


def test_create_datasource_from_file(
    me_response, datasource_data, datasource_csv, v1_interpreted_schema
):
    """Test that a datasource is created from a CSV file."""
    client = UnsafeFeroForScripting(fero_token="fakeToken", hostname="http://test.com")

    def _mck(verb):
        return mock.patch.object(UnsafeFeroForScripting, verb)

    with open(datasource_csv, "rb") as f, _mck("post") as post_function, _mck(
        "get"
    ) as get_function, _mck("upload_file") as upload_file_function:
        uf_uuid = "e7365564-6671-472a-9345-2a6f66204b24"
        file_name = datasource_csv.name
        uploaded_file_response = {
            "uuid": uf_uuid,
            "name": file_name,
            "status": "I",
            "uploaded_data_configuration": None,
            "parsed_schema": None,
            "schema_overwrites": None,
            "created": "2024-05-02T14:26:52.506682Z",
            "modified": "2024-05-02T14:26:52.506693Z",
            "error_notices": {"parsing_notices": [], "global_notices": []},
            "preview_file": None,
        }
        post_function.side_effect = [
            # /api/v2/uploaded_files/
            uploaded_file_response,
            # /api/v2/data_source/
            datasource_data,
        ]
        # For s3 upload response
        upload_file_function.return_value = Response()
        inbox_response = {
            "file_name": file_name,
            "url": "http://localhost:8080/media",
            "fields": {
                "key": f"uploaded_files/{uf_uuid}/{datasource_csv}",
                "AWSAccessKeyId": "EXAMPLE",
                "policy": "example",
                "signature": "signature",
            },
            "upload_type": "s3",
        }
        get_function.side_effect = [
            inbox_response,
            # /api/me
            me_response,
        ]
        client.create_datasource_from_file(
            "my-new-datasource", file_name, v1_interpreted_schema, f
        )
        post_function.assert_has_calls(
            [
                call(
                    "/api/v2/uploaded_files/",
                    {
                        "name": file_name,
                        "uploaded_data_configuration": {
                            "upload_format_configuration": {
                                "format_type": "tabular",
                                "file_options": {"kind": "CsvFileOptions"},
                            }
                        },
                        "parsed_schema": v1_interpreted_schema,
                    },
                ),
                call(
                    "/api/v2/data_source/",
                    {
                        "name": "my-new-datasource",
                        "access_control": me_response["profile"]["default_upload_ac"][
                            "name"
                        ],
                        "uploaded_files_uuid": uploaded_file_response["uuid"],
                    },
                ),
            ]
        )
        get_function.assert_has_calls(
            [
                call(
                    f"/api/v2/uploaded_files/{uf_uuid}/inbox_url/?file_name={file_name}"
                ),
                call("/api/me/"),
            ]
        )
        upload_file_function.assert_called_once_with(
            inbox_response,
            file_name,
            f,
        )


def test_add_objects_to_workspace(
    workspace_fixture,
    workspace_data,
    patched_unsafe_fero_client,
    datasource_data,
    process_data,
    analysis_data,
):
    """Test that objects are added to a workspace."""

    def _call(uuid, object_name):
        return call(
            f"/api/workspaces/{workspace_fixture.uuid}/update_objects/",
            {
                "uuids": [str(uuid)],
                "object_name": object_name,
                "remove": False,
                "add_dependencies": False,
            },
        )

    patched_unsafe_fero_client.post.return_value = workspace_data

    datasource = DataSource(patched_unsafe_fero_client, datasource_data)
    patched_unsafe_fero_client.add_objects_to_workspace(workspace_fixture, [datasource])

    process = Process(patched_unsafe_fero_client, process_data)
    patched_unsafe_fero_client.add_objects_to_workspace(workspace_fixture, [process])

    analysis = Analysis(patched_unsafe_fero_client, analysis_data)
    patched_unsafe_fero_client.add_objects_to_workspace(workspace_fixture, [analysis])

    patched_unsafe_fero_client.post.assert_has_calls(
        [
            _call(datasource.uuid, "DataSourceV2"),
            _call(process.api_id, "Process"),
            _call(analysis.uuid, "Analysis"),
        ]
    )
