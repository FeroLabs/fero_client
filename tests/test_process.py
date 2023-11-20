"""A module to test `Process` and related classes."""

import pytest
import requests
import pandas as pd
from pandas.testing import assert_frame_equal
from unittest import mock
from fero.process import Process, Stage, Tag, DataRequestError
from io import BytesIO


@pytest.fixture
def process_fixture(process_data, patched_fero_client):
    """Get a sample `Process` object."""
    return Process(patched_fero_client, process_data)


@pytest.fixture
def stage_fixture(process_fixture, process_stages):
    """Get a sample list of `Stage` objects."""
    return [
        Stage(process_fixture, process_fixture._client, sdata)
        for sdata in process_stages["stages"]
    ]


@pytest.fixture
def tags_fixture(process_fixture, process_tags):
    """Get a sample list of `Tag` objects."""
    return [
        Tag(process_fixture, process_fixture._client, tdata)
        for tdata in process_tags["tags"]
    ]


@pytest.fixture
def process_with_loaded_data(process_fixture, stage_fixture, tags_fixture):
    """Get a sample `Process` object with loaded stages and tags."""
    with mock.patch.object(
        Process, "stages", new_callable=mock.PropertyMock
    ) as mock_stages:
        with mock.patch.object(
            Process, "tags", new_callable=mock.PropertyMock
        ) as mock_tags:
            mock_stages.return_value = stage_fixture
            mock_tags.return_value = tags_fixture
            yield process_fixture


def test_process_get_stages(
    process_fixture, patched_fero_client, process_stages, stage_fixture
):
    """Test that a process can get stages."""
    patched_fero_client.get.return_value = process_stages
    assert process_fixture.stages == stage_fixture
    # test caching
    assert process_fixture.stages == stage_fixture

    patched_fero_client.get.assert_called_once_with(
        f"/api/processes/{process_fixture.api_id}/stages/"
    )


def test_process_get_tags(
    process_fixture, patched_fero_client, process_tags, tags_fixture
):
    """Test that a process can get tags."""
    patched_fero_client.get.return_value = process_tags
    assert process_fixture.tags == tags_fixture
    # test caching
    assert process_fixture.tags == tags_fixture

    patched_fero_client.get.assert_called_once_with(
        f"/api/processes/{process_fixture.api_id}/tags/"
    )


def test_get_tag_stage_with_tag(process_with_loaded_data):
    """Test getting a stage by providing a tag name."""
    assert process_with_loaded_data.get_tag_stage("s2_factor1").id == 100


def test_get_tag_stage_with_tag_use_tag_obj(process_with_loaded_data, tags_fixture):
    """Test getting a stage by providing a tag."""
    tag = next(tag for tag in tags_fixture if tag.name == "s2_factor1")
    assert process_with_loaded_data.get_tag_stage(tag).id == 100


def test_get_tag_stage_no_tag(process_with_loaded_data):
    """Test getting a stage that doesn't exist."""
    assert process_with_loaded_data.get_tag_stage("not_a_factor") is None


def test_stages_by_kpis_with_tag_name(process_with_loaded_data, stage_fixture):
    """Test getting stages by providing a kpi name."""
    assert (
        process_with_loaded_data.get_stages_by_kpis(["s2_factor1"]) == stage_fixture[:2]
    )


def test_stages_by_kpis_with_tag_multiple(process_with_loaded_data, stage_fixture):
    """Test getting stages by providing multiple kpi name."""
    assert (
        process_with_loaded_data.get_stages_by_kpis(["s2_factor1", "s1_factor1"])
        == stage_fixture[:2]
    )


def test_stages_by_kpis_with_tag_use_tag_obj(
    process_with_loaded_data, tags_fixture, stage_fixture
):
    """Test getting a stages by providing a tag kpi."""
    tag = next(tag for tag in tags_fixture if tag.name == "s2_factor1")
    assert process_with_loaded_data.get_stages_by_kpis([tag]) == stage_fixture[:2]


def test_stages_by_kpis_no_tag(process_with_loaded_data):
    """Test getting a stages if the kpi doesn't exist."""
    assert process_with_loaded_data.get_stages_by_kpis(["not_a_factor"]) == []


def test_get_data_works(process_with_loaded_data, patched_fero_client):
    """Test that get data makes the correct calls and loads a file."""
    patched_fero_client.post.return_value = {
        "request_id": "abcd-1234",
        "data": {"download_url": "test.com", "success": True, "message": "success"},
        "complete": True,
    }

    test_df = pd.DataFrame({"s1_factor1": [1, 2, 3], "s3_factor1": ["x", "y", "z"]})
    test_file = BytesIO()
    test_df.to_parquet(test_file)
    test_file.seek(0)
    mock_request = mock.MagicMock()
    mock_request.iter_content.return_value = [test_file.read()]
    with mock.patch("requests.get", return_value=mock_request):
        df = process_with_loaded_data.get_data(["s1_factor1", "s3_factor1"])

    test_file.close()
    assert_frame_equal(test_df, df)

    patched_fero_client.post.assert_has_calls(
        [
            mock.call(
                f"/api/processes/{process_with_loaded_data.api_id}/download_process_data/",
                {
                    "request_data": {
                        "tags": ["s1_factor1", "s3_factor1"],
                        "kpis": None,
                        "include_batch_id": False,
                    }
                },
            ),
            mock.call(
                f"/api/processes/{process_with_loaded_data.api_id}/download_process_data/",
                {"request_id": "abcd-1234"},
            ),
        ]
    )


def test_get_data_continuous_no_kpi(process_with_loaded_data):
    """Test that get data raises an error for a continuous process and with no kpis."""
    process_with_loaded_data._data["process_type"] = "C"

    with pytest.raises(DataRequestError) as exc:
        process_with_loaded_data.get_data(["s1_factor1", "s3_factor1"])
        assert (
            exc.message == "A kpi must be specified to download continuous process data"
        )


def test_get_data_unsuccesful(process_with_loaded_data, patched_fero_client):
    """Test that a fero error is raised if the server indicates an unsuccesful request."""
    patched_fero_client.post.return_value = {
        "request_id": "abcd-1234",
        "data": {
            "download_url": None,
            "success": False,
            "message": "something bad happened!",
        },
        "complete": True,
    }

    with pytest.raises(DataRequestError) as exc:
        process_with_loaded_data.get_data(["s1_factor1", "s3_factor1"])
        assert exc.message == "something bad happened!"

    patched_fero_client.post.assert_has_calls(
        [
            mock.call(
                f"/api/processes/{process_with_loaded_data.api_id}/download_process_data/",
                {
                    "request_data": {
                        "tags": ["s1_factor1", "s3_factor1"],
                        "kpis": None,
                        "include_batch_id": False,
                    }
                },
            ),
            mock.call(
                f"/api/processes/{process_with_loaded_data.api_id}/download_process_data/",
                {"request_id": "abcd-1234"},
            ),
        ]
    )


def test_get_data_raises_bad_status(
    monkeypatch, process_with_loaded_data, patched_fero_client
):
    """Test that an exception is raised for non-2xx status codes."""

    class MockResponse:
        status_code = 404

        def raise_for_status(self):
            if 400 <= self.status_code < 600:
                raise requests.exceptions.HTTPError(response=self)

    def mock_get(*args, **kwargs):
        return MockResponse()

    def mock_raise_for_status(*args, **kwargs):
        mock_response.raise_for_status()

    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(requests.Response, "raise_for_status", mock_raise_for_status)
    mock_response = MockResponse()

    with pytest.raises(requests.exceptions.HTTPError) as exc:
        process_with_loaded_data.get_data(["s1_factor1", "s3_factor1"])
        assert exc.value.response.status_code in range(400, 500)
