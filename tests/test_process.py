import pytest
from unittest import mock
from fero.process import Process, Stage, Tag


@pytest.fixture
def process_fixture(process_data, patched_fero_client):
    return Process(patched_fero_client, process_data)


@pytest.fixture
def stage_fixture(process_fixture, process_stages):
    return [
        Stage(process_fixture, process_fixture._client, sdata)
        for sdata in process_stages["stages"]
    ]


@pytest.fixture
def tags_fixture(process_fixture, process_tags):
    return [
        Tag(process_fixture, process_fixture._client, tdata)
        for tdata in process_tags["tags"]
    ]


@pytest.fixture
def process_with_loaded_data(process_fixture, stage_fixture, tags_fixture):

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
    """Test that a process can get stages"""
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
    """Test that a process can get tags"""
    patched_fero_client.get.return_value = process_tags
    assert process_fixture.tags == tags_fixture
    # test caching
    assert process_fixture.tags == tags_fixture

    patched_fero_client.get.assert_called_once_with(
        f"/api/processes/{process_fixture.api_id}/tags/"
    )


def test_get_tag_stage_with_tag(process_with_loaded_data):
    """Test getting a stage by providing a tag"""
    (process_with_loaded_data.stages[0].tag_names)
    assert process_with_loaded_data.get_tag_stage("s2_factor1").id == 100
