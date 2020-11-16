import pytest
from unittest import mock
from fero.client import Fero


@pytest.fixture
def analysis_data():
    return {
        "uuid": "21621466-b198-45be-89f9-3a5eb2c7cf48",
        "predictions_url": "/api/analyses/21621466-b198-45be-89f9-3a5eb2c7cf48/predictions/",
        "revisions_url": "/api/analyses/21621466-b198-45be-89f9-3a5eb2c7cf48/revisions/",
        "url": "/api/analyses/21621466-b198-45be-89f9-3a5eb2c7cf48/",
        "created": "2020-09-22T17:25:23.890397Z",
        "modified": "2020-09-22T17:26:32.828524Z",
        "created_by": {"id": 2, "username": "fero"},
        "name": "quality-example-data",
        "blueprint_name": "quality",
        "data_source": "d85cd464-f4ef-4aff-8440-95980924b976",
        "data_source_name": "quality-example-data",
        "data_source_deleted": False,
        "schema_overrides": {},
        "latest_revision": 0,
        "latest_revision_model": "e751f70e-aeb4-46ee-b860-bfc37f3767a7",
        "latest_revision_model_state": "I",
        "latest_trained_revision": 0,
        "latest_trained_revision_model": "e751f70e-aeb4-46ee-b860-bfc37f3767a7",
        "latest_completed_model": "e751f70e-aeb4-46ee-b860-bfc37f3767a7",
        "latest_completed_model_score": 95,
        "latest_completed_model_score_qualifier": "good",
        "latest_completed_model_modified": "2020-09-22T17:26:31.747595Z",
        "display_options": None,
        "ac_name": "grain_analysis",
    }


@pytest.fixture
def asset_data():
    return {
        "uuid": "fd57ba36-3c5d-40f5-ae0c-d7b76ab39ee5",
        "url": "/api/assets/fd57ba36-3c5d-40f5-ae0c-d7b76ab39ee5/",
        "configurations_url": "/api/assets/fd57ba36-3c5d-40f5-ae0c-d7b76ab39ee5/configurations/",
        "created": "2020-11-12T16:05:12.806853Z",
        "modified": "2020-11-13T15:05:58.266904Z",
        "created_by": {"id": 2, "username": "test"},
        "name": "TEST ASSET",
        "data_source": "fa24635d-d00d-4dd3-a11e-8009738a532f",
        "data_source_name": "Asset_Example",
        "data_source_deleted": False,
        "current_bound": {"type": "boundary", "direction": "upper", "value": 0},
        "latest_configuration": 2,
        "latest_configuration_model": "3647f65e-724f-49ba-bef5-812ca01a228d",
        "latest_configuration_model_state": "I",
        "latest_trained_configuration": 2,
        "latest_trained_configuration_model": "3647f65e-724f-49ba-bef5-812ca01a228d",
        "latest_completed_model": "3647f65e-724f-49ba-bef5-812ca01a228d",
        "latest_completed_model_score": None,
        "latest_completed_model_score_qualifier": None,
        "latest_completed_model_modified": "2020-11-13T15:05:58.033220Z",
        "stability_warning": 40,
        "stability_warning_display": "Strong Warning",
        "time_to_threshold_lower": "00:00:00",
        "time_to_threshold_mean": "00:00:00",
        "time_to_threshold_upper": "00:00:00",
        "prediction_horizon": "05:00:00",
        "prediction_at_horizon_lower": 0.1468654105352288,
        "prediction_at_horizon_mean": 1.2809534054200706,
        "prediction_at_horizon_upper": 2.424137436651667,
        "ac_name": "test_factory",
    }


@pytest.fixture
def patched_fero_client():
    with mock.patch.object(Fero, "post"):
        with mock.patch.object(Fero, "get"):
            yield Fero(fero_token="fakeToken")
