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
        "latest_completed_model_schema": {
            "kind": "InterpretedSchema",
            "columns": [
                {"name": "Factor 1", "guess": "real", "notices": []},
                {"name": "Factor 2", "guess": "real", "notices": []},
                {"name": "Factor 3", "guess": "real", "notices": []},
                {"name": "Category 0", "guess": "categorical", "notices": []},
                {"name": "Target 1", "guess": "real", "notices": []},
                {"name": "Target 2", "guess": "integer", "notices": []},
            ],
            "version": 1,
            "rowCount": 20,
            "globalErrors": [],
            "globalWarnings": [],
            "primaryKeyColumns": [],
        },
        "display_options": None,
        "ac_name": "grain_analysis",
    }


@pytest.fixture
def revision_data():
    return {
        "configured_blueprint": {
            "target_labels": ["s3_kpi"],
            "is_whitelist": False,
            "exclude_labels": ["dt"],
            "rolling_window_time": 1,
            "include_exclude_labels": ["dt"],
            "include_exclude_patterns": [],
            "force_include_labels": [],
            "force_include_patterns": [],
            "factor_labels": [
                "s3_factor2",
                "s1_factor1",
                "s1_factor2",
                "s2_factor1",
                "s3_factor1",
            ],
            "sensor_labels": [],
            "factor_grouping": "all factors",
            "factor_selection": "automatic",
        },
        "version": 0,
        "description": "",
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
def datasource_data():
    return {
        "uuid": "9f79206e-94fc-4834-8f52-84008b12df86",
        "name": "three_residences new",
        "description": "",
        "status": "R",
        "schema": {
            "columns": [
                {"guess": "integer", "name": "s1_factor1"},
                {"guess": "integer", "name": "s1_factor2"},
                {"guess": "integer", "name": "s2_factor1"},
                {"guess": "real", "name": "s3_factor1"},
                {"guess": "integer", "name": "s3_factor2"},
                {"guess": "real", "name": "s3_kpi"},
                {"name": "dt", "guess": "datetime"},
            ],
            "version": 2,
            "kind": "ParsedSchema",
        },
        "created": "2021-07-21T15:17:23.839599Z",
        "modified": "2021-07-21T15:17:24.540926Z",
        "error_notices": {"errors": []},
        "primary_key_column": None,
        "primary_datetime_column": "dt",
        "overwrites": {
            "columns": [],
            "version": 1,
            "kind": "DataSourceColumnOverwrites",
        },
        "download_url": None,
        "username": "jaz",
        "ac_name": "jazaccess",
        "transformed_source": False,
        "progress": 92,
        "live_source": False,
    }


@pytest.fixture
def process_data():
    return {
        "api_id": "617da764-9c41-4135-87fe-463b2f01b42b",
        "name": "Fixture Data",
        "created": "2022-01-31T19:54:32.541903Z",
        "modified": "2022-01-31T19:54:41.618842Z",
        "latest_revision_version": 1,
        "username": "fero",
        "process_type": "A",
        "product_type": None,
        "kind": "process",
        "latest_ready_snapshot": None,
        "data_config": {
            "config": {
                "kind": "AdvancedDataConfig",
                "liveData": [],
                "keyJoins": [],
                "initialFeed": {
                    "name": "quality-example-data.csv",
                    "datasource": "d8c5e93e-b39d-4b50-96d9-b394a0b99ad6",
                    "kind": "InitialFeedDescription",
                },
            },
            "version": 1,
            "kind": "ProcessDataConfiguration",
        },
        "shutdown_configuration": None,
        "primary_datetime_column": None,
    }


@pytest.fixture
def process_data_continuous(process_data):
    process_data["process_type"] = "C"
    return process_data


@pytest.fixture
def patched_fero_client():
    with mock.patch.object(Fero, "post"):
        with mock.patch.object(Fero, "get"):
            yield Fero(fero_token="fakeToken")
