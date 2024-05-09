"""A Module to hold pytest fixtures for tests."""

import numpy as np
import pandas as pd
import pytest
import datetime
from unittest import mock
from fero.client import Fero


@pytest.fixture
def workspace_data():
    """Get sample data matching the Workspace schema."""
    return {
        "uuid": "03d76555-1b7b-4e8e-8425-0779d5710345",
        "name": "Grain Analysis",
        "description": None,
        "modified": "2022-02-18T15:01:02.900782Z",
        "created_by": {"id": 1, "username": "admin"},
        "processes": [
            {
                "api_id": "04d9b4ff-0133-4842-9936-d99202a9f089",
                "name": "LIMS LONG",
                "modified": "2024-04-29T17:46:30.255605Z",
                "process_type": "B",
            }
        ],
        "analyses": [
            {
                "uuid": "75cd1f7f-92ae-4680-b902-d377d311d4fe",
                "name": "LIMS LONG",
                "modified": "2024-04-17T18:15:01.398146Z",
                "target_labels": [
                    "super_long_prefix_thats_so_long_it_should_not_be_allowedtarget_1"
                ],
            }
        ],
        "datasources": [
            {
                "uuid": "12ef3001-3cd8-4e54-abcc-707aab57ede6",
                "name": "batch_lims.csv",
                "modified": "2024-04-16T14:58:41.992953Z",
                "live_source": False,
            },
            {
                "uuid": "44288636-652e-44d4-a63d-d8ad6f713140",
                "name": "LIMS LONG",
                "modified": "2024-04-15T19:40:14.096580Z",
                "live_source": False,
            },
        ],
    }


@pytest.fixture
def analysis_data():
    """Get sample data matching the Analysis schema."""
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
def v1_interpreted_schema():
    """Get sample analysis schema."""
    return {
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
    }


@pytest.fixture
def revision_data():
    """Get sample data matching the Revision schema."""
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
    """Get sample data matching the Asset schema."""
    return {
        "uuid": "fdf0918b-98f3-4e25-978d-83caba7aebf6",
        "url": "/api/assets/fdf0918b-98f3-4e25-978d-83caba7aebf6/",
        "configurations_url": "/api/assets/fdf0918b-98f3-4e25-978d-83caba7aebf6/configurations/",
        "created": "2022-02-18T15:01:02.900782Z",
        "modified": "2022-02-18T15:58:01.826762Z",
        "created_by": {"id": 1, "username": "admin"},
        "name": "TEST ASSET",
        "data_source": None,
        "process": "8c4be5cc-cb92-4b46-b8bd-b9f29fb2a415",
        "process_name": "asset example continuous",
        "process_deleted": False,
        "current_bound": {"type": "boundary", "direction": "lower", "value": 4},
        "latest_configuration": 12,
        "latest_configuration_model": "ab86526f-60af-4626-9459-8b276b2c3ebd",
        "latest_configuration_model_state": "I",
        "latest_trained_configuration": 12,
        "latest_trained_configuration_model": "ab86526f-60af-4626-9459-8b276b2c3ebd",
        "latest_completed_model": "ab86526f-60af-4626-9459-8b276b2c3ebd",
        "latest_completed_model_score": None,
        "latest_completed_model_score_qualifier": None,
        "latest_completed_model_modified": "2022-02-18T15:58:01.697792Z",
        "stability_warning": 40,
        "stability_warning_display": "Strong Warning",
        "time_to_threshold_lower": "00:00:00",
        "time_to_threshold_mean": "00:00:00",
        "time_to_threshold_upper": "00:00:00",
        "prediction_horizon": "05:00:00",
        "prediction_at_horizon_lower": -4.826867616656165,
        "prediction_at_horizon_mean": -3.3494622633010933,
        "prediction_at_horizon_upper": -1.7714730482893926,
        "ac_name": "grain_analysis",
        "username": "admin",
    }


@pytest.fixture
def datasource_data():
    """Get sample data matching the Datasource schema."""
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
def datasource_csv(tmp_path):
    """Get sample CSV matching the Datasource schema."""
    rows = 30000
    start = datetime.datetime(2022, 1, 1)
    end = start + datetime.timedelta(seconds=rows - 1)
    date_rng = pd.date_range(start=start, end=end, freq="S")

    df = pd.DataFrame(
        {
            "s1_factor1": np.random.rand(rows),
            "s1_factor2": np.random.rand(rows),
            "s2_factor1": np.random.rand(rows),
            "s3_factor1": np.random.rand(rows),
            "s3_factor2": np.random.rand(rows),
            "s3_kpi": np.random.rand(rows),
            "dt": date_rng,
        }
    )
    temp_csv = tmp_path / "sample.csv"
    df.to_csv(temp_csv, index=False)
    return temp_csv


@pytest.fixture
def process_data():
    """Get sample data matching the Process schema."""
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
def process_stages():
    """Get sample data including a list of objects matching the Stage schema."""
    return {
        "stages": [
            {
                "name": "Default Name Stage 1",
                "configuration": {"toolkit": {"kind": "Advanced"}},
                "tags": [
                    {
                        "name": "s1_factor1",
                        "description": "",
                        "display_format": None,
                        "cost": None,
                        "unit": "",
                        "alias": "",
                        "minimum": None,
                        "maximum": None,
                        "lower_limit": None,
                        "upper_limit": None,
                        "step_size": None,
                        "proxy": 184,
                        "disabled": False,
                    },
                    {
                        "name": "s1_factor2",
                        "description": "",
                        "display_format": None,
                        "cost": None,
                        "unit": "",
                        "alias": "",
                        "minimum": None,
                        "maximum": None,
                        "lower_limit": None,
                        "upper_limit": None,
                        "step_size": None,
                        "proxy": 185,
                        "disabled": False,
                    },
                ],
                "order": 0,
                "proxy": 83,
                "id": 98,
            },
            {
                "name": "Default Name Stage 2",
                "configuration": {"toolkit": {"kind": "Advanced"}},
                "tags": [
                    {
                        "name": "s2_factor1",
                        "description": "",
                        "display_format": None,
                        "cost": None,
                        "unit": "",
                        "alias": "",
                        "minimum": None,
                        "maximum": None,
                        "lower_limit": None,
                        "upper_limit": None,
                        "step_size": None,
                        "proxy": 186,
                        "disabled": False,
                    }
                ],
                "order": 1,
                "proxy": 85,
                "id": 100,
            },
            {
                "name": "Default Name Stage 3",
                "configuration": {"toolkit": {"kind": "Advanced"}},
                "tags": [
                    {
                        "name": "s3_factor1",
                        "description": "",
                        "display_format": None,
                        "cost": None,
                        "unit": "",
                        "alias": "",
                        "minimum": None,
                        "maximum": None,
                        "lower_limit": None,
                        "upper_limit": None,
                        "step_size": None,
                        "proxy": 187,
                        "disabled": False,
                    },
                    {
                        "name": "s3_factor2",
                        "description": "",
                        "display_format": None,
                        "cost": None,
                        "unit": "",
                        "alias": "",
                        "minimum": None,
                        "maximum": None,
                        "lower_limit": None,
                        "upper_limit": None,
                        "step_size": None,
                        "proxy": 188,
                        "disabled": False,
                    },
                    {
                        "name": "s3_kpi",
                        "description": "",
                        "display_format": None,
                        "cost": None,
                        "unit": "",
                        "alias": "",
                        "minimum": None,
                        "maximum": None,
                        "lower_limit": None,
                        "upper_limit": None,
                        "step_size": None,
                        "proxy": 189,
                        "disabled": False,
                    },
                ],
                "order": 2,
                "proxy": 87,
                "id": 102,
            },
        ],
        "revision_version": 5,
        "stage_version": 2,
    }


@pytest.fixture
def process_tags():
    """Get sample data including a list of objects matching the Tag schema."""
    return {
        "tags": [
            {
                "name": "s1_factor1",
                "description": "",
                "display_format": None,
                "cost": None,
                "unit": "",
                "alias": "",
                "minimum": None,
                "maximum": None,
                "lower_limit": None,
                "upper_limit": None,
                "step_size": None,
                "proxy": 184,
                "disabled": False,
            },
            {
                "name": "s1_factor2",
                "description": "",
                "display_format": None,
                "cost": None,
                "unit": "",
                "alias": "",
                "minimum": None,
                "maximum": None,
                "lower_limit": None,
                "upper_limit": None,
                "step_size": None,
                "proxy": 185,
                "disabled": False,
            },
            {
                "name": "s2_factor1",
                "description": "",
                "display_format": None,
                "cost": None,
                "unit": "",
                "alias": "",
                "minimum": None,
                "maximum": None,
                "lower_limit": None,
                "upper_limit": None,
                "step_size": None,
                "proxy": 186,
                "disabled": False,
            },
            {
                "name": "s3_factor1",
                "description": "",
                "display_format": None,
                "cost": None,
                "unit": "",
                "alias": "",
                "minimum": None,
                "maximum": None,
                "lower_limit": None,
                "upper_limit": None,
                "step_size": None,
                "proxy": 187,
                "disabled": False,
            },
            {
                "name": "s3_factor2",
                "description": "",
                "display_format": None,
                "cost": None,
                "unit": "",
                "alias": "",
                "minimum": None,
                "maximum": None,
                "lower_limit": None,
                "upper_limit": None,
                "step_size": None,
                "proxy": 188,
                "disabled": False,
            },
            {
                "name": "s3_kpi",
                "description": "",
                "display_format": None,
                "cost": None,
                "unit": "",
                "alias": "",
                "minimum": None,
                "maximum": None,
                "lower_limit": None,
                "upper_limit": None,
                "step_size": None,
                "proxy": 189,
                "disabled": False,
            },
        ],
        "revision_version": 5,
        "tag_version": 2,
    }


@pytest.fixture
def process_data_continuous(process_data):
    """Get sample data matching the Process schema with a continuous process."""
    process_data["process_type"] = "C"
    return process_data


@pytest.fixture
def patched_fero_client():
    """Get fero client object using mocked access."""
    with mock.patch.object(Fero, "post"):
        with mock.patch.object(Fero, "get"):
            yield Fero(fero_token="fakeToken")


@pytest.fixture
def me_response():
    return {
        "id": 2,
        "username": "fero",
        "profile": {
            "company": {
                "name": "Farro",
                "allow_replace_datasource": True,
                "display_timezone": None,
                "allow_new_datasource": True,
                "allow_new_module": True,
                "allow_new_asset": True,
                "allow_process_datasource": True,
                "allow_create_cost_module": True,
                "allow_create_quality_module": True,
                "allow_create_yield_module": True,
                "allow_create_softsensor_module": True,
                "allow_create_emission_module": True,
                "allow_create_fault_module": True,
                "sidebar_data": False,
                "sidebar_data_v2": True,
                "sidebar_modules": True,
                "sidebar_analyses": True,
                "sidebar_assets": True,
                "sidebar_processes": True,
                "allow_aux_processes": True,
                "allow_live_processes": True,
                "feature_flags": {"workspaces": True},
            },
            "can_provision_users": False,
            "site_acl": ["access_control"],
            "write_accesses": [
                {
                    "readable_name": "Grain Analysis",
                    "name": "grain_analysis",
                    "access_type": "w",
                }
            ],
            "default_upload_ac": {
                "name": "grain_analysis",
                "readable_name": "Grain Analysis",
                "uuid": "03d76555-1b7b-4e8e-8425-0779d5710345",
            },
        },
        "impersonator_id": 1,
        "is_superuser": False,
        "email": "fero@ferolabs.com",
        "last_name": "",
        "first_name": "",
    }
