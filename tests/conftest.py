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
def patched_fero_client():
    with mock.patch.object(Fero, "post"):
        with mock.patch.object(Fero, "get"):
            yield Fero(fero_token="fakeToken")
