from fero import FeroError
import pytest
from unittest import mock
from fero.analysis import Analysis
import pandas as pd
from pandas.testing import assert_frame_equal


@pytest.fixture
def test_analysis(analysis_data, patched_fero_client):
    return Analysis(patched_fero_client, analysis_data)


@pytest.fixture
def prediction_data():
    return [
        {"value1": 1.0, "value2": 2, "value3": 3.3},
        {"value1": 4.0, "value2": 5, "value3": 6.3},
    ]


@pytest.fixture
def prediction_dataframe(prediction_data):
    return pd.DataFrame(prediction_data)


@pytest.fixture
def prediction_result_data():
    return {
        "status": "SUCCESS",
        "data": {
            "target1": {"value": {"high": [5.0], "low": [1.0], "mid": [3.0]}},
            "target2": {"value": {"high": [1.0], "low": [1.0], "mid": [1.0]}},
        },
    }


def test_creates_analysis_correctly(analysis_data, patched_fero_client):
    """Test that a valid analysis is created from valid data and a valid client"""

    analysis = Analysis(patched_fero_client, analysis_data)

    assert isinstance(analysis, Analysis)
    assert str(analysis) == "<Analysis name=quality-example-data>"


def test_has_trained_model_true(analysis_data, patched_fero_client):
    """Test that has_trained_model is true if there is a latest revision model"""

    analysis = Analysis(patched_fero_client, analysis_data)
    assert analysis.has_trained_model() is True


def test_has_trained_model_false(analysis_data, patched_fero_client):
    """Test that has_trained_model is true if there is no revision model"""
    analysis_data["latest_completed_model"] = None
    analysis = Analysis(patched_fero_client, analysis_data)
    assert analysis.has_trained_model() is False


def test_make_prediction_dictionaries(
    analysis_data, patched_fero_client, prediction_data, prediction_result_data
):
    """Test that make prediction returns expected response with a dictionary of predictions"""
    patched_fero_client.post.return_value = prediction_result_data
    analysis = Analysis(patched_fero_client, analysis_data)
    results = analysis.make_prediction(prediction_data)

    patched_fero_client.post.assert_has_calls(
        [
            mock.call(
                f"/api/revision_models/{str(analysis_data['latest_completed_model'])}/predict/",
                {"values": prediction_data[0]},
            ),
            mock.call(
                f"/api/revision_models/{str(analysis_data['latest_completed_model'])}/predict/",
                {"values": prediction_data[1]},
            ),
        ]
    )
    assert isinstance(results, list)
    assert results == [
        {
            "value1": 1.0,
            "value2": 2,
            "value3": 3.3,
            "target1_high": 5.0,
            "target1_low": 1.0,
            "target1_mid": 3.0,
            "target2_high": 1.0,
            "target2_low": 1.0,
            "target2_mid": 1.0,
        },
        {
            "value1": 4.0,
            "value2": 5,
            "value3": 6.3,
            "target1_high": 5.0,
            "target1_low": 1.0,
            "target1_mid": 3.0,
            "target2_high": 1.0,
            "target2_low": 1.0,
            "target2_mid": 1.0,
        },
    ]


def test_make_prediction_dataframe(
    analysis_data,
    patched_fero_client,
    prediction_dataframe,
    prediction_result_data,
    prediction_data,
):
    """Test that make prediction returns expected response with a dataframe"""
    patched_fero_client.post.return_value = prediction_result_data
    analysis = Analysis(patched_fero_client, analysis_data)
    results = analysis.make_prediction(prediction_dataframe)
    patched_fero_client.post.assert_has_calls(
        [
            mock.call(
                f"/api/revision_models/{str(analysis_data['latest_completed_model'])}/predict/",
                {"values": prediction_data[0]},
            ),
            mock.call(
                f"/api/revision_models/{str(analysis_data['latest_completed_model'])}/predict/",
                {"values": prediction_data[1]},
            ),
        ]
    )

    assert isinstance(results, pd.DataFrame)
    expected = pd.DataFrame(
        [
            {
                "value1": 1.0,
                "value2": 2.0,
                "value3": 3.3,
                "target1_high": 5.0,
                "target1_low": 1.0,
                "target1_mid": 3.0,
                "target2_high": 1.0,
                "target2_low": 1.0,
                "target2_mid": 1.0,
            },
            {
                "value1": 4.0,
                "value2": 5.0,
                "value3": 6.3,
                "target1_high": 5.0,
                "target1_low": 1.0,
                "target1_mid": 3.0,
                "target2_high": 1.0,
                "target2_low": 1.0,
                "target2_mid": 1.0,
            },
        ]
    )
    assert_frame_equal(
        results,
        expected,
        check_like=True,
    )


def test_make_prediction_dataframe_duplicate_cols(
    analysis_data, patched_fero_client, prediction_result_data
):
    """Test that make prediction returns managled columns if the column names would overlap"""

    dupe_data = [
        {"value1": 1.0, "value2": 2, "value3": 3.3, "target1_high": 5.5},
        {"value1": 4.0, "value2": 5, "value3": 6.3, "target1_high": 5.5},
    ]

    patched_fero_client.post.return_value = prediction_result_data
    analysis = Analysis(patched_fero_client, analysis_data)
    results = analysis.make_prediction(dupe_data)

    assert isinstance(results, list)
    assert results == [
        {
            "value1": 1.0,
            "value2": 2,
            "value3": 3.3,
            "target1_high.0": 5.0,
            "target1_low": 1.0,
            "target1_mid": 3.0,
            "target2_high": 1.0,
            "target2_low": 1.0,
            "target2_mid": 1.0,
            "target1_high": 5.5,
        },
        {
            "value1": 4.0,
            "value2": 5,
            "value3": 6.3,
            "target1_high.0": 5.0,
            "target1_low": 1.0,
            "target1_mid": 3.0,
            "target2_high": 1.0,
            "target2_low": 1.0,
            "target2_mid": 1.0,
            "target1_high": 5.5,
        },
    ]


def test_make_prediction_prediction_failure(
    analysis_data, patched_fero_client, prediction_data
):
    """Test that  a FeroError is raised if a prediction fails"""

    with pytest.raises(FeroError):
        patched_fero_client.post.return_value = {
            "status": "FAILED",
            "message": "Something broke!",
        }
        analysis = Analysis(patched_fero_client, analysis_data)
        analysis.make_prediction(prediction_data)
