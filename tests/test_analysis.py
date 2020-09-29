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
def test_analysis_with_data(test_analysis):

    test_analysis._presentation_data_cache = {
        "data": [
            {
                "id": "regression_simulator",
                "type": "regression_simulator",
                "title": "Prediction Simulator",
                "tab": "Prediction Simulator",
                "content": {
                    "factors": [
                        {
                            "factor": "Factor 1",
                            "min": 51.0569817398695,
                            "median": 148.02718539320762,
                            "max": 245.75370014648703,
                            "importances": [0.9184262466993418],
                            "step": 1.9469671840661755,
                            "default": 210.17636699824746,
                            "dtype": "float",
                        },
                        {
                            "factor": "Factor 2",
                            "min": 51.0569817398695,
                            "median": 148.02718539320762,
                            "max": 245.75370014648703,
                            "importances": [0.9184262466993418],
                            "step": 1.9469671840661755,
                            "default": 210.17636699824746,
                            "dtype": "float",
                        },
                        {
                            "factor": "Factor 3",
                            "min": 51.0569817398695,
                            "median": 148.02718539320762,
                            "max": 245.75370014648703,
                            "importances": [0.9184262466993418],
                            "step": 1.9469671840661755,
                            "default": 210.17636699824746,
                            "dtype": "float",
                        },
                    ],
                    "targets": [
                        {
                            "name": "Target 1",
                            "default_measured": 774.2382246688347,
                            "default": {
                                "low": [734.5873947705951],
                                "mid": [767.84871403268],
                                "high": [801.110033294765],
                            },
                            "min": 552.3692540713583,
                            "max": 819.9899001874504,
                        },
                        {
                            "name": "Target 2",
                            "default_measured": 774.2382246688347,
                            "default": {
                                "low": [734.5873947705951],
                                "mid": [767.84871403268],
                                "high": [801.110033294765],
                            },
                            "min": 552.3692540713583,
                            "max": 819.9899001874504,
                        },
                    ],
                },
            }
        ]
    }

    return test_analysis


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
            "Target 1": {"value": {"high": [5.0], "low": [1.0], "mid": [3.0]}},
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
            "Target 1_high": 5.0,
            "Target 1_low": 1.0,
            "Target 1_mid": 3.0,
            "target2_high": 1.0,
            "target2_low": 1.0,
            "target2_mid": 1.0,
        },
        {
            "value1": 4.0,
            "value2": 5,
            "value3": 6.3,
            "Target 1_high": 5.0,
            "Target 1_low": 1.0,
            "Target 1_mid": 3.0,
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
                "Target 1_high": 5.0,
                "Target 1_low": 1.0,
                "Target 1_mid": 3.0,
                "target2_high": 1.0,
                "target2_low": 1.0,
                "target2_mid": 1.0,
            },
            {
                "value1": 4.0,
                "value2": 5.0,
                "value3": 6.3,
                "Target 1_high": 5.0,
                "Target 1_low": 1.0,
                "Target 1_mid": 3.0,
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
        {"value1": 1.0, "value2": 2, "value3": 3.3, "Target 1_high": 5.5},
        {"value1": 4.0, "value2": 5, "value3": 6.3, "Target 1_high": 5.5},
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
            "Target 1_high.0": 5.0,
            "Target 1_low": 1.0,
            "Target 1_mid": 3.0,
            "target2_high": 1.0,
            "target2_low": 1.0,
            "target2_mid": 1.0,
            "Target 1_high": 5.5,
        },
        {
            "value1": 4.0,
            "value2": 5,
            "value3": 6.3,
            "Target 1_high.0": 5.0,
            "Target 1_low": 1.0,
            "Target 1_mid": 3.0,
            "target2_high": 1.0,
            "target2_low": 1.0,
            "target2_mid": 1.0,
            "Target 1_high": 5.5,
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


def test_analysis_factor_names(test_analysis_with_data):
    """Test that factor names are parsed correctly"""
    assert test_analysis_with_data.factor_names == ["Factor 1", "Factor 2", "Factor 3"]


def test_analysis_target_names(test_analysis_with_data):
    """Test that factor names are parsed correctly"""
    assert test_analysis_with_data.target_names == ["Target 1", "Target 2"]


def test_analysis_make_optimization_min_input(test_analysis_with_data):
    """Test that make_optimzation makes expected requests and returns a prediction"""
    pred = test_analysis_with_data.make_optimization(
        "test",
        {
            "goal": "maximize",
            "factor": {"name": "Target 1", "min": 5.0, "max": 7.0},
        },
        [{"name": "Factor 2", "min": 10, "max": 10}],
    )


def test_make_optimization_goal_not_in_analysis(test_analysis_with_data):
    """Test that a FeroError is raised if the goal doesn't include columns in the analysis"""

    with pytest.raises(FeroError) as err:
        test_analysis_with_data.make_optimization(
            "test",
            {
                "goal": "maximize",
                "factor": {"name": "factor 4", "min": 5.0, "max": 7.0},
            },
            [{"name": "Target 1", "min": 10, "max": 10}],
        )


def test_make_optimization_bad_goals(test_analysis_with_data):
    """Test that a FeroError is raised if the goal field isn't valid"""

    with pytest.raises(FeroError) as err:
        test_analysis_with_data.make_optimization(
            "test",
            {
                "goal": "circumambulate",
                "" "factor": {"name": "Factor 1", "min": 5.0, "max": 7.0},
            },
            [{"name": "Target 1", "min": 10, "max": 10}],
        )


def test_make_optimization_wrong_format_goal_factors(test_analysis_with_data):
    """Test that a FeroError is raised if the factor is malformed"""

    with pytest.raises(FeroError) as err:
        test_analysis_with_data.make_optimization(
            "test",
            {
                "goal": "maximize",
                "factor": {"name": "Factor 1", "smallest": 5.0, "max": 7.0},
            },
            [{"name": "Target 1", "min": 10, "max": 10}],
        )


def test_make_optimization_type_not_cost(test_analysis_with_data):
    """Test that a FeroError is raised if the goal config is a type, but not cost"""

    with pytest.raises(FeroError) as err:
        test_analysis_with_data.make_optimization(
            "test",
            {
                "type": "chad",
                "goal": "maximize",
                "cost_function": [
                    {"name": "Factor 1", "min": 5.0, "max": 7.0, "cost": 1000}
                ],
            },
            [{"name": "Target 1", "min": 10, "max": 10}],
        )


def test_make_optimization_type_cost_no_cost_function(test_analysis_with_data):
    """Test that a FeroError is raised if the type is cost but there is no cost_function key"""

    with pytest.raises(FeroError) as err:
        test_analysis_with_data.make_optimization(
            "test",
            {
                "type": "COST",
                "goal": "maximize",
                "factor": {"name": "Factor 1", "smallest": 5.0, "max": 7.0},
            },
            [{"name": "Target 1", "min": 10, "max": 10}],
        )


def test_make_optimization_type_cost_malformed_cost_function(test_analysis_with_data):
    """Test that a type COST optimization raises a fero error if the cost functions are malformed"""

    with pytest.raises(FeroError) as err:
        test_analysis_with_data.make_optimization(
            "test",
            {
                "type": "COST",
                "goal": "maximize",
                "cost_function": [
                    {"name": "Factor 1", "min": 5.0, "max": 7.0, "price": 1000}
                ],
            },
            [{"name": "Target 1", "min": 10, "max": 10}],
        )


def test_make_optimization_type_cost_more_than_three(test_analysis_with_data):
    """Test that a type COST optimization raises a fero error if more than 3 cost functions are specified"""

    with pytest.raises(FeroError) as err:
        test_analysis_with_data.make_optimization(
            "test",
            {
                "type": "COST",
                "goal": "maximize",
                "cost_function": [
                    {"name": "Factor 1", "min": 5.0, "max": 7.0, "cost": 1000},
                    {"name": "Factor 2", "min": 5.0, "max": 7.0, "cost": 1000},
                    {"name": "Factor 3", "min": 5.0, "max": 7.0, "cost": 1000},
                    {"name": "Factor 4", "min": 5.0, "max": 7.0, "cost": 1000},
                ],
            },
            [{"name": "Target 1", "min": 10, "max": 10}],
        )


def test_make_optimization_type_cost_target_in_function(test_analysis_with_data):
    """Test that a type COST optimization raises a Fero Error if a target is in the cost function"""

    with pytest.raises(FeroError) as err:
        test_analysis_with_data.make_optimization(
            "test",
            {
                "type": "COST",
                "goal": "maximize",
                "cost_function": [
                    {"name": "Factor 1", "min": 5.0, "max": 7.0, "cost": 1000},
                    {"name": "Target 2", "min": 5.0, "max": 7.0, "cost": 1000},
                ],
            },
            [{"name": "Target 1", "min": 10, "max": 10}],
        )


def test_make_optimization_type_cost_target_not_in_constraints(test_analysis_with_data):
    """Test that a type COST optimization raises a Fero Error if a target is not a constraint in a cost optimization"""

    with pytest.raises(FeroError) as err:
        test_analysis_with_data.make_optimization(
            "test",
            {
                "type": "COST",
                "goal": "maximize",
                "cost_function": [
                    {"name": "Factor 1", "min": 5.0, "max": 7.0, "cost": 1000},
                ],
            },
            [{"name": "Factor 2", "min": 10, "max": 10}],
        )


def test_make_optimization_constraints_not_in_analysis(test_analysis_with_data):
    """Test that a FeroError is raised if the constraints are not in the analysis"""

    with pytest.raises(FeroError) as err:
        test_analysis_with_data.make_optimization(
            "test",
            {
                "goal": "maximize",
                "factor": {"name": "Factor 1", "min": 5.0, "max": 7.0},
            },
            [{"name": "Target 4", "min": 10, "max": 10}],
        )
