"""A module to test `Analysis` and related classes."""

import io
import datetime
from uuid import uuid4
import pytest
from fero import FeroError
from unittest import mock
from fero.analysis import (
    Analysis,
    Prediction,
    CombinationConstraint,
    CombinationConstraintOperandType as operands,
    CombinationConstraintOperator as operators,
)
import pandas as pd
from pandas.testing import assert_frame_equal


@pytest.fixture
def test_analysis(
    analysis_data,
    v1_interpreted_schema,
    patched_fero_client,
    prediction_request_response,
    prediction_results_response_completed,
):
    """Create a sample `Analysis` object."""
    patched_fero_client.post.return_value = prediction_request_response
    patched_fero_client.get.return_value = prediction_results_response_completed

    analysis = Analysis(patched_fero_client, analysis_data)
    analysis._schema_cache = v1_interpreted_schema
    return analysis


@pytest.fixture
def test_analysis_with_data(test_analysis):
    """Create a sample `Analysis` holding data for the prediction simulator."""
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
                        {
                            "factor": "Category 0",
                            "median": "4",
                            "importances": [0.0004481622416081965],
                            "default": "0",
                            "dtype": "category",
                            "categories": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
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
def expected_optimization_config():
    """Create a sample optimization configuration."""
    return {
        "name": "test optimization",
        "description": "",
        "prediction_type": "O",
        "input_data": {
            "kind": "OptimizationRequest",
            "version": 1,
            "useAdaptiveGrid": False,
            "objectives": [{"factor": "Factor 2", "goal": "maximize"}],
            "basisSpecifiedColumns": [],
            "linearFunctionDefinitions": {},
            "basisValues": {
                "Factor 1": 148.02718539320762,
                "Factor 2": 148.02718539320762,
                "Factor 3": 148.02718539320762,
                "Target 1": 767.84871403268,
                "Target 2": 767.84871403268,
                "Category 0": "4",
            },
            "bounds": [
                {
                    "factor": "Factor 1",
                    "lowerBound": 60.0,
                    "upperBound": 200.0,
                    "dtype": "factor_float",
                },
                {
                    "factor": "Target 1",
                    "lowerBound": 600.0,
                    "upperBound": 700.0,
                    "dtype": "target_float",
                    "confidenceInterval": "exclude",
                },
                {
                    "factor": "Factor 2",
                    "lowerBound": 70.0,
                    "upperBound": 100.0,
                    "dtype": "factor_float",
                },
            ],
        },
    }


@pytest.fixture
def expected_optimization_config_with_combination_constraint(
    expected_optimization_config,
):
    """Add a combination constraint to the sample optimization config."""
    config = {**expected_optimization_config}
    config["input_data"]["combinationConstraints"] = {
        "conditions": [
            {
                "kind": "condition",
                "target": {
                    "value": "'Factor 1' + 'Factor 2'",
                    "kind": "formula",
                },
                "operation": {
                    "operator": ">",
                    "operand": {"kind": "constant", "value": 150.0},
                },
            }
        ],
        "join": "AND",
        "kind": "clause",
    }
    return config


@pytest.fixture
def prediction_request_response(expected_optimization_config):
    """Get a sample prediction request resopnse using the sample optimization configuration."""
    response = {
        "latest_prediction": {
            "uuid": "01bff6f2-8fb3-469e-813a-9b6cfd93e338",
            "created_by": {"id": 2, "username": "fero"},
            "created": "2020-09-30T12:35:44.897461Z",
            "modified": "2020-09-30T12:35:46.111257Z",
            "name": "g",
            "description": "",
            "prediction_type": "O",
            "progress_url": "/api/prediction_results/f0123ab1-c6f4-4bd1-b1a6-02896ba57fc7/progress/",
            "latest_results": "f0123ab1-c6f4-4bd1-b1a6-02896ba57fc7",
            "result_data": {},
            "ready": False,
            "prediction_tag": None,
            "url": "/api/analyses/21621466-b198-45be-89f9-3a5eb2c7cf48/predictions/01bff6f2-8fb3-469e-813a-9b6cfd93e338/",
            "latest_results_url": "/api/prediction_results/f0123ab1-c6f4-4bd1-b1a6-02896ba57fc7/",
        }
    }

    response["input_data"] = expected_optimization_config["input_data"]

    return response


@pytest.fixture
def prediction_results_response_started():
    """Get sample results for a prediction when the prediction has started but not completed."""
    return {
        "request": "c9448486-0c59-487f-9c9c-345d98103fcb",
        "revision_model": "e751f70e-aeb4-46ee-b860-bfc37f3767a7",
        "result_data": {},
        "state": "P",
        "prediction_type": "O",
    }


@pytest.fixture
def prediction_results_response_completed(prediction_results_response_started):
    """Get sample results for a prediction when the prediction has completed."""
    prediction_results_response_started["result_data"] = {
        "status": "SUCCESS",
        "version": 1,
        "data": {
            "values": {
                "index": [0],
                "columns": [
                    "Factor 1",
                    "Factor 2",
                    "Factor 3",
                    "Target 1 (5%)",
                    "Target 1 (Mean)",
                    "Target 1 (95%)",
                    "Target 2 (5%)",
                    "Target 2 (Mean)",
                    "Target 2 (95%)",
                ],
                "data": [
                    [
                        148.02718539320762,
                        90.234,
                        148.02718539320762,
                        666.66666,
                        677.1234,
                        699.1234,
                        700.4567,
                        677.1234,
                        699.1234,
                    ],
                ],
            }
        },
        "kind": "OptimizationResponse",
    }

    prediction_results_response_started["state"] = "C"

    return prediction_results_response_started


@pytest.fixture
def prediction_data():
    """Get sample data returned by a prediction."""
    return [
        {"value1": 1.0, "value2": 2, "value3": 3.3},
        {"value1": 4.0, "value2": 5, "value3": 6.3},
    ]


@pytest.fixture
def prediction_dataframe(prediction_data):
    """Get sample data returned by a prediction in the form of a Dataframe."""
    return pd.DataFrame(prediction_data)


@pytest.fixture
def prediction_result_data_single_v1():
    """Get sample data returned by a prediction in the form of a Dataframe."""
    return {
        "status": "SUCCESS",
        "version": 1,
        "data": {
            "Target 1": {"value": {"high": [5.0], "low": [1.0], "mid": [3.0]}},
            "target2": {"value": {"high": [1.0], "low": [1.0], "mid": [1.0]}},
        },
    }


@pytest.fixture
def prediction_result_data_bulk_v1():
    """Get sample data returned by a v1 bulk prediction."""
    return {
        "status": "SUCCESS",
        "version": 1,
        "data": {
            "Target 1": {
                "value": {"high": [5.0, 5.1], "low": [1.0, 1.1], "mid": [3.0, 3.1]}
            },
            "target2": {
                "value": {"high": [1.0, 1.1], "low": [1.0, 1.1], "mid": [1.0, 1.1]}
            },
        },
    }


@pytest.fixture
def prediction_result_data_single_v2():
    """Get sample data returned by a v2 single prediction."""
    return {
        "status": "SUCCESS",
        "version": 2,
        "data": {
            "Target 1": {
                "value": {
                    "high90": [5.0],
                    "high50": [4.0],
                    "low90": [1.0],
                    "low50": [2.0],
                    "mid": [3.0],
                }
            },
            "target2": {
                "value": {
                    "high90": [1.0],
                    "high50": [1.0],
                    "low90": [1.0],
                    "low50": [1.0],
                    "mid": [1.0],
                }
            },
        },
    }


@pytest.fixture
def prediction_result_data_bulk_v2():
    """Get sample data returned by a v2 bulk prediction."""
    return {
        "status": "SUCCESS",
        "version": 2,
        "data": {
            "Target 1": {
                "value": {
                    "high90": [5.0, 5.1],
                    "high50": [4.0, 4.1],
                    "low90": [1.0, 1.1],
                    "low50": [2.0, 2.1],
                    "mid": [3.0, 3.1],
                }
            },
            "target2": {
                "value": {
                    "high90": [1.0, 1.1],
                    "high50": [1.0, 1.1],
                    "low90": [1.0, 1.1],
                    "low50": [1.0, 1.1],
                    "mid": [1.0, 1.1],
                }
            },
        },
    }


@pytest.fixture
def batch_prediction_success_data_cache():
    """Get sample data by a successful batch prediction."""
    return {
        "request": "request_id",
        "revision_model": "revision_model_id",
        "result_data": {
            "data": {
                "filePath": "file_path.json",
                "download_url": "http://download/url/with/expiring/access/token/file.json",
            },
            "version": 1,
            "status": "SUCCESS",
            "kind": "BatchPredictionResponse",
        },
        "state": "C",
        "prediction_type": "M",
    }


@pytest.fixture
def batch_prediction_failure_data_cache():
    """Get sample data by an unsuccessful batch prediction."""
    return {
        "request": "request_id",
        "revision_model": "revision_model_id",
        "result_data": {
            "status": "FAILED",
            "message": "Something broke!",
        },
        "state": "C",
        "prediction_type": "M",
    }


def test_creates_analysis_correctly(analysis_data, patched_fero_client):
    """Test that a valid analysis is created from valid data and a valid client."""
    analysis = Analysis(patched_fero_client, analysis_data)

    assert isinstance(analysis, Analysis)
    assert str(analysis) == "<Analysis name=quality-example-data>"


def test_has_trained_model_true(analysis_data, patched_fero_client):
    """Test that has_trained_model is true if there is a latest revision model."""
    analysis = Analysis(patched_fero_client, analysis_data)
    assert analysis.has_trained_model() is True


def test_has_trained_model_false(analysis_data, patched_fero_client):
    """Test that has_trained_model is true if there is no revision model."""
    analysis_data["latest_completed_model"] = None
    analysis = Analysis(patched_fero_client, analysis_data)
    assert analysis.has_trained_model() is False


def test_gets_v1_schema_for_v1_correctly(
    analysis_data, v1_interpreted_schema, patched_fero_client
):
    """Test that we fetch the schema with the datasource endpoint for V1 analyses."""
    analysis_data["latest_completed_model_schema"] = {}
    ds_uuid = analysis_data["data_source"]
    analysis = Analysis(patched_fero_client, analysis_data)
    patched_fero_client.get.return_value = v1_interpreted_schema
    assert analysis._schema == v1_interpreted_schema
    patched_fero_client.get.assert_called_with(
        f"/api/data_sources/{ds_uuid}/latest_interpreted_schema/"
    )


def test_gets_v1_schema_for_v2_correctly(
    analysis_data, v1_interpreted_schema, patched_fero_client
):
    """Test that we fetch the schema with the process endpoint for V2 analyses."""
    analysis_data["latest_completed_model_schema"] = {}
    dummy_uuid = uuid4()
    analysis_data["process"] = str(dummy_uuid)
    analysis_data["data_source"] = None
    analysis = Analysis(patched_fero_client, analysis_data)
    patched_fero_client.get.return_value = v1_interpreted_schema
    assert analysis._schema == v1_interpreted_schema
    patched_fero_client.get.assert_called_with(
        f"/api/processes/{str(dummy_uuid)}/v1_interpreted_schema/"
    )


def test_make_prediction_dictionaries(
    analysis_data,
    patched_fero_client,
    prediction_data,
    prediction_dataframe,
    batch_prediction_success_data_cache,
):
    """Test that make prediction returns expected response as a dictionary of predictions."""
    analysis = Analysis(patched_fero_client, analysis_data)
    analysis._upload_file = mock.MagicMock()
    analysis._upload_file.return_value = "test_workspace_id"
    analysis._poll_workspace_for_prediction = mock.MagicMock()

    # Mock prediction
    test_prediction = Prediction(patched_fero_client, "result_id")
    test_prediction._data_cache = batch_prediction_success_data_cache
    test_prediction._complete = True
    test_prediction.get_results = mock.MagicMock()
    test_prediction.get_results.return_value = prediction_dataframe  # As pd.DataFrame
    analysis._poll_workspace_for_prediction.return_value = test_prediction

    results = analysis.make_prediction(prediction_data)

    assert isinstance(results, list)
    assert results == [
        {"value1": 1.0, "value2": 2, "value3": 3.3},
        {"value1": 4.0, "value2": 5, "value3": 6.3},
    ]


def test_make_prediction_dataframe(
    analysis_data,
    patched_fero_client,
    prediction_dataframe,
    batch_prediction_success_data_cache,
):
    """Test that make prediction returns expected response as a dataframe."""
    analysis = Analysis(patched_fero_client, analysis_data)
    analysis._upload_file = mock.MagicMock()
    analysis._upload_file.return_value = "test_workspace_id"
    analysis._poll_workspace_for_prediction = mock.MagicMock()

    # Mock prediction
    test_prediction = Prediction(patched_fero_client, "result_id")
    test_prediction._data_cache = batch_prediction_success_data_cache
    test_prediction._complete = True
    test_prediction.get_results = mock.MagicMock()
    test_prediction.get_results.return_value = prediction_dataframe  # As pd.DataFrame
    analysis._poll_workspace_for_prediction.return_value = test_prediction

    results = analysis.make_prediction(prediction_dataframe)

    assert isinstance(results, pd.DataFrame)
    expected = pd.DataFrame(
        [
            {"value1": 1.0, "value2": 2, "value3": 3.3},
            {"value1": 4.0, "value2": 5, "value3": 6.3},
        ]
    )
    assert_frame_equal(
        results,
        expected,
        check_like=True,
    )


def test_make_prediction_prediction_failure(
    analysis_data,
    patched_fero_client,
    prediction_data,
    batch_prediction_failure_data_cache,
):
    """Test that a FeroError is raised if a prediction fails."""
    with pytest.raises(FeroError):
        analysis = Analysis(patched_fero_client, analysis_data)
        analysis._upload_file = mock.MagicMock()
        analysis._upload_file.return_value = "test_workspace_id"
        analysis._poll_workspace_for_prediction = mock.MagicMock()
        test_prediction = Prediction(patched_fero_client, "result_id")
        test_prediction._data_cache = batch_prediction_failure_data_cache
        test_prediction._complete = True
        analysis._poll_workspace_for_prediction.return_value = test_prediction

        analysis.make_prediction(prediction_data)


def test_analysis_factor_names(test_analysis_with_data):
    """Test that factor names are parsed correctly."""
    assert test_analysis_with_data.factor_names == [
        "Factor 1",
        "Factor 2",
        "Factor 3",
        "Category 0",
    ]


def test_analysis_target_names(test_analysis_with_data):
    """Test that factor names are parsed correctly."""
    assert test_analysis_with_data.target_names == ["Target 1", "Target 2"]


def test_make_optimization_fault_blueprint(test_analysis_with_data):
    """Test that a FeroError is raised if this is a fault blueprint."""
    test_analysis_with_data._data["blueprint_name"] = "fault"

    with pytest.raises(FeroError):
        test_analysis_with_data.make_optimization(
            "test optimization",
            {
                "goal": "maximize",
                "factor": {"name": "Factor 2", "min": 70.0, "max": 100.0},
            },
            [
                {"name": "Factor 1", "min": 60.0, "max": 200.0},
                {"name": "Target 1", "min": 600.0, "max": 700.0},
            ],
        )


def test_make_optimization_goal_not_in_analysis(test_analysis_with_data):
    """Test that a FeroError is raised if the goal doesn't include columns in the analysis."""
    with pytest.raises(FeroError):
        test_analysis_with_data.make_optimization(
            "test",
            {
                "goal": "maximize",
                "factor": {"name": "factor 4", "min": 5.0, "max": 7.0},
            },
            [{"name": "Target 1", "min": 10, "max": 11}],
        )


def test_make_optimization_no_constraints(test_analysis_with_data):
    """Test that a FeroError is raised if constraints are specified."""
    with pytest.raises(FeroError):
        test_analysis_with_data.make_optimization(
            "test",
            {
                "goal": "maximize",
                "factor": {"name": "Target 1", "min": 5.0, "max": 7.0},
            },
            [],
        )


def test_make_optimization_bad_goals(test_analysis_with_data):
    """Test that a FeroError is raised if the goal field isn't valid."""
    with pytest.raises(FeroError):
        test_analysis_with_data.make_optimization(
            "test",
            {
                "goal": "circumambulate",
                "factor": {"name": "Factor 1", "min": 5.0, "max": 7.0},
            },
            [{"name": "Target 1", "min": 10, "max": 11}],
        )


def test_make_optimization_wrong_format_goal_factors(test_analysis_with_data):
    """Test that a FeroError is raised if the factor is malformed."""
    with pytest.raises(FeroError):
        test_analysis_with_data.make_optimization(
            "test",
            {
                "goal": "maximize",
                "factor": {"name": "Factor 1", "smallest": 5.0, "max": 7.0},
            },
            [{"name": "Target 1", "min": 10, "max": 11}],
        )


def test_make_optimization_type_not_cost(test_analysis_with_data):
    """Test that a FeroError is raised if the goal config is a type, but not cost."""
    with pytest.raises(FeroError):
        test_analysis_with_data.make_optimization(
            "test",
            {
                "type": "chad",
                "goal": "maximize",
                "cost_function": [
                    {"name": "Factor 1", "min": 5.0, "max": 7.0, "cost": 1000}
                ],
            },
            [{"name": "Target 1", "min": 10, "max": 11}],
        )


def test_make_optimization_type_cost_no_cost_function(test_analysis_with_data):
    """Test that a FeroError is raised if the type is cost but there is no cost_function key."""
    with pytest.raises(FeroError):
        test_analysis_with_data.make_optimization(
            "test",
            {
                "type": "COST",
                "goal": "maximize",
                "factor": {"name": "Factor 1", "smallest": 5.0, "max": 7.0},
            },
            [{"name": "Target 1", "min": 10, "max": 11}],
        )


def test_make_optimization_type_cost_malformed_cost_function(test_analysis_with_data):
    """Test that a type COST optimization raises a fero error if the cost functions are malformed."""
    with pytest.raises(FeroError):
        test_analysis_with_data.make_optimization(
            "test",
            {
                "type": "COST",
                "goal": "maximize",
                "cost_function": [
                    {"name": "Factor 1", "min": 5.0, "max": 7.0, "price": 1000}
                ],
            },
            [{"name": "Target 1", "min": 10, "max": 11}],
        )


def test_make_optimization_type_cost_more_than_three(test_analysis_with_data):
    """Test that a type COST optimization raises a fero error if more than 3 cost functions are specified."""
    with pytest.raises(FeroError):
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
            [{"name": "Target 1", "min": 10, "max": 11}],
        )


def test_make_optimization_type_cost_target_not_in_function_or_constraints(
    test_analysis_with_data,
):
    """Test that a type COST optimization raises a Fero Error if a target not in either the cost function or the constaints."""
    with pytest.raises(FeroError):
        test_analysis_with_data.make_optimization(
            "test",
            {
                "type": "COST",
                "goal": "maximize",
                "cost_function": [
                    {"name": "Factor 1", "min": 5.0, "max": 7.0, "cost": 1000},
                ],
            },
            [{"name": "Factor 2", "min": 10, "max": 11}],
        )


def test_make_optimization_constraints_not_in_analysis(test_analysis_with_data):
    """Test that a FeroError is raised if the constraints are not in the analysis."""
    with pytest.raises(FeroError):
        test_analysis_with_data.make_optimization(
            "test",
            {
                "goal": "maximize",
                "factor": {"name": "Factor 1", "min": 5.0, "max": 7.0},
            },
            [{"name": "Target 4", "min": 10, "max": 11}],
        )


def test_make_optimization_categorical_goal(test_analysis_with_data):
    """Test that a FeroError is raised if the goal is category."""
    with pytest.raises(FeroError):
        test_analysis_with_data.make_optimization(
            "test",
            {
                "goal": "maximize",
                "factor": {"name": "Category 0", "min": 5.0, "max": 7.0},
            },
            [{"name": "Target 4", "min": 10, "max": 11}],
        )


def test_make_optimization_type_cost_categorical_function(test_analysis_with_data):
    """Test that a type COST optimization raises a Fero Error if a category is a cost funciton."""
    with pytest.raises(FeroError):
        test_analysis_with_data.make_optimization(
            "test",
            {
                "type": "COST",
                "goal": "maximize",
                "cost_function": [
                    {"name": "Category 0", "min": 5.0, "max": 7.0, "cost": 1000},
                ],
            },
            [{"name": "Target 1", "min": 10, "max": 11}],
        )


def test_make_optimization_combination_constraint_unknown_column(
    test_analysis_with_data,
):
    """Test that a combination constraint raises a Fero Error if it references an unknown column."""
    with pytest.raises(FeroError):
        test_analysis_with_data.make_optimization(
            "test optimization",
            {
                "goal": "maximize",
                "factor": {"name": "Factor 2", "min": 70.0, "max": 100.0},
            },
            [
                {"name": "Factor 1", "min": 60.0, "max": 200.0},
                {"name": "Target 1", "min": 600.0, "max": 700.0},
            ],
            include_confidence_intervals=False,
            combination_constraints=[
                CombinationConstraint(
                    ("'Factor 1' + 'Factor 2'", operands.FORMULA),
                    operators.GREATER_THAN,
                    ("Unknown Column", operands.COLUMN),
                )
            ],
        )


def test_analysis_make_optimization_simple_case(
    test_analysis_with_data, expected_optimization_config_with_combination_constraint
):
    """Test that make_optimzation makes expected requests and returns a prediction."""
    pred = test_analysis_with_data.make_optimization(
        "test optimization",
        {
            "goal": "maximize",
            "factor": {"name": "Factor 2", "min": 70.0, "max": 100.0},
        },
        [
            {"name": "Factor 1", "min": 60.0, "max": 200.0},
            {"name": "Target 1", "min": 600.0, "max": 700.0},
        ],
        include_confidence_intervals=False,
        combination_constraints=[
            CombinationConstraint(
                ("'Factor 1' + 'Factor 2'", operands.FORMULA),
                operators.GREATER_THAN,
                (150.0, operands.CONSTANT),
            )
        ],
    )

    assert pred.result_id == "f0123ab1-c6f4-4bd1-b1a6-02896ba57fc7"
    test_analysis_with_data._client.post.assert_called_with(
        f"/api/analyses/{str(test_analysis_with_data.uuid)}/workspaces/",
        {
            "request": expected_optimization_config_with_combination_constraint,
            "visible": False,
        },
    )
    test_analysis_with_data._client.get.assert_called()


def test_analysis_make_optimization_simple_case_categorical(
    test_analysis_with_data, expected_optimization_config
):
    """Test that make_optimzation makes expected requests with a categorical value and returns a prediction."""
    pred = test_analysis_with_data.make_optimization(
        "test optimization",
        {
            "goal": "maximize",
            "factor": {"name": "Factor 2", "min": 70.0, "max": 100.0},
        },
        [
            {"name": "Category 0"},
            {"name": "Target 1", "min": 600.0, "max": 700.0},
        ],
        include_confidence_intervals=False,
    )
    expected_optimization_config["input_data"]["bounds"][0] = {
        "factor": "Category 0",
        "dtype": "factor_category",
    }
    assert pred.result_id == "f0123ab1-c6f4-4bd1-b1a6-02896ba57fc7"
    test_analysis_with_data._client.post.assert_called_with(
        f"/api/analyses/{str(test_analysis_with_data.uuid)}/workspaces/",
        {"request": expected_optimization_config, "visible": False},
    )
    test_analysis_with_data._client.get.assert_called()


def test_analysis_make_optimization_cost(
    test_analysis_with_data, expected_optimization_config
):
    """Test that make_optimzation makes expected requests and prediction for a cost optimization."""
    expected_optimization_config["input_data"]["objectives"] = [
        {"factor": "FERO_COST_FUNCTION", "goal": "maximize"}
    ]

    expected_optimization_config["input_data"]["bounds"] = [
        {
            "factor": "Factor 1",
            "lowerBound": 60.0,
            "upperBound": 200.0,
            "dtype": "factor_float",
        },
        {
            "factor": "Target 1",
            "lowerBound": 600.0,
            "upperBound": 700.0,
            "dtype": "target_float",
            "confidenceInterval": "exclude",
        },
        {
            "factor": "Factor 2",
            "lowerBound": 70.0,
            "upperBound": 100.0,
            "dtype": "factor_float",
        },
        {
            "factor": "Factor 3",
            "lowerBound": 100.0,
            "upperBound": 150.0,
            "dtype": "factor_float",
        },
        {
            "factor": "FERO_COST_FUNCTION",
            "lowerBound": 2700.00,
            "upperBound": 4000.0,
            "dtype": "function",
        },
    ]

    expected_optimization_config["input_data"]["linearFunctionDefinitions"] = {
        "FERO_COST_FUNCTION": {
            "Factor 2": 10.0,
            "Factor 3": 20.0,
        }
    }

    pred = test_analysis_with_data.make_optimization(
        "test optimization",
        {
            "type": "COST",
            "goal": "maximize",
            "cost_function": [
                {"name": "Factor 2", "min": 70.0, "max": 100.0, "cost": 10.0},
                {"name": "Factor 3", "min": 100.0, "max": 150.0, "cost": 20.0},
            ],
        },
        [
            {"name": "Factor 1", "min": 60.0, "max": 200.0},
            {"name": "Target 1", "min": 600.0, "max": 700.0},
        ],
        include_confidence_intervals=False,
    )

    assert pred.result_id == "f0123ab1-c6f4-4bd1-b1a6-02896ba57fc7"
    test_analysis_with_data._client.post.assert_called_with(
        f"/api/analyses/{str(test_analysis_with_data.uuid)}/workspaces/",
        {"request": expected_optimization_config, "visible": False},
    )


def test_analysis_make_optimization_simple_case_basis_override(
    test_analysis_with_data, expected_optimization_config
):
    """Test that make_optimzation makes expected requests and returns a prediction with basis_value overrides."""
    expected_optimization_config["input_data"]["basisValues"]["Factor 1"] = 150.0
    pred = test_analysis_with_data.make_optimization(
        "test optimization",
        {
            "goal": "maximize",
            "factor": {"name": "Factor 2", "min": 70.0, "max": 100.0},
        },
        [
            {"name": "Factor 1", "min": 60.0, "max": 200.0},
            {"name": "Target 1", "min": 600.0, "max": 700.0},
        ],
        fixed_factors={"Factor 1": 150.00},
        include_confidence_intervals=False,
    )

    assert pred.result_id == "f0123ab1-c6f4-4bd1-b1a6-02896ba57fc7"
    test_analysis_with_data._client.post.assert_called_with(
        f"/api/analyses/{str(test_analysis_with_data.uuid)}/workspaces/",
        {"request": expected_optimization_config, "visible": False},
    )


def test_analysis_make_optimization_include_confidence(
    test_analysis_with_data, expected_optimization_config
):
    """Test that make_optimzation makes expected requests and returns a prediction with confidence intervals requested."""
    expected_optimization_config["input_data"]["bounds"][1][
        "confidenceInterval"
    ] = "include"
    pred = test_analysis_with_data.make_optimization(
        "test optimization",
        {
            "goal": "maximize",
            "factor": {"name": "Factor 2", "min": 70.0, "max": 100.0},
        },
        [
            {"name": "Factor 1", "min": 60.0, "max": 200.0},
            {"name": "Target 1", "min": 600.0, "max": 700.0},
        ],
        include_confidence_intervals=True,
    )

    assert pred.result_id == "f0123ab1-c6f4-4bd1-b1a6-02896ba57fc7"
    test_analysis_with_data._client.post.assert_called_with(
        f"/api/analyses/{str(test_analysis_with_data.uuid)}/workspaces/",
        {"request": expected_optimization_config, "visible": False},
    )


def test_analysis_make_optimization_synchronous_false(
    test_analysis_with_data, expected_optimization_config
):
    """Test that make_optimzation makes expected requests and returns a prediction with synchronous false."""
    pred = test_analysis_with_data.make_optimization(
        "test optimization",
        {
            "goal": "maximize",
            "factor": {"name": "Factor 2", "min": 70.0, "max": 100.0},
        },
        [
            {"name": "Factor 1", "min": 60.0, "max": 200.0},
            {"name": "Target 1", "min": 600.0, "max": 700.0},
        ],
        include_confidence_intervals=False,
        synchronous=False,
    )

    assert pred.result_id == "f0123ab1-c6f4-4bd1-b1a6-02896ba57fc7"
    test_analysis_with_data._client.post.assert_called_with(
        f"/api/analyses/{str(test_analysis_with_data.uuid)}/workspaces/",
        {"request": expected_optimization_config, "visible": False},
    )
    test_analysis_with_data._client.get.assert_not_called()


def test_prediction_complete_false(
    patched_fero_client, prediction_results_response_started
):
    """Test that prediction.complete is false if not completed."""
    patched_fero_client.get.return_value = prediction_results_response_started

    pred = Prediction(patched_fero_client, "c9448486-0c59-487f-9c9c-345d98103fcb")
    assert pred.complete is False


def test_prediction_complete_true(
    patched_fero_client, prediction_results_response_completed
):
    """Test that prediction.complete is true if completed."""
    patched_fero_client.get.return_value = prediction_results_response_completed

    pred = Prediction(patched_fero_client, "c9448486-0c59-487f-9c9c-345d98103fcb")
    assert pred.complete is True


def test_prediction_get_result_not_complete(
    patched_fero_client, prediction_results_response_started
):
    """Test that a Fero Error is raised if get_result is called before a prediction is complete."""
    patched_fero_client.get.return_value = prediction_results_response_started
    pred = Prediction(patched_fero_client, "c9448486-0c59-487f-9c9c-345d98103fcb")

    with pytest.raises(FeroError) as err:
        pred.get_results()

    assert err.value.message == "Prediction is not complete."


def test_prediction_get_result_dataframe(
    patched_fero_client, prediction_results_response_completed
):
    """Test that the expected data frame is generated by get_results."""
    patched_fero_client.get.return_value = prediction_results_response_completed
    pred = Prediction(patched_fero_client, "c9448486-0c59-487f-9c9c-345d98103fcb")

    excepted = pd.DataFrame(
        [
            [
                148.02718539320762,
                90.234,
                148.02718539320762,
                666.66666,
                677.1234,
                699.1234,
                700.4567,
                677.1234,
                699.1234,
            ],
        ],
        index=[0],
        columns=[
            "Factor 1",
            "Factor 2",
            "Factor 3",
            "Target 1 (5%)",
            "Target 1 (Mean)",
            "Target 1 (95%)",
            "Target 2 (5%)",
            "Target 2 (Mean)",
            "Target 2 (95%)",
        ],
    )

    assert_frame_equal(excepted, pred.get_results())


def test_prediction_get_result_dict(
    patched_fero_client, prediction_results_response_completed
):
    """Test that the expected list of dicts is returned for format records."""
    patched_fero_client.get.return_value = prediction_results_response_completed
    pred = Prediction(patched_fero_client, "c9448486-0c59-487f-9c9c-345d98103fcb")

    excepted = [
        {
            "Factor 1": 148.02718539320762,
            "Factor 2": 90.234,
            "Factor 3": 148.02718539320762,
            "Target 1 (5%)": 666.66666,
            "Target 1 (Mean)": 677.1234,
            "Target 1 (95%)": 699.1234,
            "Target 2 (5%)": 700.4567,
            "Target 2 (Mean)": 677.1234,
            "Target 2 (95%)": 699.1234,
        }
    ]
    assert excepted == pred.get_results(format="records")


def test_analysis_upload_file_makes_expected_calls_azure(
    analysis_data, patched_fero_client
):
    """Test that _upload_file makes the expected calls and returns the expected values."""
    patched_fero_client.post.return_value = {
        "upload_type": "azure",
        "workspace_id": "uuid",
        "other_info": "test_value",
    }
    analysis = Analysis(patched_fero_client, analysis_data)
    analysis._client._azure_upload = mock.MagicMock()
    analysis._client._s3_upload = mock.MagicMock()
    fp = io.StringIO("test_data")
    analysis._upload_file(fp, "test_tag", "test_type")

    patched_fero_client.post.assert_has_calls(
        [
            mock.call(
                f"/api/analyses/{str(analysis.uuid)}/workspaces/inbox_url/",
                {"file_tag": "test_tag", "prediction_type": "test_type"},
            )
        ]
    )
    analysis._client._azure_upload.assert_has_calls(
        [mock.call({"upload_type": "azure", "other_info": "test_value"}, fp)]
    )
    analysis._client._s3_upload.assert_has_calls([])


def test_analysis_upload_file_makes_expected_calls_s3(
    analysis_data, patched_fero_client
):
    """Test that _upload_file makes the expected calls and returns the expected values."""
    patched_fero_client.post.return_value = {
        "upload_type": "s3",
        "workspace_id": "uuid",
        "other_info": "test_value",
    }
    analysis = Analysis(patched_fero_client, analysis_data)
    analysis._client._azure_upload = mock.MagicMock()
    analysis._client._s3_upload = mock.MagicMock()
    fp = io.StringIO("test_data")
    analysis._upload_file(fp, "test_tag", "test_type")

    patched_fero_client.post.assert_has_calls(
        [
            mock.call(
                f"/api/analyses/{str(analysis.uuid)}/workspaces/inbox_url/",
                {"file_tag": "test_tag", "prediction_type": "test_type"},
            )
        ]
    )
    analysis._client._s3_upload.assert_has_calls(
        [mock.call({"upload_type": "s3", "other_info": "test_value"}, "test_tag", fp)]
    )
    analysis._client._azure_upload.assert_has_calls([])


def test_is_retraining_not_retraining(analysis_data, patched_fero_client):
    """Test that is_retraining() returns false if the analysis is not training."""
    patched_fero_client.get.return_value = analysis_data.copy()
    analysis_data["latest_revision_model_state"] = "T"
    analysis = Analysis(patched_fero_client, analysis_data)
    assert analysis.is_retraining() is False
    patched_fero_client.get.assert_called_with(
        f"/api/analyses/{analysis_data['uuid']}/"
    )


def test_is_retraining_retraining(analysis_data, patched_fero_client):
    """Test that is_retraining() returns true if the analysis is training."""
    analysis_data["latest_revision_model_state"] = "T"
    patched_fero_client.get.return_value = analysis_data.copy()

    analysis = Analysis(patched_fero_client, analysis_data)
    assert analysis.is_retraining() is True
    patched_fero_client.get.assert_called_with(
        f"/api/analyses/{analysis_data['uuid']}/"
    )


def test_not_retraining_new_version_race_condition(analysis_data, patched_fero_client):
    """Test that is_retraining is called again if the latest version."""
    second_version = analysis_data.copy()
    second_version["latest_revision"] = second_version["latest_revision"] + 1

    patched_fero_client.get.side_effect = [
        analysis_data,
        second_version,
        second_version,
        second_version,
    ]

    analysis = Analysis(patched_fero_client, analysis_data)
    assert analysis.is_retraining() is False
    patched_fero_client.get.assert_has_calls(
        [mock.call(f"/api/analyses/{analysis_data['uuid']}/") for _ in range(4)]
    )


def test_revise_makes_expected_calls(analysis_data, patched_fero_client, revision_data):
    """Test that revise makes the expected calls to the server."""
    analysis = Analysis(patched_fero_client, analysis_data)
    patched_fero_client.get.side_effect = [
        analysis_data,
        analysis_data,
        revision_data,
        analysis_data,
        analysis_data,
    ]
    patched_fero_client.post.return_value = revision_data

    with mock.patch("datetime.datetime") as dt:
        now = datetime.datetime(2021, 8, 4, 0, 0, 1)
        dt.now.return_value = now
        analysis.revise()

        patched_fero_client.get.assert_has_calls(
            [
                mock.call(f"/api/analyses/{analysis_data['uuid']}/"),
                mock.call(
                    f"/api/analyses/{analysis_data['uuid']}/revisions/{analysis_data['latest_revision']}/"
                ),
                mock.call(f"/api/analyses/{analysis_data['uuid']}/"),
            ]
        )
        patched_fero_client.post.assert_called_with(
            f"/api/analyses/{analysis_data['uuid']}/revisions/",
            {
                "description": f"Revision created via fero client on {now.isoformat()}",
                "configured_blueprint": revision_data["configured_blueprint"],
            },
        )


def test_revise_does_not_revise_if_training(analysis_data, patched_fero_client):
    """Test a revision is not made if the analysis is currently being trained."""
    analysis = Analysis(patched_fero_client, analysis_data)
    analysis_data["latest_revision_model_state"] = "T"
    patched_fero_client.get.return_value = analysis_data

    analysis.revise()
    patched_fero_client.get.called_once_with(f"/api/analyses/{analysis_data['uuid']}/")
    patched_fero_client.post.assert_not_called()
