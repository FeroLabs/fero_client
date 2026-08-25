"""A module to test `Analysis.get_live_predictions` and the objects it returns."""

import datetime

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fero import FeroError
from fero.analysis import Analysis
from fero.live_predictions import (
    FlexibleLiveOptimization,
    FlexibleLivePrediction,
    LiveOptimization,
    LivePrediction,
    LivePredictionSort,
    LivePredictionType,
)

ANALYSIS_UUID = "b6c56b58-5b03-4f8d-9c1a-5b1a4f88df4a"
PREDICTIONS_URL = f"/api/analyses/{ANALYSIS_UUID}/get_live_predictions/"


def envelope(prediction_type, **overrides):
    """Build the common fields the API reports for every live prediction."""
    data = {
        "uuid": "0d1c3fd4-3e28-4ae5-bd3f-9f2c62c1d7a1",
        "workspace_id": "1f4f0d3a-cc1e-4d47-9d1a-9b1d4f9a4f11",
        "analysis_id": ANALYSIS_UUID,
        "name": "Live Prediction",
        "description": "",
        "prediction_tag": "gc-p-1234",
        "type": prediction_type,
        "created": "2026-08-24T12:00:00.123456Z",
        "modified": "2026-08-24T12:00:05.123456Z",
        "created_by": "admin",
        "live_order_value": "2026-08-24T11:59:00",
        "revision_model_id": "42",
        "state": "C",
        "complete": True,
        "status": "SUCCESS",
        "message": None,
        "basis": {"Factor 1": 1.0},
        "targets": None,
        "values": None,
        "scenarios": None,
        "default_scenario_index": None,
    }
    data.update(overrides)
    return data


def targets(mid):
    """Build the target intervals the API reports for a point prediction."""
    return {
        "Target 1": {
            "low90": mid - 20,
            "low50": mid - 10,
            "mid": mid,
            "high50": mid + 10,
            "high90": mid + 20,
        }
    }


def optimization_values(value):
    """Build the split-orient frame the API reports for an optimization."""
    return {
        "columns": ["Factor 1", "Target 1"],
        "index": [0],
        "data": [[value, value * 2]],
    }


@pytest.fixture
def live_analysis(analysis_data, patched_fero_client):
    """Create an `Analysis` whose client returns canned live prediction responses."""
    analysis_data = dict(analysis_data, uuid=ANALYSIS_UUID)
    return Analysis(patched_fero_client, analysis_data)


def respond_with(analysis, results, **extra):
    """Point the analysis' client at a canned `get_live_predictions` response."""
    response = {
        "type": "prediction",
        "sort": "-created",
        "limit": 10,
        "results": results,
    }
    response.update(extra)
    analysis._client.get.return_value = response


def test_requests_documented_defaults(live_analysis):
    """Calling with no arguments requests 10 newest-first point predictions."""
    respond_with(live_analysis, [])

    assert live_analysis.get_live_predictions() == []
    live_analysis._client.get.assert_called_once_with(
        PREDICTIONS_URL,
        params={"type": "prediction", "sort": "-created", "limit": 10},
    )


@pytest.mark.parametrize(
    "prediction_type,sort",
    [
        (LivePredictionType.FLEXIBLE_PREDICTION, LivePredictionSort.OLDEST_FIRST),
        (LivePredictionType.OPTIMIZATION, LivePredictionSort.LIVE_ORDER_DESCENDING),
        (
            LivePredictionType.FLEXIBLE_OPTIMIZATION,
            LivePredictionSort.LIVE_ORDER_ASCENDING,
        ),
    ],
)
def test_enum_arguments_are_sent_as_their_values(live_analysis, prediction_type, sort):
    """Enum members are sent to the API as their string values."""
    respond_with(live_analysis, [])

    live_analysis.get_live_predictions(type=prediction_type, sort=sort, limit=25)

    live_analysis._client.get.assert_called_once_with(
        PREDICTIONS_URL,
        params={"type": prediction_type.value, "sort": sort.value, "limit": 25},
    )


def test_plain_strings_are_accepted(live_analysis):
    """Callers may pass the string values instead of importing the enums."""
    respond_with(live_analysis, [])

    live_analysis.get_live_predictions(type="optimization", sort="created")

    live_analysis._client.get.assert_called_once_with(
        PREDICTIONS_URL,
        params={"type": "optimization", "sort": "created", "limit": 10},
    )


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"type": "predictions"}, "not a valid type"),
        ({"type": "L"}, "not a valid type"),
        ({"sort": "-modified"}, "not a valid sort"),
        ({"limit": 0}, "must be between 1 and 1000"),
        ({"limit": 1001}, "must be between 1 and 1000"),
        ({"limit": "10"}, "must be an integer"),
        ({"limit": 10.5}, "must be an integer"),
        ({"limit": True}, "must be an integer"),
    ],
)
def test_invalid_arguments_fail_before_the_request(live_analysis, kwargs, expected):
    """Bad arguments raise locally, so no request is made at all."""
    respond_with(live_analysis, [])

    with pytest.raises(FeroError, match=expected):
        live_analysis.get_live_predictions(**kwargs)

    live_analysis._client.get.assert_not_called()


def test_limit_bounds_are_inclusive(live_analysis):
    """Both ends of the documented limit range are accepted."""
    respond_with(live_analysis, [])

    for limit in (1, 1000):
        live_analysis.get_live_predictions(limit=limit)
        assert live_analysis._client.get.call_args.kwargs["params"]["limit"] == limit


def test_prediction_exposes_targets_and_metadata(live_analysis):
    """A point prediction reports its targets and the envelope fields."""
    respond_with(live_analysis, [envelope("prediction", targets=targets(100))])

    prediction = live_analysis.get_live_predictions()[0]

    assert isinstance(prediction, LivePrediction)
    assert prediction.uuid == "0d1c3fd4-3e28-4ae5-bd3f-9f2c62c1d7a1"
    assert prediction.analysis_id == ANALYSIS_UUID
    assert prediction.prediction_tag == "gc-p-1234"
    assert prediction.created_by == "admin"
    assert prediction.complete is True
    assert prediction.basis == {"Factor 1": 1.0}
    assert prediction.created == datetime.datetime(
        2026, 8, 24, 12, 0, 0, 123456, tzinfo=datetime.timezone.utc
    )
    assert prediction.targets["Target 1"].mid == 100
    assert prediction.targets["Target 1"].low90 == 80
    assert prediction.targets["Target 1"].high90 == 120


def test_prediction_to_dataframe(live_analysis):
    """A point prediction converts to a frame of one row per target."""
    respond_with(live_analysis, [envelope("prediction", targets=targets(100))])

    frame = live_analysis.get_live_predictions()[0].to_dataframe()

    assert_frame_equal(
        frame,
        pd.DataFrame(
            [[80, 90, 100, 110, 120]],
            index=["Target 1"],
            columns=["low90", "low50", "mid", "high50", "high90"],
        ),
    )


def test_flexible_prediction_exposes_scenarios(live_analysis):
    """A flexible prediction is one object holding a scenario per flex basis."""
    respond_with(
        live_analysis,
        [
            envelope(
                "flexible_prediction",
                scenarios=[
                    {
                        "index": 0,
                        "basis": {"Grade": "A"},
                        "status": "SUCCESS",
                        "message": None,
                        "targets": targets(100),
                    },
                    {
                        "index": 1,
                        "basis": {"Grade": "B"},
                        "status": "SUCCESS",
                        "message": None,
                        "targets": targets(200),
                    },
                ],
                default_scenario_index=1,
            )
        ],
    )

    prediction = live_analysis.get_live_predictions(
        type=LivePredictionType.FLEXIBLE_PREDICTION
    )[0]

    assert isinstance(prediction, FlexibleLivePrediction)
    assert len(prediction.scenarios) == 2
    assert prediction.scenarios[0].basis == {"Grade": "A"}
    assert prediction.scenarios[1].targets["Target 1"].mid == 200
    assert prediction.default_scenario is prediction.scenarios[1]


def test_flexible_prediction_to_dataframe_is_indexed_by_scenario(live_analysis):
    """A flexible prediction converts to a frame indexed by scenario and target."""
    respond_with(
        live_analysis,
        [
            envelope(
                "flexible_prediction",
                scenarios=[
                    {
                        "index": 0,
                        "basis": {},
                        "status": "SUCCESS",
                        "message": None,
                        "targets": targets(100),
                    },
                    {
                        "index": 1,
                        "basis": {},
                        "status": "SUCCESS",
                        "message": None,
                        "targets": targets(200),
                    },
                ],
                default_scenario_index=0,
            )
        ],
    )

    frame = live_analysis.get_live_predictions(
        type=LivePredictionType.FLEXIBLE_PREDICTION
    )[0].to_dataframe()

    assert list(frame.index) == [(0, "Target 1"), (1, "Target 1")]
    assert list(frame["mid"]) == [100, 200]


def test_flexible_prediction_default_scenario_is_none_when_unset(live_analysis):
    """A flexible prediction with no scenarios has no default scenario."""
    respond_with(
        live_analysis,
        [envelope("flexible_prediction", scenarios=[], default_scenario_index=None)],
    )

    prediction = live_analysis.get_live_predictions(
        type=LivePredictionType.FLEXIBLE_PREDICTION
    )[0]

    assert prediction.scenarios == []
    assert prediction.default_scenario is None
    assert prediction.to_dataframe().empty


def test_failed_flexible_scenario_is_reported(live_analysis):
    """A scenario that failed is listed with its message and no targets."""
    respond_with(
        live_analysis,
        [
            envelope(
                "flexible_prediction",
                scenarios=[
                    {
                        "index": 0,
                        "basis": None,
                        "status": "FAILURE",
                        "message": "unknown category",
                        "targets": {},
                    }
                ],
                default_scenario_index=0,
            )
        ],
    )

    scenario = live_analysis.get_live_predictions(
        type=LivePredictionType.FLEXIBLE_PREDICTION
    )[0].scenarios[0]

    assert scenario.status == "FAILURE"
    assert scenario.message == "unknown category"
    assert scenario.targets == {}


def test_optimization_to_dataframe(live_analysis):
    """An optimization converts its optimal values to a frame."""
    respond_with(
        live_analysis, [envelope("optimization", values=optimization_values(3.0))]
    )

    optimization = live_analysis.get_live_predictions(
        type=LivePredictionType.OPTIMIZATION
    )[0]

    assert isinstance(optimization, LiveOptimization)
    assert_frame_equal(
        optimization.to_dataframe(),
        pd.DataFrame([[3.0, 6.0]], columns=["Factor 1", "Target 1"], index=[0]),
    )


def test_optimization_exposes_optimal_values_without_pandas(live_analysis):
    """An optimization's results are reachable as plain dictionaries."""
    respond_with(
        live_analysis, [envelope("optimization", values=optimization_values(3.0))]
    )

    optimization = live_analysis.get_live_predictions(
        type=LivePredictionType.OPTIMIZATION
    )[0]

    assert optimization.optimal_values == [{"Factor 1": 3.0, "Target 1": 6.0}]


def test_optimization_exposes_every_optimal_solution(live_analysis):
    """An optimization with several equally optimal solutions reports all of them."""
    respond_with(
        live_analysis,
        [
            envelope(
                "optimization",
                values={
                    "columns": ["Factor 1", "Target 1"],
                    "index": [0, 1],
                    "data": [[3.0, 6.0], [4.0, 8.0]],
                },
            )
        ],
    )

    optimization = live_analysis.get_live_predictions(
        type=LivePredictionType.OPTIMIZATION
    )[0]

    assert optimization.optimal_values == [
        {"Factor 1": 3.0, "Target 1": 6.0},
        {"Factor 1": 4.0, "Target 1": 8.0},
    ]


@pytest.mark.parametrize("values", [None, {"columns": [], "index": [], "data": []}])
def test_optimization_without_values_has_no_optimal_values(live_analysis, values):
    """An optimization that produced no solution reports an empty list, not an error."""
    respond_with(live_analysis, [envelope("optimization", values=values)])

    optimization = live_analysis.get_live_predictions(
        type=LivePredictionType.OPTIMIZATION
    )[0]

    assert optimization.optimal_values == []


def test_incomplete_optimization_reports_empty_optimal_values(live_analysis):
    """Unlike `to_dataframe`, direct access reports what is there rather than raising."""
    respond_with(
        live_analysis,
        [envelope("optimization", complete=False, status=None, state="P", values=None)],
    )

    optimization = live_analysis.get_live_predictions(
        type=LivePredictionType.OPTIMIZATION
    )[0]

    assert optimization.optimal_values == []
    with pytest.raises(FeroError, match="is not complete"):
        optimization.to_dataframe()


def test_flexible_optimization_scenarios_expose_optimal_values(live_analysis):
    """Each scenario of a flexible optimization exposes its own plain results."""
    respond_with(
        live_analysis,
        [
            envelope(
                "flexible_optimization",
                scenarios=[
                    {
                        "index": 0,
                        "basis": {"Grade": "A"},
                        "values": optimization_values(3.0),
                    },
                    {
                        "index": 1,
                        "basis": {"Grade": "B"},
                        "values": optimization_values(4.0),
                    },
                ],
                default_scenario_index=1,
            )
        ],
    )

    optimization = live_analysis.get_live_predictions(
        type=LivePredictionType.FLEXIBLE_OPTIMIZATION
    )[0]

    assert optimization.scenarios[0].optimal_values == [
        {"Factor 1": 3.0, "Target 1": 6.0}
    ]
    assert optimization.default_scenario.optimal_values == [
        {"Factor 1": 4.0, "Target 1": 8.0}
    ]


def test_prediction_targets_convert_to_plain_dictionaries(live_analysis):
    """A prediction's targets are reachable as plain dictionaries."""
    respond_with(live_analysis, [envelope("prediction", targets=targets(100))])

    prediction = live_analysis.get_live_predictions()[0]

    assert prediction.targets["Target 1"].to_dict() == {
        "low90": 80,
        "low50": 90,
        "mid": 100,
        "high50": 110,
        "high90": 120,
    }


def test_optimization_without_values_returns_none(live_analysis):
    """An optimization that produced no values converts to `None`, not an error."""
    respond_with(live_analysis, [envelope("optimization", values=None)])

    optimization = live_analysis.get_live_predictions(
        type=LivePredictionType.OPTIMIZATION
    )[0]

    assert optimization.to_dataframe() is None


def test_flexible_optimization_exposes_scenarios(live_analysis):
    """A flexible optimization holds a scenario per basis and names the riskiest."""
    respond_with(
        live_analysis,
        [
            envelope(
                "flexible_optimization",
                scenarios=[
                    {
                        "index": 0,
                        "basis": {"Grade": "A"},
                        "values": optimization_values(3.0),
                    },
                    {
                        "index": 1,
                        "basis": {"Grade": "B"},
                        "values": optimization_values(4.0),
                    },
                ],
                default_scenario_index=1,
            )
        ],
    )

    optimization = live_analysis.get_live_predictions(
        type=LivePredictionType.FLEXIBLE_OPTIMIZATION
    )[0]

    assert isinstance(optimization, FlexibleLiveOptimization)
    assert optimization.default_scenario is optimization.scenarios[1]
    assert_frame_equal(
        optimization.scenarios[1].to_dataframe(),
        pd.DataFrame([[4.0, 8.0]], columns=["Factor 1", "Target 1"], index=[0]),
    )

    combined = optimization.to_dataframe()
    assert list(combined["scenario"]) == [0, 1]
    assert list(combined["Factor 1"]) == [3.0, 4.0]


def test_incomplete_prediction_raises_on_to_dataframe(live_analysis):
    """Converting a prediction that has not finished raises rather than returning junk."""
    respond_with(
        live_analysis,
        [envelope("prediction", complete=False, status=None, state="P", targets={})],
    )

    prediction = live_analysis.get_live_predictions()[0]

    assert prediction.complete is False
    with pytest.raises(FeroError, match="is not complete"):
        prediction.to_dataframe()


def test_failed_prediction_raises_with_its_message(live_analysis):
    """Converting a failed prediction surfaces the failure message."""
    respond_with(
        live_analysis,
        [
            envelope(
                "prediction", status="FAILURE", message="model unavailable", targets={}
            )
        ],
    )

    prediction = live_analysis.get_live_predictions()[0]

    with pytest.raises(FeroError, match="model unavailable"):
        prediction.to_dataframe()


def test_unknown_type_from_server_is_reported_clearly(live_analysis):
    """A type this client version does not know about names the upgrade path."""
    respond_with(live_analysis, [envelope("some_future_type")])

    with pytest.raises(FeroError, match="Upgrading the fero package"):
        live_analysis.get_live_predictions()
