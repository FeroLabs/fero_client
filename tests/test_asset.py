"""A module to test the `Asset` class."""

import pytest
import json
from fero.asset import Asset
import pandas as pd
from pandas.testing import assert_frame_equal


@pytest.fixture
def test_asset(asset_data, patched_fero_client, prediction_result_success_data):
    """Get a sample `Asset` object."""
    patched_fero_client.post.return_value = prediction_result_success_data
    return Asset(patched_fero_client, asset_data)


@pytest.fixture
def default_predictions():
    """Get sample dataframe of asset predictions."""
    return pd.DataFrame(
        **{
            "data": [
                [
                    7.93735099382404,
                    7.253059448467135,
                    7.688163173887008,
                    8.201992282568096,
                    8.562654808412713,
                    7.290020590899204,
                    7.137018988443888,
                    7.222864154749371,
                    7.3457978216120265,
                    7.474805839163134,
                    5.0696783341277625,
                    3.6869059267194997,
                    4.470695153741833,
                    5.683819078338525,
                    6.529167263822492,
                    1.7271781080698847,
                    1.2293876958841274,
                    1.5332391289059648,
                    1.9211059805503128,
                    2.1972651616834313,
                ],
                [
                    8.059697064287855,
                    6.962005033013789,
                    7.721065781687776,
                    8.453028511724858,
                    8.970620656521195,
                    7.385830908631496,
                    7.2164206074582635,
                    7.312782100772867,
                    7.4474424849045455,
                    7.603513646484785,
                    4.969889551638316,
                    3.2410276157675737,
                    4.180035194039614,
                    5.742825222050875,
                    6.794145544189322,
                    1.6465033404595186,
                    1.0098659780967365,
                    1.397920672348523,
                    1.924213021790375,
                    2.2023476424790025,
                ],
                [
                    8.192820567540917,
                    6.753720344190171,
                    7.691716734161963,
                    8.67238696189184,
                    9.56488526360722,
                    7.44409425873223,
                    7.2532241117672855,
                    7.3604140870516,
                    7.502760719158076,
                    7.7434025244433995,
                    4.854845925095373,
                    2.641086035815071,
                    3.913470878543495,
                    5.7220130262037365,
                    7.075994865914881,
                    1.5516005049741246,
                    0.7784695291543537,
                    1.2338449666203226,
                    1.8708144728173635,
                    2.318108568235832,
                ],
                [
                    8.349753362887295,
                    6.552024567348383,
                    7.619240288083308,
                    8.990032839229166,
                    10.303746588561847,
                    7.460260370638008,
                    7.250522528886993,
                    7.369555018121625,
                    7.511282423365672,
                    7.813681019224355,
                    4.6152964634338804,
                    1.934912207059016,
                    3.491997640608437,
                    5.690586705047334,
                    7.333590667213361,
                    1.416888407225492,
                    0.45107889112482186,
                    1.0140468362241668,
                    1.8300442319324934,
                    2.375700217914943,
                ],
                [
                    8.492439329004604,
                    6.199366100810615,
                    7.498189870344907,
                    9.30211022900819,
                    11.131550725647848,
                    7.482777901396371,
                    7.244147925303968,
                    7.3802383232561874,
                    7.530010225868999,
                    7.9247913725868155,
                    4.379943452590086,
                    1.282381711049496,
                    3.115976079652962,
                    5.591919257727733,
                    7.400851518682516,
                    1.2809534054200706,
                    0.1468654105352288,
                    0.802657230404595,
                    1.76232849577855,
                    2.424137436651667,
                ],
            ],
            "index": [
                "2019-04-29T20:00:00Z",
                "2019-04-29T21:00:00Z",
                "2019-04-29T22:00:00Z",
                "2019-04-29T23:00:00Z",
                "2019-04-30T00:00:00Z",
            ],
            "columns": [
                "mean:Factor 4",
                "p5:Factor 4",
                "p25:Factor 4",
                "p75:Factor 4",
                "p95:Factor 4",
                "mean:Factor 5",
                "p5:Factor 5",
                "p25:Factor 5",
                "p75:Factor 5",
                "p95:Factor 5",
                "mean:Factor 9",
                "p5:Factor 9",
                "p25:Factor 9",
                "p75:Factor 9",
                "p95:Factor 9",
                "mean:Asset 1",
                "p5:Asset 1",
                "p25:Asset 1",
                "p75:Asset 1",
                "p95:Asset 1",
            ],
        }
    )


@pytest.fixture
def test_asset_with_data(test_asset, default_predictions):
    """Get a sample `Asset` with loaded Scenario Simulator predictions."""
    test_asset._presentation_data_cache = [
        {
            "id": "sensor_forecaster",
            "tab": "Scenario Simulator",
            "type": "sensor_forecaster",
            "title": "Scenario Simulator",
            "content": {
                "factors": [
                    {"name": "Factor 4"},
                    {"name": "Factor 5"},
                    {"name": "Factor 9"},
                ],
                "sensors": [{"name": "Factor 3"}, {"name": "Factor 6"}],
                "targets": [{"name": "Asset 1"}],
                "default_predictions": default_predictions.to_dict(orient="split"),
            },
        }
    ]

    return test_asset


@pytest.fixture
def prediction_result_success_data():
    """Get sample data for a successful asset prediction."""
    return {
        "status": "SUCCESS",
        "data": {
            "index": [
                "2019-04-29T20:00:00Z",
                "2019-04-29T21:00:00Z",
                "2019-04-29T22:00:00Z",
                "2019-04-29T23:00:00Z",
                "2019-04-30T00:00:00Z",
            ],
            "columns": [
                "mean:Asset 1",
                "p5:Asset 1",
                "p25:Asset 1",
                "p75:Asset 1",
                "p95:Asset 1",
            ],
            "data": [
                [3.0, 4.0, 5.0, 6.0, 7.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 3.0, 4.0, 5.0, 6.0],
                [4.0, 5.0, 6.0, 7.0, 8.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
            ],
        },
    }


def test_creates_asset_correctly(asset_data, patched_fero_client):
    """Test that a valid asset is created from valid data and a valid client."""
    asset = Asset(patched_fero_client, asset_data)

    assert isinstance(asset, Asset)
    assert str(asset) == "<Asset name=TEST ASSET>"


def test_has_trained_model_true(asset_data, patched_fero_client):
    """Test that `has_trained_model` is true if there is a latest configuration model."""
    asset = Asset(patched_fero_client, asset_data)
    assert asset.has_trained_model() is True


def test_has_trained_model_false(asset_data, patched_fero_client):
    """Test that `has_trained_model` is false if there is no configuration model."""
    asset_data["latest_completed_model"] = None
    asset = Asset(patched_fero_client, asset_data)
    assert asset.has_trained_model() is False


def test_predict_default(test_asset_with_data, default_predictions):
    """Test that default the prediction gives the expected result."""
    result = test_asset_with_data.predict()
    assert_frame_equal(result, default_predictions)


def test_predict_specified_factor_df(
    test_asset_with_data, default_predictions, prediction_result_success_data
):
    """Test a prediction with a manually specified `DataFrame` of factors."""
    specified = pd.DataFrame(
        {"Factor 4": default_predictions["mean:Factor 4"].to_list()}
    )
    result = test_asset_with_data.predict(specified)
    success_data = pd.DataFrame(**prediction_result_success_data["data"])
    expected_frame = default_predictions.drop(
        [c for c in default_predictions.columns if "Factor 4" in c], axis=1
    )
    expected_frame["specified:Factor 4"] = default_predictions["mean:Factor 4"]
    target_columns = [
        "mean:Asset 1",
        "p5:Asset 1",
        "p25:Asset 1",
        "p75:Asset 1",
        "p95:Asset 1",
    ]
    for c in target_columns:
        expected_frame[c] = success_data[c]
    assert_frame_equal(expected_frame, result)


def test_predict_specified_factor_dict(
    test_asset_with_data, default_predictions, prediction_result_success_data
):
    """Test a prediction with a manually specified `dict` of factors."""
    specified = {"Factor 4": default_predictions["mean:Factor 4"].to_list()}
    result = test_asset_with_data.predict(specified)
    success_df = pd.DataFrame(**prediction_result_success_data["data"])
    expected_dict = default_predictions.drop(
        [c for c in default_predictions.columns if "Factor 4" in c], axis=1
    ).to_dict("list")
    expected_dict["index"] = success_df.index.to_list()
    expected_dict["specified:Factor 4"] = default_predictions["mean:Factor 4"].to_list()
    target_columns = [
        "mean:Asset 1",
        "p5:Asset 1",
        "p25:Asset 1",
        "p75:Asset 1",
        "p95:Asset 1",
    ]
    for c in target_columns:
        expected_dict[c] = success_df[c].to_list()
    assert json.dumps(expected_dict, sort_keys=True) == json.dumps(
        result, sort_keys=True
    )
