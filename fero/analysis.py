import fero
import pandas as pd
from marshmallow import Schema, fields
from typing import Union, List, Optional

from fero import FeroError


class AnalysisSchema(Schema):

    url = fields.String(required=True)

    predictions_url = fields.String(required=True)
    revisions_url = fields.String(required=True)

    created_by = fields.Dict(required=True)

    data_source_name = fields.String(required=True)

    data_source_deleted = fields.Boolean()

    latest_revision = fields.Integer()

    latest_revision_model = fields.UUID(required=True)

    latest_revision_model_state = fields.String(required=True)

    latest_trained_revision = fields.Integer(required=True, allow_none=True)

    latest_trained_revision_model = fields.UUID(allow_none=True)

    latest_completed_model = fields.UUID(allow_none=True)

    latest_completed_model_score = fields.Integer(required=True, allow_none=True)

    latest_completed_model_score_qualifier = fields.String(
        required=True, allow_none=True
    )

    latest_completed_model_modified = fields.DateTime(required=True, allow_none=True)

    schema_overrides = fields.Dict(default=dict, missing=dict)

    display_options = fields.Dict(allow_none=True, default=dict, missing=dict)
    ac_name = fields.String(required=True)

    uuid = fields.UUID(required=True)
    name = fields.String(required=True)
    data_source = fields.UUID(required=True)
    created = fields.DateTime(require=True)
    modified = fields.DateTime(require=True)
    blueprint_name = fields.String(required=True)


class Prediction:
    def __init__(self, client: "fero.Fero", prediction_request_id: str):
        self._client = client
        self._


class Analysis:

    _presentation_data_cache: Union[dict, None] = None
    _reg_factors: Union[List[dict], None] = None
    _reg_targets: Union[List[dict], None] = None
    _factor_names: Union[list, None] = None
    _target_names: Union[list, None] = None

    def __init__(self, client: "fero.Fero", data: dict):
        self._client = client
        schema = AnalysisSchema()
        self._data = schema.load(data)

    def __getattr__(self, name: str):
        return self._data.get(name)

    def __repr__(self):
        return f"<Analysis name={self.name}>"

    __str__ = __repr__

    @property
    def _regression_factors(self):
        if self._reg_factors is None:
            self._parse_regression_data()

        return self._reg_factors

    @property
    def _regression_targets(self):
        if self._reg_targets is None:
            self._parse_regression_data()

        return self._reg_targets

    @property
    def factor_names(self):
        if self._factor_names is None:
            self._parse_regression_data()

        return self._factor_names

    @property
    def target_names(self):
        if self._target_names is None:
            self._parse_regression_data()

        return self._target_names

    @property
    def _presentation_data(self):
        """This is big and ugly, so keep it private but cached"""
        if self._presentation_data_cache is None:
            self._get_presentation_data()

        return self._presentation_data_cache

    @staticmethod
    def _make_col_name(col_name: str, cols: List[str]) -> str:
        """Mangles duplicate columns"""
        c = 0
        og_col_name = col_name
        while col_name in cols:
            col_name = f"{og_col_name}.{c}"
            c += 1

        return col_name

    def _parse_regression_data(self):
        reg_data = next(
            d
            for d in self._presentation_data["data"]
            if d["id"] == "regression_simulator"
        )
        self._reg_factors = reg_data["content"]["factors"]
        self._reg_targets = reg_data["content"]["targets"]
        self._factor_names = [f["factor"] for f in self._reg_factors]
        self._target_names = [t["name"] for t in self._reg_targets]

    def _flatten_result(self, result: dict, prediction_row: dict) -> dict:
        """Flattens nested results returned by the api into a single dict and combines it with the provided data"""
        flat_result = {}
        cols = prediction_row.keys()
        for target, values in result["data"].items():
            low_col = self._make_col_name(f"{target}_low", cols)
            mid_col = self._make_col_name(f"{target}_mid", cols)
            high_col = self._make_col_name(f"{target}_high", cols)
            flat_result.update(
                {
                    low_col: values["value"]["low"][0],
                    mid_col: values["value"]["mid"][0],
                    high_col: values["value"]["high"][0],
                }
            )
        flat_result.update(prediction_row)
        return flat_result

    def _get_presentation_data(self):
        self._presentation_data_cache = self._client.get(
            f"/api/revision_models/{self.latest_trained_revision_model}/presentation_data/"
        )

    def has_trained_model(self) -> bool:
        """Checks whether this analysis has a trained model associated with it.

        :return: True if there is a model, False otherwise
        :rtype: bool
        """
        return self.latest_completed_model is not None

    def make_prediction(
        self, prediction_data: Union[pd.DataFrame, List[dict]]
    ) -> Union[pd.DataFrame, List[dict]]:
        """Makes a prediction from the provided data using the most recent trained model for the analysis.

        `make_prediction` takes either a data frame or list of dictionaries of values that will be sent to Fero
        to make a prediction of what the targets of the Analysis will be.  The results are returned as either a data frame
        or list of dictionaries with both the original prediction data and the predicted targets in each row or dict.
        Each target has a `high`, `low`, and `mid` value and these are added to the target variable name with an `_`.

        :param prediction_data:  Either a data frame or list of dictionaries specifying values to be used in the model.
        :type prediction_data: Union[pd.DataFrame, List[dict]]
        :raises FeroError: Raised if no model has been trained or the server returns an error message
        :return: A data frame or list of dictionaries depending on how the function was called
        :rtype: Union[pd.DataFrame, List[dict]]
        """
        if not self.has_trained_model:
            raise FeroError("No model has been trained on this analysis.")

        is_df = isinstance(prediction_data, pd.DataFrame)

        # convert to dictionary for serialization
        if is_df:
            prediction_data = [dict(row) for _, row in prediction_data.iterrows()]

        prediction_results = []
        # make a prediction for each row
        for row in prediction_data:

            prediction_request = {"values": row}
            prediction_result = self._client.post(
                f"/api/revision_models/{str(self.latest_completed_model)}/predict/",
                prediction_request,
            )
            if prediction_result.get("status") != "SUCCESS":
                raise FeroError(
                    prediction_result.get(
                        "message", "The prediction failed for unknown reasons."
                    )
                )

            prediction_results.append(self._flatten_result(prediction_result, row))

        # convert back to a data frame if need
        return pd.DataFrame(prediction_results) if is_df else prediction_results

    def make_optimization(
        self,
        name: str,
        goal: dict,
        constrains: dict,
        fixed_factors: Optional[dict] = None,
        include_confidence_intervales: bool = False,
        synchonous: bool = True,
    ) -> Prediction:
        """Perform an optimization using the most recent model for the analysis.

        By default this function will block until the optimization is complete, however specifying `synchonous=False`
        will instead return a prediction object referencing the optimization being made.  This prediction will not contain
        results until the `complete` property is true.

        Standard Goal Config
        {
            "goal": "maximize",
            "factor": {"name": "factor1", "min": 5, "max": 10}
        }

        Cost Goal Config
        {
            "type": "COST",
            "goal": "minimize"
            "cost_function": [
                {"min": 5, "max": 10, "cost": 1000, "factor": "factor1},
                {"min": 5, "max": 10, "cost": 500, "factor": "factor1}
            ]
        }

        Constraints Config
        {
            "factor2": {"min": 10, "max": 10}
            "taget1": {"min": 100, "max": 500}
        }

        :param name: Name for this optimizatino
        :type name: str
        :param goal: [description]
        :type goal: dict
        :param constrains: A dictionary
        :type constrains: dict
        :param descrfixed_factors: [description], defaults to None
        :type descrfixed_factors: dict, optional
        :param synchonous: [description], defaults to True
        :type synchonous: bool, optional
        :return: [description]
        :rtype: Prediction
        """
