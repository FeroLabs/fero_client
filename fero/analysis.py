import fero
import pandas as pd
from marshmallow import Schema, fields
from typing import Union

from fero import FeroError


class Prediction:
    def __init__(self, data: dict):
        self._data = data

    def __dict__(self):
        return self._data

    def to_series(data) -> pd.Series:
        pass


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


class Analysis:
    def __init__(self, client: "fero.Fero", data: dict):
        self._client = client
        schema = AnalysisSchema()
        self._data = schema.load(data)

    def __getattr__(self, name: str):
        return self._data.get(name)

    def __repr__(self):
        return f"<Analysis name={self.name}>"

    __str__ = __repr__

    def has_trained_model(self) -> bool:
        return self.latest_completed_model is not None

    def make_prediction(self, prediction_data: Union[pd.Series, dict]) -> Prediction:
        if not self.has_trained_model:
            raise fero.FeroError("No model has been trained on this analysis.")

        if isinstance(prediction_data, pd.Series):
            prediction_data = dict(prediction_data)

        prediction_request = {"values": prediction_data}
        prediction_results = self._client.post(
            f"/api/revision_models/{str(self.latest_completed_model)}/predict/",
            prediction_request,
        )

        if prediction_results.get("status") != "SUCCESS":
            raise FeroError(
                prediction_results.get(
                    "message", "The prediction failed for unknown reasons."
                )
            )

        return Prediction(prediction_results.get("data", {}))
