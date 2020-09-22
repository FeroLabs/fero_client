import fero
import pandas as pd
from marshmallow import Schema, fields
from typing import Union, List

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

    def _results_to_df(self, results: List[dict]) -> pd.DataFrame:
        pass

    @staticmethod
    def _flatten_result(result: dict, prediction_row: dict) -> dict:
        flat_result = {}

        for target, values in result["data"].items():
            flat_result.update(
                {
                    f"{target}_low": values["value"]["low"][0],
                    f"{target}_mid": values["value"]["mid"][0],
                    f"{target}_high": values["value"]["high"][0],
                }
            )
        flat_result.update(prediction_row)
        return flat_result

    def has_trained_model(self) -> bool:
        return self.latest_completed_model is not None

    def make_prediction(
        self, prediction_data: Union[pd.DataFrame, List[dict]]
    ) -> Union[pd.DataFrame, List[dict]]:
        if not self.has_trained_model:
            raise fero.FeroError("No model has been trained on this analysis.")

        is_df = isinstance(prediction_data, pd.DataFrame)

        if is_df:
            prediction_data = [dict(row) for row in prediction_data.iterrows()]

        prediction_results = []
        for row in prediction_data:

            prediction_request = {"values": row}
            prediction_result = self._client.post(
                f"/api/revision_models/{str(self.latest_completed_model)}/predict/",
                prediction_request,
            )

            if prediction_result.get("status") != "SUCCESS":
                raise FeroError(
                    prediction_results.get(
                        "message", "The prediction failed for unknown reasons."
                    )
                )

            prediction_results.append(self._flatten_result(prediction_result, row))

        return pd.DataFrame(prediction_results) if is_df else prediction_results
