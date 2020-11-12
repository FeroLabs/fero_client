import fero
from marshmallow import (
    Schema,
    fields,
    validate,
    validates_schema,
    EXCLUDE
)
from typing import Union, List, Optional


class AssetSchema(Schema):
    class Meta:
        unknown = EXCLUDE

    url = fields.String(required=True)

    configurations_url = fields.String(required=True)

    data_source_name = fields.String(required=True)

    data_source_deleted = fields.Boolean()

    current_bound = fields.Dict(allow_none=True, default=dict, missing=dict)

    latest_configuration = fields.Integer()

    latest_configuration_model = fields.UUID(required=True)

    latest_configuration_model_state = fields.String(required=True)

    latest_trained_configuration = fields.Integer(required=True, allow_none=True)

    latest_trained_configuration_model = fields.UUID(allow_none=True)

    latest_completed_model = fields.UUID(allow_none=True)

    latest_completed_model_score = fields.Integer(required=True, allow_none=True)

    latest_completed_model_score_qualifier = fields.String(
        required=True, allow_none=True
    )

    latest_completed_model_modified = fields.DateTime(required=True, allow_none=True)

    configured_blueprint = fields.Dict()

    stability_warning = fields.Integer()

    stability_warning_display = fields.String()

    time_to_threshold_lower = fields.Time()

    time_to_threshold_mean = fields.Time()

    time_to_threshold_upper = fields.Time()

    prediction_horizon = fields.Time()

    prediction_at_horizon_lower = fields.Float()

    prediction_at_horizon_mean = fields.Float()

    prediction_at_horizon_upper = fields.Float()

    ac_name = fields.String(required=True)

    uuid = fields.UUID(required=True)
    name = fields.String(required=True)
    created = fields.DateTime(require=True)
    modified = fields.DateTime(require=True)


class Asset:
    """An object for interacting with a specific Asset on Fero.

    The Asset is the primary way to access a model associated with an asset or time-series data set.
    Once an asset is created, it can be used to perform various actions such as making predictions
    and evaluating horizon times to configured thresholds.
    """

    def __init__(self, client: "fero.Fero", data: dict):
        self._client = client
        schema = AssetSchema()
        print('>>>> DATA!: {}'.format(data))
        self._data = schema.load(data)

    def __getattr__(self, name: str):
        return self._data.get(name)

    def __repr__(self):
        return f"<Asset name={self.name}>"

    __str__ = __repr__

