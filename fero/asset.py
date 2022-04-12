"""This module holds classes related to a fero asset."""

import pandas as pd
from fero import FeroError
from marshmallow import Schema, fields, EXCLUDE
from typing import Union, Optional, Mapping
from .common import FeroObject


class AssetSchema(Schema):
    """A schema to store data related to a fero asset."""

    class Meta:
        """
        Specify that unknown fields included on this schema should be excluded.

        See https://marshmallow.readthedocs.io/en/stable/quickstart.html#handling-unknown-fields.
        """

        unknown = EXCLUDE

    url = fields.String(required=True)

    configurations_url = fields.String(required=True)

    process = fields.UUID(required=True)

    process_deleted = fields.Boolean()

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
    created_by = fields.Dict(required=True)
    modified = fields.DateTime(require=True)


class Asset(FeroObject):
    """An object for interacting with a specific Asset on Fero.

    The Asset is the primary way to access a model associated with an asset or time-series data set.
    Once an asset is created, it can be used to perform various actions such as making predictions
    and evaluating horizon times to configured thresholds.
    """

    schema_class = AssetSchema

    def __repr__(self) -> str:
        """Represent the `Asset` by its name."""
        return f"<Asset name={self.name}>"

    __str__ = __repr__

    def _get_presentation_data(self):
        """Retrieve the presentation data of this asset from the api."""
        self._presentation_data_cache = self._client.get(
            f"/api/configuration_models/{self.latest_trained_configuration_model}/presentation_data/"
        )["data"]

    @property
    def _presentation_data(self):
        """Get the presentation data of this asset."""
        if self._presentation_data_cache is None:
            self._get_presentation_data()

        return self._presentation_data_cache

    @property
    def _default_predictions(self):
        default_predictions = None
        for p in self._presentation_data:
            if "id" in p and p["id"] == "sensor_forecaster":
                default_predictions = p["content"]["default_predictions"]

        if default_predictions is None:
            return pd.DataFrame()
        else:
            return pd.DataFrame(**default_predictions)

    def has_trained_model(self) -> bool:
        """Check whether this asset has a trained model associated with it.

        :return: True if there is a model, False otherwise
        """
        return self.latest_completed_model is not None

    def predict(
        self, specified_values: Optional[Union[pd.DataFrame, Mapping[str, list]]] = None
    ) -> Union[pd.DataFrame, Mapping[str, list]]:
        """Make predictions using the most recent trained asset configuration.

        Predictions are made at regular intervals for the specified horizon time following the end of the training set.

        `predict` returns a DataFrame with predictions for each controllable factor and the target
        for each timestamp in the prediction horizon. `predict` optionally accepts a DataFrame or list
        of dictionaries representing values for one or more controllable factors. If provided, Fero will
        substitute the given values for a controllable factor when predicting the target metric, returning
        either a DataFrame or dict, according to the input type.

        :param specified_values:  Either a data frame or mapping to factors to value lists, specifying values
            to use for controllable factors in the predictions.
        :raises FeroError: Raised if no model has been trained or the server returns an error message
        :return: A data frame or list of dictionaries depending on how the function was called
        """
        if not self.has_trained_model:
            raise FeroError("No model has been trained for this asset.")

        if specified_values is None:
            return self._default_predictions

        is_df = isinstance(specified_values, pd.DataFrame)

        # convert to dictionary for serialization
        if is_df:
            specified_values = specified_values.to_dict("list")

        prediction_request = (
            {"values": specified_values} if specified_values is not None else {}
        )
        prediction_result = self._client.post(
            f"/api/configuration_models/{str(self.latest_trained_configuration_model)}/predict/",
            prediction_request,
        )
        if prediction_result.get("status") != "SUCCESS":
            raise FeroError(
                prediction_result.get(
                    "message", "The prediction failed for unknown reasons."
                )
            )
        result = self._default_predictions.copy()

        prediction_results = pd.DataFrame(**prediction_result["data"])
        if any(result.index != prediction_results.index):
            raise FeroError("Predictions include mismatched timestamps.")
        result[prediction_results.columns] = prediction_results
        specified_columns = list(specified_values.keys())
        prefixes = ["mean:", "p5:", "p25:", "p75:", "p95:"]
        drop_cols = [f"{p}{c}" for c in specified_columns for p in prefixes]
        result.drop(drop_cols, axis=1, inplace=True)
        specified_data = pd.DataFrame(
            {f"specified:{k}": v for k, v in specified_values.items()}
        ).set_index(result.index)
        result = result.join(specified_data)
        if not is_df:
            dict_result = result.to_dict("list")
            dict_result["index"] = result.index.to_list()
            return dict_result
        return result
