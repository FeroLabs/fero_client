"""This module holds classes related to a fero analysis."""

import time
import fero
import uuid
import io
import datetime
from enum import Enum
from fero import FeroError
import pandas as pd
from marshmallow import (
    Schema,
    fields,
    validate,
    validates_schema,
    ValidationError,
    EXCLUDE,
)
from typing import Any, Union, List, Optional, IO, Tuple
from .common import FeroObject


VALID_GOALS = ["minimize", "maximize"]
FERO_COST_FUNCTION = "FERO_COST_FUNCTION"
V1_RESULT_SUFFIXES = ["low", "mid", "high"]
V2_RESULT_SUFFIXES = ["low90", "low50", "mid", "high50", "high90"]
BULK_PREDICTION_TYPE = "M"
VALID_JOINS = ["AND", "OR"]


class AnalysisSchema(Schema):
    """A schema to store data related to a fero analysis."""

    class Meta:
        """
        Specify that unknown fields included on this schema should be excluded.

        See https://marshmallow.readthedocs.io/en/stable/quickstart.html#handling-unknown-fields.
        """

        unknown = EXCLUDE

    url = fields.String(required=True)

    predictions_url = fields.String(required=True)
    revisions_url = fields.String(required=True)

    created_by = fields.Dict(required=True)

    data_source_name = fields.String(required=False)
    process_name = fields.String(required=False)

    data_source_deleted = fields.Boolean()
    process_deleted = fields.Boolean()

    latest_revision = fields.Integer()

    latest_revision_model = fields.UUID(required=True)

    latest_revision_model_state = fields.String(required=True)

    latest_trained_revision = fields.Integer(required=False, allow_none=True)

    latest_trained_revision_model = fields.UUID(allow_none=True)

    latest_completed_model = fields.UUID(allow_none=True)
    # We should eventually make this required again when we update the server with defaults
    latest_completed_model_score = fields.Integer(required=False, allow_none=True)

    latest_completed_model_score_qualifier = fields.String(
        required=False, allow_none=True
    )

    latest_completed_model_modified = fields.DateTime(required=False, allow_none=True)

    latest_completed_model_schema = fields.Dict(default=dict, missing=dict)

    schema_overrides = fields.Dict(default=dict, missing=dict)

    display_options = fields.Dict(allow_none=True, default=dict, missing=dict)
    ac_name = fields.String(required=True)

    uuid = fields.UUID(required=True)
    name = fields.String(required=True)
    data_source = fields.UUID(required=True, allow_none=True)
    process = fields.UUID(required=False, allow_none=True)
    created = fields.DateTime(require=True)
    modified = fields.DateTime(require=True)
    blueprint_name = fields.String(required=True)
    hidden = fields.Boolean(required=False)


class RevisionSchema(Schema):
    """A schema to store data related to a fero analysis revision."""

    class Meta:
        """
        Specify that unknown fields included on this schema should be excluded.

        See https://marshmallow.readthedocs.io/en/stable/quickstart.html#handling-unknown-fields.
        """

        unknown = EXCLUDE

    version = fields.Integer()
    configured_blueprint = fields.Dict()


class FactorSchema(Schema):
    """A schema to store data related to a factor linked to a fero analysis optimization."""

    name = fields.String(required=True)
    min = fields.Number(required=False, default=None)
    max = fields.Number(required=False, default=None)
    cost = fields.Number(required=False)

    @validates_schema
    def relative_min_and_max(self, data: dict, **kwargs):
        """Validate that either both min and max or none, or they are both defined, and min < max.

        :raises ValidationError: if min >= max or only one is defined.
        """
        min = data.get("min", None)
        max = data.get("max", None)
        if min is None and max is None:
            return
        elif max is None:
            raise ValidationError(
                {"max": ["A value for 'max' must be provided when 'min' is present."]}
            )
        elif min is None:
            raise ValidationError(
                {"min": ["A value for 'min' must be provided when 'max' is present."]}
            )
        elif min >= max:
            raise ValidationError(
                {"min": ["The value of 'min' must be less than the value of 'max'."]}
            )


class CostSchema(FactorSchema):
    """A schema to store data related to a cost linked to a fero analysis optimization."""

    cost = fields.Float(required=True)

    @validates_schema
    def numeric_factors(self, data: dict, **kwargs):
        """Validate that the cost function factors, being numeric, include min and max parameters.

        :raises ValidationError: if no min or max parameter is defined
        """
        min = data.get("min", None)
        max = data.get("max", None)
        if min is None or max is None:
            raise ValidationError(
                {"factor": ["Cost factors must include both a min and max."]}
            )


class BaseGoalSchema(Schema):
    """A base schema to store data related to the goal of a fero analysis optimization."""

    goal = fields.String(validate=validate.OneOf(VALID_GOALS))


class StandardOptimizeGoal(BaseGoalSchema):
    """A schema to store data related to a goal to min/max some tag in a fero analysis optimization."""

    factor = fields.Nested(FactorSchema(), required=True)

    @validates_schema
    def numeric_goal(self, data: dict, **kwargs):
        """Validate that the goal, being numeric, includes a min and max parameter.

        :raises ValidationError: if no min or max parameter is defined
        """
        if (
            data["factor"].get("min", None) is None
            or data["factor"].get("max", None) is None
        ):
            raise ValidationError(
                {"factor": ["Optimization goal factor must include a min and max."]}
            )


class CostOptimizeGoal(BaseGoalSchema):
    """A schema to store data related to a goal to minimize cost in a fero analysis optimization."""

    type = fields.String(validate=validate.OneOf(["COST"]), required=True)
    cost_function = fields.Nested(CostSchema, many=True, required=True)


class CombinationConstraintOperandType(Enum):
    """An enum listing all valid types of Combination Constraint Operands."""

    FORMULA = "formula"
    CONSTANT = "constant"
    COLUMN = "column"


class CombinationConstraintOperator(Enum):
    """An enum listing all valid types of Combination Constraint Operators."""

    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="


class CombinationConstraint:
    """A class to facilitate structuring additional optimization constraints. See README for details."""

    def __init__(
        self,
        operand_a: Tuple[Union[str, int, float], CombinationConstraintOperandType],
        operator: CombinationConstraintOperator,
        operand_b: Tuple[Union[str, int, float], CombinationConstraintOperandType],
    ):
        """
        Create a `Combination Constraint` with two provided operands and an operator.

        :param operand_a: A tuple with the first operand's value and type. (Left side of operator).
        :param operator: Enum value indicating which a supported operation for the operands.
        :param operand_b: A tuple with the second operand's value and type. (Right side of operator).
        """
        self._operand_a = operand_a
        self._operand_b = operand_b
        self._operator = operator

    def combination_constraint_to_dict(self):
        """Return valid dictionary representation of this constraint."""
        a_value, a_kind = self._operand_a
        b_value, b_kind = self._operand_b
        return {
            "kind": "condition",
            "target": {"kind": a_kind.value, "value": a_value},
            "operation": {
                "operator": self._operator.value,
                "operand": {"kind": b_kind.value, "value": b_value},
            },
        }

    def verify_combination_constraint(self, analysis):
        """Check that literal column references are valid for this analysis.

        Formula verification occurs on Fero's servers. Any errors will be provided in optimization results.
        """
        for [value, kind] in [self._operand_a, self._operand_b]:
            if kind == CombinationConstraintOperandType.COLUMN:
                if value not in analysis.factor_names:
                    raise FeroError(
                        f'"{value}" is not a recognized factor in this analysis'
                    )


class Prediction:
    """Represents the results of a prediction submitted to Fero.

    Predictions are run asynchrounously by Fero and data will only be available if the `complete` property
    is true.  Attempting to access the data for the prediction before it is complete will result in a `FeroError`
    being raised.
    """

    def __init__(self, client: "fero.Fero", prediction_result_id: str):
        """
        Create a `Prediction` using the ID from a set of prediction results.

        :param client: a fero client to interface with the API
        :param prediction_result_id: the ID tied to the prediction's result
        """
        self._client = client
        self.result_id = prediction_result_id
        self._data_cache = None
        self._complete = False
        self._result_id = None

    def __repr__(self) -> str:
        """Represent the `Prediction` by its prediction result ID."""
        return f"<Prediction request_id={self.result_id}>"

    def __getattr__(self, name: str) -> Any:
        """
        Access an entry in this object's schema.

        :param name: the entry in the schema to access
        :return: the data stored in this entry of the schema
        """
        return self._data.get(name)

    __str__ = __repr__

    @property
    def _data(self):
        """Get the data related to this prediction result if it is available."""
        # check if it's complete which also pulls data
        if not self.complete:
            return None

        return self._data_cache

    @property
    def status(self):
        """Check the status of the prediction if it is available."""
        if not self.complete:
            return None
        if "result_data" in self._data and "status" in self._data["result_data"]:
            return self._data["result_data"]["status"]
        return None

    @property
    def complete(self):
        """Check if the Prediction is complete."""
        if not self._complete:
            self._data_cache = self._client.get(
                f"/api/prediction_results/{self.result_id}/"
            )
            self._complete = self._data_cache["state"] == "C"

        return self._complete

    def get_results(self, format="dataframe") -> Union[pd.DataFrame, List[dict]]:
        """Return the prediction results of the prediction.

        By default this will be a pandas DataFrame, but specifying `format="record"`
        will instead return a list of dictionaries where each key specifies
        a factor and the value is the prediction.

        :param format: The format to return the result as, defaults to "dataframe"
        :type format: str, optional
        :raises FeroError: Raised if the prediction is not yet complete or is completed and failed.
        :return: The results of the prediction
        :rtype: Union[pd.DataFrame, List[dict]]
        """
        if not self.complete:
            raise FeroError("Prediction is not complete.")
        if self.status != "SUCCESS":
            raise FeroError(
                f"Prediction failed with the following message: {self._data['result_data']['message']}"
            )
        if self.prediction_type == BULK_PREDICTION_TYPE:
            data_url = self._data["result_data"]["data"]["download_url"]
            data = self._client.get_preauthenticated(data_url)
            return pd.DataFrame(**data)
        else:
            data = self._data["result_data"]["data"]["values"]
            if format == "records":
                return [
                    {col: val for col, val in zip(data["columns"], row)}
                    for row in data["data"]
                ]
            else:
                return pd.DataFrame(
                    data["data"], columns=data["columns"], index=data["index"]
                )


class Analysis(FeroObject):
    """An object for interacting with a specific Analysis on Fero.

    The Analysis is the primary way to access a model associated with a data set.
    Once an analysis is created it can be used to perform various actions such as making a prediction
    based on provided data or optimizing certain values based on the Fero model of the data.
    """

    _presentation_data_cache: Optional[dict] = None
    _reg_factors: Optional[List[dict]] = None
    _reg_targets: Optional[List[dict]] = None
    _factor_names: Optional[List[str]] = None
    _target_names: Optional[List[str]] = None

    schema_class = AnalysisSchema

    def __repr__(self) -> str:
        """Represent the `Analysis` by its name."""
        return f"<Analysis name={self.name}>"

    __str__ = __repr__

    @property
    def _regression_factors(self) -> Optional[List[dict]]:
        """Get data on the factors of the Analysis."""
        if self._reg_factors is None:
            self._parse_regression_data()

        return self._reg_factors

    @property
    def _regression_targets(self) -> Optional[List[dict]]:
        """Get data on the targets of the Analysis."""
        if self._reg_targets is None:
            self._parse_regression_data()

        return self._reg_targets

    @property
    def factor_names(self) -> Optional[List[str]]:
        """Get the names of the factors in the model of the Analysis."""
        if self._factor_names is None:
            self._parse_regression_data()

        return self._factor_names

    @property
    def target_names(self) -> Optional[List[str]]:
        """Get the names of the targets of the Analysis."""
        if self._target_names is None:
            self._parse_regression_data()

        return self._target_names

    @property
    def _presentation_data(self) -> Optional[dict]:
        """Get the presentation data of this analysis."""
        if self._presentation_data_cache is None:
            self._get_presentation_data()

        return self._presentation_data_cache

    @property
    def _schema(self):
        """Get the schema for the process related to this Analysis."""
        if self._schema_cache is None:
            v1_schema = self._data.get("latest_completed_model_schema", None)
            if (v1_schema is None or len(v1_schema) == 0) and self.process is not None:
                v1_schema = self._client.get(
                    f"/api/processes/{self.process}/v1_interpreted_schema/"
                )
            elif (
                v1_schema is None or len(v1_schema) == 0
            ) and self.data_source is not None:
                v1_schema = self._client.get(
                    f"/api/data_sources/{self.data_source}/latest_interpreted_schema/"
                )
            self._schema_cache = v1_schema

        return self._schema_cache

    @property
    def all_tag_names(self):
        """Get names for any tag available to the Analysis."""
        return [column["name"] for column in self._schema["columns"]]

    @staticmethod
    def _make_col_name(col_name: str, cols: List[str]) -> str:
        """Mangle duplicate columns."""
        c = 0
        og_col_name = col_name
        while col_name in cols:
            col_name = f"{og_col_name}.{c}"
            c += 1

        return col_name

    def is_retraining(self) -> bool:
        """Check if an analysis is currently being retrained.

        :return: True if the analysis is in the process of being retrained, false otherwise.
        """
        self._refresh_analysis()
        training = self.latest_revision_model_state == "T"
        latest_revision = self.latest_revision
        if not training:
            self._refresh_analysis()
            # requery if a new version somehow got created when we refreshed.  Do this until they match
            if self.latest_revision != latest_revision:
                return self.is_retraining()

        return training

    def revise(self):
        """Trigger an analysis revision which causes the analysis to be retrained with any new data or changes in the process configurations.

        This method will not revise the analysis if a new analysis is currently being trained.  It currently only supports revising an analysis with the previously selected options.

        For more granular configuration of a revision please use the Fero Labs website.
        """
        # don't revise if currently training
        if self.is_retraining():
            return

        latest_revision = self._client.get(
            f"/api/analyses/{self.uuid}/revisions/{self.latest_revision}/"
        )
        latest_revision = RevisionSchema().load(latest_revision)
        latest_blueprint = latest_revision["configured_blueprint"]

        self._client.post(
            f"/api/analyses/{self.uuid}/revisions/",
            {
                "description": f"Revision created via fero client on {datetime.datetime.now().isoformat()}",
                "configured_blueprint": latest_blueprint,
            },
        )
        self._refresh_analysis()

    def _refresh_analysis(self):
        """Reload the latest analysis data from the server."""
        data = self._client.get(f"/api/analyses/{self.uuid}/")
        schema = AnalysisSchema()
        self._data = schema.load(data)

    def _upload_file(self, fp: IO, file_tag: str, prediction_type: str) -> str:
        """Upload a single file to the uploaded files location and returns ID of workspace to check."""
        inbox_response = self._client.post(
            f"/api/analyses/{self.uuid}/workspaces/inbox_url/",
            {"file_tag": file_tag, "prediction_type": prediction_type},
        )

        workspace_id = inbox_response.pop("workspace_id")
        self._client.upload_file(inbox_response, file_tag, fp)
        return workspace_id

    def _parse_regression_data(self):
        """Get and parse the regression simulator object from presentation data."""
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
        """Flatten nested results returned by the api into a single dict and combine it with the provided data."""
        flat_result = {}
        cols = prediction_row.keys()

        for target, values in result["data"].items():
            suffix_list = (
                V2_RESULT_SUFFIXES if result["version"] == 2 else V1_RESULT_SUFFIXES
            )
            flat_result.update(
                {
                    self._make_col_name(f"{target}_{suffix}", cols): values["value"][
                        suffix
                    ][0]
                    for suffix in suffix_list
                }
            )
        flat_result.update(prediction_row)
        return flat_result

    def _get_presentation_data(self):
        """Retrieve the presentation data of this analysis revision from the api."""
        self._presentation_data_cache = self._client.get(
            f"/api/revision_models/{self.latest_trained_revision_model}/presentation_data/"
        )

    def has_trained_model(self) -> bool:
        """Check whether this analysis has a trained model associated with it.

        :return: True if there is a model, False otherwise
        """
        return self.latest_completed_model is not None

    def make_prediction(
        self, prediction_data: Union[pd.DataFrame, List[dict]]
    ) -> Union[pd.DataFrame, List[dict]]:
        """Make a prediction from the provided data using the most recent trained model for the analysis.

        This method is optimized for analyses that support fast, bulk prediction.

        `make_prediction` takes either a data frame or list of dictionaries of values that will be sent to Fero
        to make a prediction of what the targets of the Analysis will be. The results are returned as either a dataframe
        or list of dictionaries with both the original prediction data and the predicted targets in each row or dict.
        Prediction data should not contain any missing values. Each target has a `high`, `low`, and `mid` value and
        these are added to the target variable name with an `_`.

        :param prediction_data:  Either a data frame or list of dictionaries specifying values to be used in the model.
        :type prediction_data: Union[pd.DataFrame, List[dict]]
        :raises FeroError: Raised if no model has been trained or the server returns an error message
        :return: A data frame or list of dictionaries depending on how the function was called
        :rtype: Union[pd.DataFrame, List[dict]]
        """
        if not self.has_trained_model:
            raise FeroError("No model has been trained on this analysis.")

        is_dict_list = isinstance(prediction_data, list)

        prediction_df = (
            pd.DataFrame(prediction_data) if is_dict_list else prediction_data
        )

        data_file = io.StringIO()
        prediction_df.to_json(data_file, orient="split")
        data_file.seek(0)
        upload_identifier = str(uuid.uuid4())
        workspace_id = self._upload_file(
            data_file, upload_identifier, BULK_PREDICTION_TYPE
        )
        prediction = self._poll_workspace_for_prediction(workspace_id)
        if prediction.status != "SUCCESS":
            raise FeroError(
                prediction.result_data.get(
                    "message", "The prediction failed for unknown reasons"
                )
            )
        output = prediction.get_results()
        return list(output.T.to_dict().values()) if is_dict_list else output

    def _get_factor_dtype(self, factor_name: str) -> Union[str, None]:
        """Return the dtype of a factor or `None` if the factor isn't found."""
        try:
            goal_data = next(
                f for f in self._regression_factors if f["factor"] == factor_name
            )
            return f"factor_{goal_data['dtype']}"
        except StopIteration:
            return None

    def _get_target_dtype(self, target_name: str) -> Union[str, None]:
        """Return the dtype of a target or `None` if the target isn't found."""
        try:
            # Only support real and integer targets
            guess = next(
                f["guess"]
                for f in self._schema["columns"]
                if f["name"] == target_name and f["guess"] in ["real", "integer"]
            )
            return "target_float" if guess == "real" else "target_int"
        except StopIteration:
            return None

    def _verify_standard_goal(self, goal: dict):
        """Verify the goal config relative to the analysis."""
        goal_name = goal["factor"]["name"]

        # The goal label must be a target or factor
        if goal_name not in self.target_names + self.factor_names:
            raise FeroError(f'"{goal_name}" is not a target or factor for this model.')

        # If this is a factor makes sure it's not a float
        if goal_name not in self.target_names and self._get_factor_dtype(
            goal_name
        ) not in ["factor_float", "factor_int"]:
            raise FeroError("The data type of the goal must be float or integer.")

    def _verify_cost_goal(self, goal: dict):
        """Verify that a cost goal is correct."""
        for c in goal["cost_function"]:
            name = c["name"]
            factor_type = self._get_factor_dtype(name)
            # Implicitly find missing factors
            if factor_type is not None and factor_type not in [
                "factor_float",
                "factor_int",
            ]:
                raise FeroError(
                    "The data type of all cost function factors must be float or integer."
                )
            else:
                target_type = self._get_target_dtype(name)
                if target_type is None:
                    raise FeroError(
                        f'"{name}" is not a factor or target in this model.'
                    )
                # Implicitly find missing factors
                if target_type not in [
                    "target_float",
                    "target_int",
                ]:
                    raise FeroError(
                        "The data type of all cost function targets must be float or integer."
                    )

    def _verify_constraints(self, constraints: List[dict], **kwargs):
        """Verify provided constraints are in the analysis."""
        use_adaptive = kwargs.get("use_adaptive", False)
        for constraint in constraints:
            constraint_name = constraint["name"]
            constraint_type = (
                self._get_factor_dtype(constraint_name)
                if constraint_name in self.factor_names
                else (
                    self._get_target_dtype(constraint_name)
                    if constraint_name in self.target_names
                    else None
                )
            )
            if constraint_type is None:
                raise FeroError(
                    f'Constraint "{constraint_name}" is not part of this model.'
                )
            elif constraint_type == "factor_category" and (
                constraint.get("min", None) is not None
                or constraint.get("max", None) is not None
            ):
                raise FeroError(
                    f'Categorical factor "{constraint_name}" should not define a min or max value.'
                )
            elif use_adaptive and constraint_type == "factor_category":
                raise FeroError(
                    "Categorical factors are not yet supported for adaptive optimizations."
                )
            elif constraint_type in [
                "factor_int",
                "factor_float",
                "target_int",
                "target_float",
            ] and (
                constraint.get("min", None) is None
                or constraint.get("max", None) is None
            ):
                raise FeroError(
                    f'Numeric constraint "{constraint_name}" requires a min and max value.'
                )

    def _cross_verify_optimization(
        self,
        goal: dict,
        constraints: List[dict],
        **kwargs,
    ):
        """Verify the config has the correct combined targets, costs and factors."""
        is_cost = goal.get("type", None) == "COST"

        cost_factors = []
        goal_factor = []
        target_factor = []
        constraint_factors = []
        constraint_targets = []
        use_adaptive = kwargs.get("use_adaptive", False)

        if len(constraints) < 1:
            raise FeroError("At least one constraint must be specified.")
        if is_cost:
            for factor in goal["cost_function"]:
                if factor["name"] in self.factor_names:
                    cost_factors.append(factor["name"])
                else:
                    target_factor.append(factor["name"])
        else:
            if goal["factor"]["name"] in self.factor_names:
                goal_factor.append(goal["goal"])
            else:
                target_factor.append(goal["goal"])

        for constraint in constraints:
            if constraint["name"] in self.factor_names:
                constraint_factors.append(constraint["name"])
            else:
                constraint_targets.append(constraint["name"])

        if len(target_factor + constraint_targets) < 1:
            raise FeroError("No Targets specified")

        if (
            len(constraint_factors + cost_factors + goal_factor) > 3
            and not use_adaptive
        ):
            raise FeroError(
                "A maximum of three factors can be specified in an optimization."
            )
        elif use_adaptive and len(constraint_factors + cost_factors + goal_factor) > 5:
            raise FeroError(
                "A maximum of five factors can be specified in an adaptive optimization."
            )

    def _verify_combination_constraints(
        self, combination_constraints: Optional[List[CombinationConstraint]]
    ):
        if combination_constraints is None or len(combination_constraints) < 1:
            return

        for constraint in combination_constraints:
            constraint.verify_combination_constraint(self)

    def _verify_fixed_factors(self, fixed_factors: dict):
        """Check that the provided fixed factors are in the analysis."""
        all_columns = self.factor_names
        for key in fixed_factors.keys():
            if key not in all_columns:
                raise FeroError(f'"{key}" is not part of this analysis.')

    def _get_basis_values(self):
        """Get median fixed values from presentation data."""
        reg_data = next(
            d
            for d in self._presentation_data["data"]
            if d["id"] == "regression_simulator"
        )
        # yuck massive ugly json
        base_values = {f["factor"]: f["median"] for f in reg_data["content"]["factors"]}

        target_values = {
            t["name"]: t["default"]["mid"][0] for t in reg_data["content"]["targets"]
        }

        base_values.update(target_values)
        return base_values

    def _build_optimize_request(
        self,
        name,
        goal,
        constraints,
        fixed_factors,
        include_confidence_intervals,
        combination_constraints,
        **kwargs,
    ) -> dict:
        """Format the content for an optimization request."""
        is_cost = goal.get("type", None) == "COST"
        ci_value = "include" if include_confidence_intervals else "exclude"
        opt_request = {
            "name": name,
            "description": "",
            "prediction_type": "O",
        }

        input_data = {
            "kind": "OptimizationRequest",
            "version": 1,
            "objectives": [
                {
                    "factor": FERO_COST_FUNCTION if is_cost else goal["factor"]["name"],
                    "goal": goal["goal"],
                }
            ],
            "basisSpecifiedColumns": [],
            "linearFunctionDefinitions": {},
            "useAdaptiveGrid": kwargs.get("use_adaptive", False),
        }
        if combination_constraints is not None and len(combination_constraints) > 0:
            input_data["combinationConstraints"] = {
                "conditions": [
                    constraint.combination_constraint_to_dict()
                    for constraint in combination_constraints
                ],
                "join": "AND",
                "kind": "clause",
            }
        basis_values = self._get_basis_values()
        basis_values.update(fixed_factors)
        input_data["basisValues"] = basis_values

        factor_bounds = []
        for c in constraints:
            if c["name"] in self.factor_names:
                dtype = self._get_factor_dtype(c["name"])
                factor_bounds.append(
                    {
                        "factor": c["name"],
                        "lowerBound": c.get("min", None),
                        "upperBound": c.get("max", None),
                        "dtype": dtype,
                    }
                    if dtype != "factor_category"
                    else {"factor": c["name"], "dtype": dtype}
                )
        target_bounds = [
            {
                "factor": c["name"],
                "lowerBound": c.get("min", None),
                "upperBound": c.get("max", None),
                "dtype": self._get_target_dtype(c["name"]),
                "confidenceInterval": ci_value,
            }
            for c in constraints
            if c["name"] in self.target_names
        ]

        bounds = factor_bounds + target_bounds

        if is_cost:
            cost_lower_bound = 0
            cost_upper_bound = 0

            for factor in goal["cost_function"]:
                cost_lower_bound += factor["min"] * factor["cost"]
                cost_upper_bound += factor["max"] * factor["cost"]

            for c in goal["cost_function"]:
                factor_is_target = c["name"] in self.target_names
                bound = {
                    "factor": c["name"],
                    "lowerBound": c["min"],
                    "upperBound": c["max"],
                    "dtype": f"{'target' if factor_is_target else 'factor'}_float",
                }
                if factor_is_target:
                    bound["confidenceInterval"] = ci_value
                bounds.append(bound)

            bounds.append(
                {
                    "factor": FERO_COST_FUNCTION,
                    "lowerBound": cost_lower_bound,
                    "upperBound": cost_upper_bound,
                    "dtype": "function",
                }
            )

            input_data["linearFunctionDefinitions"] = {
                FERO_COST_FUNCTION: {
                    c["name"]: c["cost"] for c in goal["cost_function"]
                }
            }

        else:
            goal_name = goal["factor"]["name"]
            goal_is_target = goal_name in self.target_names
            dtype = (
                self._get_target_dtype(goal_name)
                if goal_is_target
                else self._get_factor_dtype(goal_name)
            )
            goal_bound = {
                "factor": goal["factor"]["name"],
                "lowerBound": goal["factor"]["min"],
                "upperBound": goal["factor"]["max"],
                "dtype": (
                    dtype
                    if dtype is not None
                    else f"{'target' if goal_is_target else 'factor'}_float"
                ),
            }
            if goal_is_target:
                goal_bound["confidenceInterval"] = ci_value

            bounds.append(goal_bound)

        input_data["bounds"] = bounds
        opt_request["input_data"] = input_data

        return opt_request

    def _request_prediction(
        self, prediction_request: dict, synchronous: bool
    ) -> Prediction:
        """Make the prediction request and poll unitl complete if this request is synchronous."""
        response_data = self._client.post(
            f"/api/analyses/{str(self.uuid)}/workspaces/",
            {"request": prediction_request, "visible": False},
        )
        prediction = Prediction(
            self._client, response_data["latest_prediction"]["latest_results"]
        )

        # If synchronous block until the prediction is complete
        while synchronous and not prediction.complete:
            time.sleep(0.5)

        return prediction

    def _poll_workspace_for_prediction(self, workspace_id: str):
        """Poll workspace until it and a result exist."""
        workspace_url = f"/api/analyses/{str(self.uuid)}/workspaces/{workspace_id}"
        workspace_data = None
        while workspace_data is None:
            time.sleep(0.5)
            workspace_data = self._client.get(
                workspace_url,
                params={"prediction_type": BULK_PREDICTION_TYPE},
                allow_404=True,
            )

        prediction = Prediction(
            self._client, workspace_data["latest_prediction"]["latest_results"]
        )
        while not prediction.complete:
            time.sleep(0.5)

        return prediction

    def make_optimization(
        self,
        name: str,
        goal: dict,
        constraints: List[dict],
        fixed_factors: Optional[dict] = None,
        include_confidence_intervals: bool = True,
        synchronous: bool = True,
        combination_constraints: Optional[List[CombinationConstraint]] = None,
        **kwargs,
    ) -> Prediction:
        """Perform an optimization using the most recent model for the analysis.

        By default this function will block until the optimization is complete, however specifying `synchonous=False`
        will instead return a prediction object referencing the optimization being made. This prediction will not contain
        results until the `complete` property is true.

        :param name: Name for this optimization
        :type name: str
        :param goal: A dictionary describing the goal of the optimization
        :type goal: dict
        :param constraints: A dictionary describing the constraints of the optimization
        :type constraints: dict
        :param fixed_factors: Values of factors to stay fixed if not provided the mean values are used, defaults to None
        :type fixed_factors: dict, optional
        :param include_confidence_intervals: Whether the optimization should include the lower 5% and upper 95% prediction levels, defaults to True
        :type include_confidence_intervals: bool, optional
        :param synchronous: Whether the optimization should return only after being complete.  This can take a bit, defaults to True
        :type synchronous: bool, optional
        :param combination_constraints: A list of additional constraints on the optimization results, typically relating multiple factors or targets.
        :type combination_constraints: list, optional
        :return: The results of the optimization
        :rtype: Prediction
        """
        if self.blueprint_name == "fault":
            raise FeroError("Fault analysis optimization are not supported")

        cost_goal = "type" in goal
        if fixed_factors is None:
            fixed_factors = {}

        goal_schema = CostOptimizeGoal() if cost_goal else StandardOptimizeGoal()
        goal_validation = goal_schema.validate(goal)
        if goal_validation:
            raise FeroError(f"Error validating goal <f{str(goal_validation)}>")

        constraints_schema = FactorSchema(many=True)
        constraints_validation = constraints_schema.validate(constraints)
        if constraints_validation:
            raise FeroError(
                f"Error validating constraints <f{str(constraints_validation)}>"
            )

        if cost_goal:
            self._verify_cost_goal(goal)
        else:
            self._verify_standard_goal(goal)

        self._verify_constraints(constraints, **kwargs)
        self._verify_combination_constraints(combination_constraints, **kwargs)
        self._cross_verify_optimization(goal, constraints, **kwargs)
        self._verify_fixed_factors(fixed_factors)

        optimize_request = self._build_optimize_request(
            name,
            goal,
            constraints,
            fixed_factors,
            include_confidence_intervals,
            combination_constraints,
            **kwargs,
        )
        return self._request_prediction(optimize_request, synchronous)

    @property
    def configured_blueprint(self) -> dict:
        """Return the configured blueprint for this analysis."""
        if self._configured_blueprint is None:
            revision = self._client.get(
                f"/api/analyses/{self.uuid}/revisions/{self.latest_revision}"
            )
            self._configured_blueprint = revision["configured_blueprint"]
        return self._configured_blueprint
