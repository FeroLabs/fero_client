import time
import fero
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
from typing import Union, List, Optional


VALID_GOALS = ["minimize", "maximize"]
FERO_COST_FUNCTION = "FERO_COST_FUNCTION"


class AnalysisSchema(Schema):
    class Meta:
        unknown = EXCLUDE

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


class FactorSchema(Schema):
    name = fields.String(required=True)
    min = fields.Float(required=True)
    max = fields.Float(required=True)


class CostSchema(FactorSchema):
    cost = fields.Float(required=True)


class BaseGoalSchema(Schema):

    goal = fields.String(validate=validate.OneOf(VALID_GOALS))


class StandardOptimizeGoal(BaseGoalSchema):

    factor = fields.Nested(FactorSchema(), required=True)


class CostOptimizeGoal(BaseGoalSchema):
    type = fields.String(validate=validate.OneOf(["COST"]), required=True)
    cost_function = fields.Nested(CostSchema, many=True, required=True)

    @validates_schema
    def max_functions(self, data, **kwargs):

        if len(data["cost_function"]) > 3:
            raise ValidationError(
                {"cost_function": ["No more than three costs functions allowed"]}
            )


class Prediction:
    def __init__(self, client: "fero.Fero", prediction_request_id: str):
        self._client = client
        self.request_id = prediction_request_id
        self._data_cache = None
        self._complete = False

    @property
    def _data(self):

        # check if it's complete which also pulls data
        if not self.complete:
            return None

        return self._data_cache

    @property
    def complete(self):
        """Checks if the Prediction is complete"""
        if not self._complete:
            self._data_cache = self._client.get(
                f"/api/prediction_results/{self.request_id}/"
            )
            self._complete = self._data_cache["state"] == "C"

        return self._complete

    def get_results(self, format="dataframe") -> Union[pd.DataFrame, List[dict]]:

        if not self.complete:
            raise FeroError("Prediction is not complete.")
        data = self._data["results_data"]["data"]["values"]
        if format == "records":
            return [
                {col: val for col, val in zip(data["columns"], row)}
                for row in data["data"]
            ]
        else:
            return pd.DataFrame(
                data["data"], columns=data["columns"], index=data["index"]
            )


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

    def _get_factor_dtype(self, factor_name: str) -> Union[str, None]:
        """Returns the dtype of a factor or None if the factor isn't found"""
        try:
            goal_data = next(
                f for f in self._regression_factors if f["factor"] == factor_name
            )
            return goal_data["dtype"]
        except StopIteration:
            return None

    def _verify_standard_goal(self, goal: dict):
        """Verifies the goal config relative to the analysis"""
        goal_name = goal["factor"]["name"]

        # The goal label must be a target or factor
        if goal_name not in self.target_names + self.factor_names:
            raise FeroError(f'"{goal_name}" is not a valid goal')

        # If this is a factor makes sure it's not a float
        if (
            goal_name not in self.target_names
            and self._get_factor_dtype(goal_name) != "float"
        ):
            raise FeroError("Goal must be a float")

    def _verify_cost_goal(self, goal: dict):
        """Verify that a cost goal is correct"""
        for factor in goal["cost_function"]:
            factor_name = factor["name"]
            factor_type = self._get_factor_dtype(factor_name)
            if factor_type is None:
                raise FeroError(f'"{factor_name}" is not a factor')
            # Implicitly find missing factors
            if factor_type != "float":
                raise FeroError("Cost functions factors must be factors")

    def _verify_constraints(self, constraints: List[dict]):
        """verify provided constraints are in the analysis"""
        all_names = self.factor_names + self.target_names
        for constraint in constraints:
            constraint_name = constraint["name"]
            if constraint_name not in all_names:
                raise FeroError(
                    f'Constraint "{constraint_name}" is not in this analysis'
                )

    def _cross_verify_optimization(self, goal: dict, constraints: List[dict]):
        """Verify the config has the correct combined targets, costs and factors"""
        is_cost = goal.get("type", None) == "COST"

        cost_factors = []
        goal_factor = []
        target_factor = []
        constraint_factors = []
        constraint_targets = []

        if is_cost:
            for factor in goal["cost_function"]:
                cost_factors.append(factor["name"])
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

        if len(constraint_factors + cost_factors + goal_factor) > 3:
            raise FeroError(
                "A maximum of three factors can be specified in an optimization"
            )

    def _verify_fixed_factors(self, fixed_factors: dict):
        """Check that the provided fixed factors are in the analysis"""
        all_columns = self.target_names + self.factor_names
        for key in fixed_factors.keys():
            if key not in all_columns:
                raise FeroError(f'"{key}" is not a valid factor')

    def _get_basis_values(self):
        """Gets median fixed values from presentation data"""
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
        self, name, goal, constraints, fixed_factors, include_confidence_intervals
    ) -> dict:

        is_cost = goal.get("type", None) == "COST"
        ci_value = "include" if include_confidence_intervals else "exclude"
        opt_request = {
            "name": name,
            "description": "",
            "prediction_type": "O",
            # These appear to just be, uhm, here
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
        }
        basis_values = self._get_basis_values()
        basis_values.update(fixed_factors)
        input_data["basisValues"] = basis_values

        factor_bounds = [
            {
                "factor": c["name"],
                "lowerBound": c["min"],
                "upperBound": c["max"],
                "dtype": self._get_factor_dtype(c["name"]),
            }
            for c in constraints
            if c["name"] in self.factor_names
        ]

        target_bounds = [
            {
                "factor": c["name"],
                "lowerBound": c["min"],
                "upperBound": c["max"],
                "dtype": "float",
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
            dtype = self._get_factor_dtype(goal["factor"]["name"])
            goal_bound = {
                "factor": goal["factor"]["name"],
                "lowerBound": goal["factor"]["min"],
                "upperBound": goal["factor"]["max"],
                "dtype": dtype if dtype is not None else "float",
            }

            if dtype is None:
                goal_bound["confidenceInterval"] = ci_value

            bounds.append(goal_bound)

        input_data["bounds"] = bounds
        opt_request["input_data"] = input_data

        return opt_request

    def _request_prediction(
        self, prediction_request: dict, synchonous: bool
    ) -> Prediction:

        response_data = self._client.post(
            f"/api/analyses/{str(self.uuid)}/predictions/", prediction_request
        )
        prediction = Prediction(self._client, response_data["latest_results"])

        # If synchronous block until the prediction is complete
        while synchonous and not prediction.complete:
            time.sleep(0.5)

        return prediction

    def make_optimization(
        self,
        name: str,
        goal: dict,
        constraints: List[dict],
        fixed_factors: Optional[dict] = None,
        include_confidence_intervals: bool = False,
        synchronous: bool = True,
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
        [
            {"name": "factor2",  "min": 10, "max": 10}
            {"name": "target1", "min": 100, "max": 500}
        ]

        :param name: Name for this optimizatino
        :type name: str
        :param goal: [description]
        :type goal: dict
        :param constrains: A dictionary
        :type constrains: dict
        :param descrfixed_factors: [description], defaults to None
        :type descrfixed_factors: dict, optional
        :param synchronous: [description], defaults to True
        :type synchronous: bool, optional
        :return: [description]
        :rtype: Prediction
        """
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
            raise FeroError(f"Error validating goal <f{str(constraints_validation)}>")

        if cost_goal:
            self._verify_cost_goal(goal)
        else:
            self._verify_standard_goal(goal)

        self._verify_constraints(constraints)
        self._cross_verify_optimization(goal, constraints)
        self._verify_fixed_factors(fixed_factors)

        optimize_request = self._build_optimize_request(
            name, goal, constraints, fixed_factors, include_confidence_intervals
        )
        print(optimize_request)
        return self._request_prediction(optimize_request, synchronous)
