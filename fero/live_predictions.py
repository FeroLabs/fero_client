"""This module holds classes representing live predictions and optimizations from Fero.

Fero runs four kinds of live prediction against an analysis -- point predictions and
optimizations, each in a standard and a "flexible" variant. A flexible prediction evaluates
the same request against several scenarios at once, so it is represented here as a single
object carrying a list of `scenarios` rather than as several separate predictions.

Use `LivePredictionType` to choose which kind to request and `LivePredictionSort` to choose
their ordering; both are accepted by `fero.analysis.Analysis.get_live_predictions`.
"""

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from fero import FeroError

# The interval keys reported for every predicted target, ordered low to high.
TARGET_INTERVAL_KEYS = ("low90", "low50", "mid", "high50", "high90")

# Fero caps a single `get_live_predictions` request at the API page size so it cannot be
# used as a bulk export channel.
MAX_LIMIT = 1000
DEFAULT_LIMIT = 10


class LivePredictionType(Enum):
    """The kind of live prediction to retrieve from an analysis."""

    #: A live point prediction of the analysis targets against a single basis.
    PREDICTION = "prediction"
    #: A live point prediction evaluated against several scenarios at once.
    FLEXIBLE_PREDICTION = "flexible_prediction"
    #: A live optimization of the analysis factors against a single basis.
    OPTIMIZATION = "optimization"
    #: A live optimization evaluated against several scenarios at once.
    FLEXIBLE_OPTIMIZATION = "flexible_optimization"


class LivePredictionSort(Enum):
    """The ordering to apply to retrieved live predictions."""

    #: Most recently created first. This is the default.
    NEWEST_FIRST = "-created"
    #: Oldest created first.
    OLDEST_FIRST = "created"
    #: Ordered by the live data's own ordering value, highest first. Prefer this over
    #: `NEWEST_FIRST` when the source data can arrive out of order.
    LIVE_ORDER_DESCENDING = "-live_order_value"
    #: Ordered by the live data's own ordering value, lowest first.
    LIVE_ORDER_ASCENDING = "live_order_value"


def coerce_enum_value(value: Any, enum_class: type, argument_name: str) -> str:
    """Coerce `value` to the string value of `enum_class`, raising a clear `FeroError`.

    Both the enum members and their plain string equivalents are accepted so callers are
    not forced to import the enums.
    """
    if isinstance(value, enum_class):
        return value.value

    valid = [member.value for member in enum_class]
    if value in valid:
        return value

    raise FeroError(
        f'"{value}" is not a valid {argument_name}. Expected one of {", ".join(sorted(valid))}.'
    )


def validate_limit(limit: int) -> int:
    """Check `limit` is a whole number within the range Fero will accept."""
    # bool is an int subclass and would otherwise sail through as 0 or 1.
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise FeroError(f"limit must be an integer, got {limit!r}.")
    if limit < 1 or limit > MAX_LIMIT:
        raise FeroError(f"limit must be between 1 and {MAX_LIMIT}, got {limit}.")
    return limit


def _parse_datetime(value: Optional[str]) -> Optional[datetime.datetime]:
    """Parse an ISO 8601 timestamp from the API, tolerating a trailing "Z"."""
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _frame_from_split(values: Optional[dict]) -> Optional[pd.DataFrame]:
    """Build a `DataFrame` from the split-orient frame the API returns."""
    if not values:
        return None
    return pd.DataFrame(
        values.get("data", []),
        columns=values.get("columns", []),
        index=values.get("index", []),
    )


class TargetPrediction:
    """The predicted distribution for a single target.

    `mid` is the expected value; the `low50`/`high50` and `low90`/`high90` pairs bound the
    50% and 90% confidence intervals around it. Any of them may be `None` if Fero could not
    produce that bound.
    """

    def __init__(self, data: Dict[str, Any]):
        """Create a `TargetPrediction` from the interval data returned by the API."""
        self._data = data
        for key in TARGET_INTERVAL_KEYS:
            setattr(self, key, data.get(key))

    def __repr__(self) -> str:
        """Represent the `TargetPrediction` by its expected value."""
        return f"<TargetPrediction mid={self.mid}>"

    __str__ = __repr__

    def to_dict(self) -> Dict[str, Any]:
        """Return the target's intervals as a plain dictionary."""
        return {key: getattr(self, key) for key in TARGET_INTERVAL_KEYS}


def _targets_frame(targets: Dict[str, TargetPrediction]) -> pd.DataFrame:
    """Build a frame of one row per target with a column per interval."""
    return pd.DataFrame(
        [target.to_dict() for target in targets.values()],
        index=list(targets),
        columns=list(TARGET_INTERVAL_KEYS),
    )


class PredictionScenario:
    """The result of one scenario of a flexible live prediction."""

    def __init__(self, data: Dict[str, Any]):
        """Create a `PredictionScenario` from a scenario entry returned by the API."""
        self._data = data
        #: The scenario's position within its prediction.
        self.index: int = data.get("index")
        #: The factor values this scenario was evaluated against, or `None` if Fero could
        #: not reconstruct them.
        self.basis: Optional[Dict[str, Any]] = data.get("basis")
        #: "SUCCESS" or "FAILURE" for this scenario specifically.
        self.status: Optional[str] = data.get("status")
        #: The failure message if this scenario failed.
        self.message: Optional[str] = data.get("message")
        #: The predicted distribution per target name. Empty if this scenario failed.
        self.targets: Dict[str, TargetPrediction] = {
            name: TargetPrediction(target)
            for name, target in (data.get("targets") or {}).items()
        }

    def __repr__(self) -> str:
        """Represent the `PredictionScenario` by its index."""
        return f"<PredictionScenario index={self.index}>"

    __str__ = __repr__

    def to_dataframe(self) -> pd.DataFrame:
        """Return this scenario's targets as a frame of one row per target."""
        return _targets_frame(self.targets)


class OptimizationScenario:
    """The result of one scenario of a flexible live optimization."""

    def __init__(self, data: Dict[str, Any]):
        """Create an `OptimizationScenario` from a scenario entry returned by the API."""
        self._data = data
        #: The scenario's position within its optimization.
        self.index: int = data.get("index")
        #: The factor values this scenario was optimized around, or `None` if Fero could
        #: not reconstruct them.
        self.basis: Optional[Dict[str, Any]] = data.get("basis")

    def __repr__(self) -> str:
        """Represent the `OptimizationScenario` by its index."""
        return f"<OptimizationScenario index={self.index}>"

    __str__ = __repr__

    def to_dataframe(self) -> Optional[pd.DataFrame]:
        """Return this scenario's optimal values, or `None` if it produced none."""
        return _frame_from_split(self._data.get("values"))


class LivePredictionBase:
    """Fields shared by every kind of live prediction.

    A live prediction that is not yet `complete`, or that failed, still appears in a
    listing; check `complete` and `status` before relying on its results.
    """

    def __init__(self, data: Dict[str, Any]):
        """Create a live prediction object from an API result entry."""
        self._data = data
        #: Unique identifier of this prediction.
        self.uuid: str = data.get("uuid")
        #: Identifier of the workspace holding this prediction's revisions.
        self.workspace_id: str = data.get("workspace_id")
        #: Identifier of the analysis this prediction belongs to.
        self.analysis_id: str = data.get("analysis_id")
        #: Name of the prediction.
        self.name: Optional[str] = data.get("name")
        #: Description of the prediction.
        self.description: Optional[str] = data.get("description")
        #: The live data tag this prediction was made for, if any.
        self.prediction_tag: Optional[str] = data.get("prediction_tag")
        #: The `LivePredictionType` value describing this prediction.
        self.type: str = data.get("type")
        #: When Fero created the prediction.
        self.created: Optional[datetime.datetime] = _parse_datetime(data.get("created"))
        #: When the prediction was last modified.
        self.modified: Optional[datetime.datetime] = _parse_datetime(
            data.get("modified")
        )
        #: Username of the prediction's creator, if it has one.
        self.created_by: Optional[str] = data.get("created_by")
        #: The ordering value carried on the source live data, if any.
        self.live_order_value: Optional[str] = data.get("live_order_value")
        #: Identifier of the model revision used to make the prediction.
        self.revision_model_id: Optional[str] = data.get("revision_model_id")
        #: Whether Fero has finished running this prediction.
        self.complete: bool = bool(data.get("complete"))
        #: "SUCCESS" or "FAILURE" once the prediction is complete.
        self.status: Optional[str] = data.get("status")
        #: The failure message if the prediction failed.
        self.message: Optional[str] = data.get("message")
        #: The factor values the prediction was made against.
        self.basis: Dict[str, Any] = data.get("basis") or {}

    def __repr__(self) -> str:
        """Represent the live prediction by its type and identifier."""
        return f"<{type(self).__name__} uuid={self.uuid}>"

    __str__ = __repr__

    def _require_complete(self):
        """Raise a `FeroError` unless this prediction finished successfully."""
        if not self.complete:
            raise FeroError(f"Live prediction {self.uuid} is not complete.")
        if self.status == "FAILURE":
            raise FeroError(
                f"Live prediction {self.uuid} failed: {self.message or 'no message provided'}"
            )


class LivePrediction(LivePredictionBase):
    """A live point prediction of an analysis' targets against a single basis."""

    def __init__(self, data: Dict[str, Any]):
        """Create a `LivePrediction` from an API result entry."""
        super().__init__(data)
        #: The predicted distribution per target name.
        self.targets: Dict[str, TargetPrediction] = {
            name: TargetPrediction(target)
            for name, target in (data.get("targets") or {}).items()
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return the prediction as a frame of one row per target.

        :raises FeroError: Raised if the prediction is not complete or has failed.
        :return: A frame indexed by target name with a column per confidence interval
        :rtype: pd.DataFrame
        """
        self._require_complete()
        return _targets_frame(self.targets)


class FlexibleLivePrediction(LivePredictionBase):
    """A live point prediction evaluated against several scenarios at once."""

    def __init__(self, data: Dict[str, Any]):
        """Create a `FlexibleLivePrediction` from an API result entry."""
        super().__init__(data)
        #: One `PredictionScenario` per scenario, in the order Fero evaluated them.
        self.scenarios: List[PredictionScenario] = [
            PredictionScenario(scenario) for scenario in (data.get("scenarios") or [])
        ]
        #: Index of the scenario Fero considers most representative.
        self.default_scenario_index: Optional[int] = data.get("default_scenario_index")

    @property
    def default_scenario(self) -> Optional[PredictionScenario]:
        """Get the scenario Fero considers most representative, if there is one."""
        if self.default_scenario_index is None:
            return None
        try:
            return self.scenarios[self.default_scenario_index]
        except IndexError:
            return None

    def to_dataframe(self) -> pd.DataFrame:
        """Return every scenario as a frame of one row per scenario and target.

        :raises FeroError: Raised if the prediction is not complete or has failed.
        :return: A frame indexed by (scenario index, target name)
        :rtype: pd.DataFrame
        """
        self._require_complete()
        frames = []
        for scenario in self.scenarios:
            frame = scenario.to_dataframe()
            frame.index = pd.MultiIndex.from_product(
                [[scenario.index], frame.index], names=["scenario", "target"]
            )
            frames.append(frame)

        if not frames:
            return pd.DataFrame(columns=list(TARGET_INTERVAL_KEYS))

        return pd.concat(frames)


class LiveOptimization(LivePredictionBase):
    """A live optimization of an analysis' factors against a single basis."""

    def to_dataframe(self) -> Optional[pd.DataFrame]:
        """Return the optimal factor and target values found by the optimization.

        :raises FeroError: Raised if the optimization is not complete or has failed.
        :return: A frame of the optimal values, or `None` if the optimization found none
        :rtype: Optional[pd.DataFrame]
        """
        self._require_complete()
        return _frame_from_split(self._data.get("values"))


class FlexibleLiveOptimization(LivePredictionBase):
    """A live optimization evaluated against several scenarios at once."""

    def __init__(self, data: Dict[str, Any]):
        """Create a `FlexibleLiveOptimization` from an API result entry."""
        super().__init__(data)
        #: One `OptimizationScenario` per scenario, in the order Fero evaluated them.
        self.scenarios: List[OptimizationScenario] = [
            OptimizationScenario(scenario) for scenario in (data.get("scenarios") or [])
        ]
        #: Index of the scenario Fero considers riskiest, and so most worth acting on.
        self.default_scenario_index: Optional[int] = data.get("default_scenario_index")

    @property
    def default_scenario(self) -> Optional[OptimizationScenario]:
        """Get the scenario Fero considers riskiest, if there is one."""
        if self.default_scenario_index is None:
            return None
        try:
            return self.scenarios[self.default_scenario_index]
        except IndexError:
            return None

    def to_dataframe(self) -> pd.DataFrame:
        """Return every scenario's optimal values in a single frame.

        :raises FeroError: Raised if the optimization is not complete or has failed.
        :return: A frame of the optimal values with a `scenario` column identifying each
        :rtype: pd.DataFrame
        """
        self._require_complete()
        frames = []
        for scenario in self.scenarios:
            frame = scenario.to_dataframe()
            if frame is None:
                continue
            frame = frame.copy()
            frame.insert(0, "scenario", scenario.index)
            frames.append(frame)

        if not frames:
            return pd.DataFrame(columns=["scenario"])

        return pd.concat(frames, ignore_index=True)


# The object built for each `LivePredictionType`.
LIVE_PREDICTION_CLASSES = {
    LivePredictionType.PREDICTION.value: LivePrediction,
    LivePredictionType.FLEXIBLE_PREDICTION.value: FlexibleLivePrediction,
    LivePredictionType.OPTIMIZATION.value: LiveOptimization,
    LivePredictionType.FLEXIBLE_OPTIMIZATION.value: FlexibleLiveOptimization,
}


def live_prediction_from_data(data: Dict[str, Any]) -> LivePredictionBase:
    """Build the object matching the `type` reported on an API result entry."""
    prediction_class = LIVE_PREDICTION_CLASSES.get(data.get("type"))
    if prediction_class is None:
        raise FeroError(
            f'Fero returned an unknown live prediction type "{data.get("type")}". '
            "Upgrading the fero package may add support for it."
        )
    return prediction_class(data)
