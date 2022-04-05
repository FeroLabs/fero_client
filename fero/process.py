"""This module defines classes related to a fero process."""

import fero
import requests
from tempfile import NamedTemporaryFile
from functools import lru_cache as memoized
from typing import Sequence, Optional, Union
from .common import FeroObject, poll
from marshmallow import (
    Schema,
    fields,
    validate,
    EXCLUDE,
)
import pandas as pd
from .exceptions import FeroError

# 1 Mb
CHUNK_SIZE = 1048576


class DataRequestError(FeroError):
    """Raised when data cannot be generated."""

    pass


class FeroProcessTypes:
    """An Enum class representing the type of fero process."""

    ADVANCED = "A"
    BATCH = "B"
    CONTINUOUS = "C"

    @classmethod
    def validator(cls):
        """Validate that a given process type is one of the possible options."""
        return validate.OneOf([cls.ADVANCED, cls.BATCH, cls.CONTINUOUS])


class SnapShotStatus:
    """An Enum class representing the status of a process snpashot."""

    READY = "R"
    INITIALIZED = "I"
    PROCESSING = "P"
    ERROR = "E"

    @classmethod
    def validator(cls):
        """Validate that a given snapshot status is one of the possible options."""
        return validate.OneOf([cls.READY, cls.INITIALIZED, cls.PROCESSING, cls.ERROR])


class TagSchema(Schema):
    """A schema to store data related to a fero process tag."""

    class Meta:
        """
        Specify that unknown fields included on this schema should be excluded.

        See https://marshmallow.readthedocs.io/en/stable/quickstart.html#handling-unknown-fields.
        """

        unknown = EXCLUDE

    name = fields.String(required=True)
    description = fields.String(required=True)
    display_format = fields.Integer(required=True, allow_none=True)
    cost = fields.Float(required=True, allow_none=True)
    unit = fields.String(required=True)
    alias = fields.String(required=True)
    minimum = fields.Dict(required=True, allow_none=True)
    maximum = fields.Dict(required=True, allow_none=True)
    lower_limit = fields.Dict(required=True, allow_none=True)
    upper_limit = fields.Dict(required=True, allow_none=True)
    step_size = fields.Float(required=True, allow_none=True)
    disabled = fields.Boolean(required=True)
    proxy = fields.Integer(required=True)


class StageSchema(Schema):
    """A schema to store data related to a fero process stage."""

    class Meta:
        """
        Specify that unknown fields included on this schema should be excluded.

        See https://marshmallow.readthedocs.io/en/stable/quickstart.html#handling-unknown-fields.
        """

        unknown = EXCLUDE

    id = fields.Integer(required=True)
    name = fields.String(required=True)
    order = fields.Integer(required=True)
    tags = fields.List(fields.Nested(TagSchema), required=True)
    configuration = fields.Dict(required=True)


class NestedSnapshotSchema(Schema):
    """A schema to store data related to a fero process snapshot."""

    uuid = fields.String(required=True)
    status = fields.String(
        required=True,
        validate=SnapShotStatus.validator(),
    )


class ProcessSchema(Schema):
    """A schema to store data related to a fero process."""

    class Meta:
        """
        Specify that unknown fields included on this schema should be excluded.

        See https://marshmallow.readthedocs.io/en/stable/quickstart.html#handling-unknown-fields.
        """

        unknown = EXCLUDE

    api_id = fields.String(required=True)
    name = fields.String(required=True)
    created = fields.DateTime(require=True)
    modified = fields.DateTime(require=True)
    latest_revision_version = fields.Integer(required=True)
    username = fields.String(required=True)
    process_type = fields.String(required=True, validate=FeroProcessTypes.validator())
    product_type = fields.String(Required=False, allow_none=True)
    kind = fields.String(required=True, validate=validate.OneOf(["process"]))
    latest_ready_snapshot = fields.Nested(NestedSnapshotSchema, allow_none=True)
    # TODO update this to a nested schema once the configuration settles more
    data_config = fields.Dict(required=True)


class Tag(FeroObject):
    """High level object representing a process tag.

    A tag is a single measurement type for a process and specifies metadata associated with this measurement such
    as limits and cost.
    """

    schema_class = TagSchema

    def __init__(self, process: "Process", client: "fero.Fero", data: dict):
        """
        Create a new representation of a Tag from the given data.

        :param process: The process this tag belongs to
        :param client: The fero client object that gives access to the `process`
        :param data: A dictionary holding values for the fields of a `TagSchema`
        """
        self._process = process
        super().__init__(client, data)

    def __repr__(self) -> str:
        """Represent the `Tag` by its name and its `Process`."""
        return f"<Tag name={self.name} Process name={self._process}>"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        """Check whether `other` is a Tag with the same data as this tag."""
        return isinstance(other, Tag) and self._data == other._data

    @classmethod
    def tag_name(cls, instance: Union["Tag", str]) -> str:
        """
        Get the name of a given tag.

        :param instance: Either the `Tag` or the string name of the tag
        :return: The name of the specified tag
        """
        return instance if isinstance(instance, str) else instance.name


class Stage(FeroObject):
    """High level object representing a process stage.

    A tag is a discrete grouping of related tags that are part of process and describes metadata about how to process each tag
    based on the type of process and configuration of the stage.
    """

    schema_class = StageSchema

    def __init__(self, process: "Process", client: "fero.Fero", data: dict):
        """
        Create a new representation of a Stage from the given data.

        :param process: The process this stage belongs to
        :param client: The fero client object that gives access to the `process`
        :param data: A dictionary holding values for the fields of a `StageSchema`
        """
        self._process = process
        super().__init__(client, data)
        self._data["tags"] = [
            Tag(process, client, tdata) for tdata in self._data["tags"]
        ]

    def __repr__(self) -> str:
        """Represent the `Stage` by its name and its `Process`."""
        return f"<Stage name={self.name} Process name={self._process}>"

    __str__ = __repr__

    def __eq__(self, other) -> bool:
        """Check whether `other` is a Stage with the same data as this stage."""
        return isinstance(other, Stage) and self._data == other._data

    @property
    def tag_names(self) -> Sequence[str]:
        """Get the names of the tags in the stage.

        :return: A list of tag names in the stage
        """
        return [Tag.tag_name(t) for t in self.tags]


class Process(FeroObject):
    """High level object for interacting with a Process object defined in Fero.

    A Process is a set of configurations that describe an industrial process and allow
    Fero to combine, transform, and enrich raw data sources before the final data is
    used in an Analysis.
    """

    schema_class = ProcessSchema

    def __repr__(self) -> str:
        """Represent the `Process` by its name."""
        return f"<Process name={self.name}>"

    __str__ = __repr__

    @property
    @memoized(maxsize=1)
    def stages(self) -> Sequence[Stage]:
        """Get the stages for this process from Fero.

        Request is memoized to avoid unnecessary additional calls.

        :return: A List of stages
        """
        return [
            Stage(self, self._client, stage_data)
            for stage_data in self._client.get(
                f"/api/processes/{self.api_id}/stages/"
            ).get("stages")
        ]

    @property
    @memoized(maxsize=1)
    def tags(self) -> Sequence[Tag]:
        """Get the tags for this process from Fero.

        Request is memoized to avoid unnecessary additional calls.

        :return: List of tags associated with the process
        """
        return [
            Tag(self, self._client, tag_data)
            for tag_data in self._client.get(f"/api/processes/{self.api_id}/tags/").get(
                "tags"
            )
        ]

    def _tag_stage_index(self, tag: Union[Tag, str]) -> Union[int, None]:

        stage_idx = None
        tag_name = Tag.tag_name(tag)
        for idx, s in enumerate(self.stages):
            if tag_name in s.tag_names:
                stage_idx = idx
                break

        return stage_idx

    def get_tag_stage(self, tag: Union[Tag, str]) -> Union[Stage, None]:
        """Return the process stage specified in the stage or None if the tag is not found.

        :param tag: Tag to search for
        :return: [description]
        """
        stage_idx = self._tag_stage_index(tag)
        return None if stage_idx is None else self.stages[stage_idx]

    def get_stages_by_kpis(self, kpis: Sequence[Union[Tag, str]]) -> Sequence[Stage]:
        """Get a list of stages that would be used by a process for the given kpis.

        :param kpis: List of kpi tags
        :return: a list of stages
        """
        idxs = [self._tag_stage_index(k) for k in kpis]
        idxs = [i for i in idxs if i is not None]

        return [] if len(idxs) == 0 else self.stages[: max(idxs) + 1]

    def get_data(
        self,
        tags: Sequence[Union[Tag, str]],
        kpis: Optional[Sequence[Union[Tag, str]]] = None,
    ) -> pd.DataFrame:
        """Download a pandas data frame with specified process data.

        Returns a pandas data frame consisting of the specified tags and bounded by an optional list of kpis.
        For all process types specifying a kpi will remove any tags from stages that are after
        the stage the kpi is assigned to.  For advanced and batch process types not providing a kpi will
        get tags from all stages.  For a continuous process, a kpi must be specified.

        :param tags: List of Tags or tag names to include in the data frame
        :type tags: Sequence[Union[Tag, str]]
        :param kpis: List of tags or tag names to use for kpis, defaults to None
        :type kpis: Optional[Sequence[Union[Tag, str]]], optional
        :raises DataRequestError: Raised if data can not be generated
        :return: A data frame of the data generated by the process.
        :rtype: pd.DataFrame
        """
        if self.process_type == FeroProcessTypes.CONTINUOUS and kpis is None:
            raise DataRequestError(
                "A kpi must be specified to download continuous process data"
            )

        tag_names = [Tag.tag_name(tag) for tag in tags]
        kpi_names = None
        if kpis:
            kpi_names = [tag if isinstance(tag, str) else tag.name for tag in kpis]

        initial_response = self._client.post(
            f"/api/processes/{self.api_id}/download_process_data/",
            {"request_data": {"tags": tag_names, "kpis": kpi_names}},
        )

        request_id = initial_response["request_id"]

        def get_data():
            return self._client.post(
                f"/api/processes/{self.api_id}/download_process_data/",
                {"request_id": request_id},
            )

        download_response = poll(get_data, lambda d: d["complete"])

        download_data = download_response["data"]

        if not download_data["success"]:
            raise DataRequestError(download_data["message"])

        # Download the file and write as a stream since it could be large
        req = requests.get(download_data["download_url"], stream=True)
        with NamedTemporaryFile() as fp:
            for chunk in req.iter_content(chunk_size=CHUNK_SIZE):
                fp.write(chunk)
            fp.seek(0)
            df = pd.read_parquet(fp.name)
        return df
