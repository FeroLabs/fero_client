import fero
import requests
from tempfile import NamedTemporaryFile
from functools import lru_cache as memoized
from io import BytesIO
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
    pass


class FeroProcessTypes:
    ADVANCED = "A"
    BATCH = "B"
    CONTINUOUS = "C"

    @classmethod
    def validator(cls):
        return validate.OneOf([cls.ADVANCED, cls.BATCH, cls.CONTINUOUS])


class SnapShotStatus:
    READY = "R"
    INITIALIZED = "I"
    PROCESSING = "P"
    ERROR = "E"

    @classmethod
    def validator(cls):
        return validate.OneOf([cls.READY, cls.INITIALIZED, cls.PROCESSING, cls.ERROR])


class TagSchema(Schema):
    class Meta:
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


class StageSchema(Schema):
    class Meta:
        unknown = EXCLUDE

    id = fields.Integer(required=True)
    name = fields.String(required=True)
    order = fields.Integer(required=True)
    tags = fields.List(fields.Nested(TagSchema), required=True)
    configuration = fields.Dict(required=True)


class NestedSnapshotSchema(Schema):

    uuid = fields.String(required=True)
    status = fields.String(
        required=True,
        validate=SnapShotStatus.validator(),
    )


class ProcessSchema(Schema):
    class Meta:
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

    A tag is a single measurement type for a process and specifies metadata associated with this measurement since
    as limits and cost.
    """

    schema_class = TagSchema

    def __init__(self, process: "Process", client: "fero.Fero", data: dict):
        self._process = process
        super().__init__(client, data)

    def __repr__(self):
        return f"<Tag name={self.name} Process name={self._process}>"

    __str__ = __repr__


class Stage(FeroObject):
    """High level object representing a process stage.

    A tag is a discrete grouping of related tags that are part of process and describes metadata about how to process each tag
    based on the type of process and configuration of the stage.
    """

    schema_class = StageSchema

    def __init__(self, process: "Process", client: "fero.Fero", data: dict):
        self._process = process
        super().__init__(client, data)

    def __repr__(self):
        return f"<Stage name={self.name} Process name={self._process}>"

    __str__ = __repr__


class Process(FeroObject):
    """High level object for interacting with a Process object defined in Fero.

    A Process is a set of configurations that describe an industrial process and allow
    Fero to combine, transform, and enrich raw data sources before the final data is
    used in an Analysis.
    """

    schema_class = ProcessSchema

    def __repr__(self):
        return f"<Process name={self.name}>"

    __str__ = __repr__

    @property
    @memoized(maxsize=1)
    def stages(self) -> Sequence[Stage]:
        """Memoized request to get stages from Fero

        :return: A List of stages
        :rtype: List[Stage]
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
        """Memoized request to get tags from Fero

        :return: List of tags associated with the process
        :rtype: Sequence[Tag]
        """
        return [
            Tag(self, self._client, tag_data)
            for tag_data in self._client.get(f"/api/processes/{self.api_id}/tags/").get(
                "tags"
            )
        ]

    def get_data(
        self,
        tags: Sequence[Union[Tag, str]],
        kpis: Optional[Sequence[Union[Tag, str]]] = None,
    ) -> pd.DataFrame:

        if self.process_type == FeroProcessTypes.CONTINUOUS and kpis is None:
            raise DataRequestError(
                "A kpi must be specified to download continuous process data"
            )

        tag_names = [tag if isinstance(tag, str) else tag.name for tag in tags]
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
