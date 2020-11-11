import os
from os import name
import fero
import enum
from fero import FeroError
from typing import Optional, List, IO, Union
import requests
from marshmallow import (
    Schema,
    fields,
    EXCLUDE,
    validate,
    ValidationError,
    validates_schema,
)


class DataSourceColumnSchema(Schema):

    name = fields.String(required=True)
    guess = fields.String(required=True)
    format = fields.String()

    @validates_schema
    def format_only_datetime(self, data, **kwargs):
        if data["guess"] != "datetime" and "format" in data:
            raise ValidationError("Format specified for non-date time column")


class ParsedSchemaSchema(Schema):

    columns = fields.List(fields.Nested(DataSourceColumnSchema), required=True)
    version = fields.Constant("2")
    kind = fields.Constant("ParsedSchema")


class DataSourceStatuses(enum.Enum):
    INITIALIZED = "I"
    PROCESSING = "P"
    READY = "R"
    ERROR = "E"

    @classmethod
    def choices(cls) -> List[str]:
        return [v.value for v in cls]


class DataSourceSchema(Schema):

    uuid = fields.UUID(required=True)

    name = fields.String(required=True)

    latest_errors = fields.String(load_only=True, allow_none=True)
    created = fields.DateTime(required=True, load_only=True)
    modified = fields.DateTime(required=True, load_only=True)

    status = fields.String(
        required=True, validate=validate.OneOf(DataSourceStatuses.choices())
    )

    schema = fields.Nested(ParsedSchemaSchema, allow_none=True)

    down_sampled = fields.Bool(default=False)


class UploadFileStatuses(enum.Enum):
    INITIALIZED = "I"
    PARSING = "P"
    ANALYZING = "A"
    CREATING = "C"
    ERROR = "E"
    DONE = "D"
    DELETING = "R"  # R for removing
    USER_CONFIRMATION = "U"

    @classmethod
    def choices(cls) -> List[str]:
        return [v.value for v in cls]


class UploadedFilesSchema(Schema):
    class Meta:
        unknown = EXCLUDE

    uuid = fields.UUID(required=True)
    name = fields.String(required=True)
    status = fields.String(
        required=True, validate=validate.OneOf(UploadFileStatuses.choices())
    )
    parsed_schema = fields.Nested(ParsedSchemaSchema, load_only=True, allow_none=True)
    uploaded_data_configuration = fields.Dict(load_only=True, allow_none=True)
    schema_overwrites = fields.Nested(ParsedSchemaSchema, allow_none=True)
    latest_errors = fields.String(load_only=True, allow_none=True)
    created = fields.DateTime(required=True, load_only=True)
    modified = fields.DateTime(required=True, load_only=True)


class DataSource:

    _data_source = None
    _uploaded_files = None

    def __init__(
        self,
        client: "fero.Fero",
        data_source_data: Optional[dict] = None,
        uploaded_files_data: Optional[dict] = None,
    ):

        self._client = client
        if data_source_data is not None:
            self._data_source = DataSourceSchema().load(data_source_data)

        if uploaded_files_data is not None:
            self._uploaded_files = UploadedFilesSchema().load(uploaded_files_data)

    @property
    def instantiated(self):
        """Indicates whether the data source has been fully created or is still being configured"""
        return self._data_source is not None

    @property
    def uploaded_data_configuration(self):
        return self._uploaded_files.get("uploaded_data_configuration", None)

    def _create_uploaded_files(self, name):

        data = {"name": name}

        res = self._client.post("/api/v2/uploaded_files/", data)
        self._uploaded_files = UploadedFilesSchema().load(res)

    def _upload_file(self, fp: IO) -> None:
        """Uploads a single file to the uploaded files location"""

        file_name = os.path.basename(fp.name)

        inbox_response = self._client.get(
            f"/api/v2/uploaded_files/{self._uploaded_files['uuid']}/inbox_url/",
            params={"file_name": file_name},
        )

        files = {
            "file": (
                file_name,
                fp,
            )
        }

        res = requests.post(
            inbox_response["url"],
            data=inbox_response["fields"],
            files=files,
        )

        if res.status_code != 204:
            raise FeroError("Error Uploading File")

    def upload_files(self, files: List[Union[IO, str]], name=None):
        """Uploads a file to Fero, creating a file upload object if needed.

        :param file: [description]
        :type file: Union[IO, str]
        """
        if self.instantiated:
            raise FeroError("Unable to upload file to an instantiated data source")

        if len(files) < 1:
            raise FeroError("No files specified")

        fps = [open(fp) if isinstance(fp, str) else fp for fp in files]

        upload_name = name if name else os.path.basename(fps[0].name)
        self._create_uploaded_files(upload_name)

        for fp in fps:
            self._upload_file(fp)
            fp.close()

        self._client.post(
            f"/api/v2/uploaded_files/{self._uploaded_files['uuid']}/initial_upload_complete/",
            data=None,
        )

    def refresh(self):

        if self._uploaded_files is not None:
            upload_data = self._client.get(
                f"/api/v2/uploaded_files/{self._uploaded_files['uuid']}/"
            )
            self._uploaded_files = UploadedFilesSchema().load(upload_data)

    def delete(self):
        if self._uploaded_files is not None:
            self._client.delete(
                f"/api/v2/uploaded_files/{self._uploaded_files['uuid']}/"
            )
            self._uploaded_files = None

    def configure_csv_upload(
        self,
        separator=None,
        decimal_separator=None,
        thousands_separator=None,
        encoding=None,
    ):
        """Sets the upload as a csv and specifies configuration"""

        csv_config = {
            "kind": "CsvFileOptions",
            "separator": separator,
            "decimal_separator": decimal_separator,
            "thousands_separator": thousands_separator,
            "encoding": encoding,
        }

        if self._uploaded_files is not None:
            self._uploaded_files["uploaded_data_configuration"][
                "upload_format_configuration"
            ]["file_options"] = csv_config
        else:
            self._uploaded_files = {
                "uploaded_data_configuration": {
                    "upload_format_configuration": {"file_options": csv_config}
                }
            }

    def configure_pivot(
        self, name_column: str, value_column: str, join_column: Optional[str] = None
    ):

        pivot_config = {
            "name_column": name_column,
            "value_column": value_column,
            "join_column": join_column,
            "file_options": self._uploaded_files.get("file_options", None),
            "format_type": "pivoted",
        }

        if self._uploaded_files is not None:
            self._uploaded_files["uploaded_data_configuration"][
                "upload_format_configuration"
            ] = pivot_config
        else:
            self._uploaded_files = {
                "uploaded_data_configuration": {
                    "upload_format_configuration": pivot_config
                }
            }

    def update_file_configuration(self) -> dict:
        res = self._client.patch(
            f"/api/v2/uploaded_files/{self._uploaded_files['uuid']}/",
            {"uploaded_data_configuration": self.uploaded_data_configuration},
        )
        self._uploaded_files = UploadedFilesSchema().load(res)

        return self._uploaded_files

    def create_datasource(
        self, data_source_name: str, access_name: Optional[str] = None
    ):
        if self._uploaded_files is None:
            raise FeroError("No files specified to process")
        user = self._client.get_user()
        if access_name is None:
            access_name = user["profile"]["default_upload_ac"]["name"]

        if access_name not in [a["name"] for a in user["profile"]["write_accesses"]]:
            raise FeroError("Invalid Access requested")

        post_data = {
            "name": data_source_name,
            "requested_access": access_name,
            "uploaded_files_uuid": str(self._uploaded_files["uuid"]),
        }
        self._data_source = DataSourceSchema().load(
            self._client.post("/api/v2/data_source/", post_data)
        )
        return self._data_source
