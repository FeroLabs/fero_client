import os
import fero
from fero import FeroError

from marshmallow import (
    Schema,
    fields,
    validate,
    EXCLUDE,
)


class DataSourceSchema(Schema):
    class Meta:
        unknown = EXCLUDE

    uuid = fields.UUID(required=True)

    primary_key_column = fields.String(required=True, allow_none=True)
    primary_datetime_column = fields.String(required=True, allow_none=True)

    schema = fields.Dict(required=True, allow_none=True)

    name = fields.String(required=True)
    description = fields.String(required=True)

    created = fields.DateTime(required=True)
    modified = fields.DateTime(required=True)
    ac_name = fields.String(required=True)
    username = fields.String(required=True)

    INITIALIZED = "I"
    PROCESSING = "P"
    LOADING_FILE = "L"
    ANALYZING_FILE = "A"
    WRITING_FILE = "W"
    COMPRESSING_TABLE = "C"
    READY = "R"
    ERROR = "E"
    status = fields.String(
        validate=validate.OneOf(["I", "P", "L", "A", "W", "C", "R", "E"]), required=True
    )

    error_notices = fields.Dict(required=True, default=lambda: {"errors": []})
    progress = fields.Integer(required=True, default=0)

    overwrites = fields.Dict(required=True, allow_none=True)

    transformed_source = fields.Bool(required=True, default=False)
    live_source = fields.Bool(required=True, default=False)
    default_upload_config = fields.Dict(required=False)


class DataSource:
    def __init__(self, client: "fero.Fero", data: dict):
        self._client = client
        schema = DataSourceSchema()
        self._data = schema.load(data)

    def __getattr__(self, name: str):
        return self._data.get(name)

    def __repr__(self):
        return f"<Data Source name={self.name}>"

    __str__ = __repr__

    def append_csv(self, file_path: str):
        """Appends a specified csv file to the data source.

        :param file_path: Location of the csv file to append
        :type file_path: str
        :raises FeroError: Raised if the file does not match a naive csv check
        """
        if not file_path.endswith(".csv"):
            raise FeroError("Fero only supports csv appends")

        file_name = os.path.basename(file_path)

        inbox_response = self._client.post(
            f"/api/v2/data_source/{self.uuid}/inbox_url/", {"file_name": file_name}
        )
        with open(file_path) as fp:
            self._client.upload_file(inbox_response, file_name, fp)
