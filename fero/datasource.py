"""This Module defines classes related to a fero datasource."""

import os
import time

import requests
import fero
from fero import FeroError
from typing import Optional, Union
from marshmallow import (
    Schema,
    fields,
    validate,
    EXCLUDE,
)

from .common import FeroObject

# 1 Mb
CHUNK_SIZE = 1048576


class DataSourceSchema(Schema):
    """A schema to store data related to a fero datasource."""

    class Meta:
        """
        Specify that unknown fields included on this schema should be excluded.

        See https://marshmallow.readthedocs.io/en/stable/quickstart.html#handling-unknown-fields.
        """

        unknown = EXCLUDE

    uuid = fields.UUID(required=True)

    # deprecated in favor of primary_key_columns
    primary_key_column = fields.String(required=False, allow_none=True)
    primary_key_columns = fields.List(fields.String, required=False, allow_none=True)
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
    raw_file = fields.Url(required=False, allow_none=True)
    download_url = fields.Url(required=False, allow_none=True)
    hidden = fields.Boolean(required=False)


class DataSource(FeroObject):
    """An object for interacting with a specific Data Source on Fero.

    Adding data to a Data Source allows Fero to interface with that data.
    This data can then be added to a Fero Process and later analyzed in
    an Asset or Analysis.
    """

    schema_class = DataSourceSchema

    def __repr__(self) -> str:
        """Represent the `DataSource` by its name."""
        return f"<Data Source name={self.name}>"

    __str__ = __repr__

    def append_csv(self, file_path: str, wait_until_complete: bool = False):
        """Append a specified csv file to the data source.

        :param file_path: Location of the csv file to append
        :type file_path: str
        :raises FeroError: Raised if the file does not match a naive csv check
        """
        if not file_path.endswith(".csv"):
            raise FeroError("Fero only supports csv appends")

        file_name = os.path.basename(file_path)

        inbox_response = self._client.post(
            f"/api/v2/data_source/{self.uuid}/inbox_url/",
            {"file_name": file_name, "action": "A"},
        )
        with open(file_path) as fp:
            self._client.upload_file(inbox_response, file_name, fp)

        upload_status = UploadedFileStatus(self._client, inbox_response["upload_uuid"])
        return (
            upload_status.wait_until_complete()
            if wait_until_complete
            else upload_status
        )

    def replace_csv(self, file_path: str, wait_until_complete: bool = False):
        """Append a specified csv file to the data source.

        :param file_path: Location of the csv file to append
        :type file_path: str
        :raises FeroError: Raised if the file does not match a naive csv check
        """
        if not file_path.endswith(".csv"):
            raise FeroError("Fero only supports csv appends")

        file_name = os.path.basename(file_path)

        inbox_response = self._client.post(
            f"/api/v2/data_source/{self.uuid}/inbox_url/",
            {"file_name": file_name, "action": "R"},
        )
        with open(file_path) as fp:
            self._client.upload_file(inbox_response, file_name, fp)

        upload_status = UploadedFileStatus(self._client, inbox_response["upload_uuid"])

        return (
            upload_status.wait_until_complete()
            if wait_until_complete
            else upload_status
        )

    def get_download_status(self) -> Optional[str]:
        """Get current status of the data source download.

        :return: String if file download is completely processed, None if still processing
        """
        data = self._client.get(f"/api/v2/data_source/{self.uuid}/")
        self._data = self.schema_class().load(data)
        return self.download_url

    def wait_until_download_complete(self):
        """Continuously check the status of the download until it succeeds or fails."""
        status = self.get_download_status()
        while not status:
            time.sleep(0.5)
            status = self.get_download_status()

    def download(self, raw=False, file=None) -> str:
        """Download data source data as a csv.

        :param raw: Whether to download the raw data
        :raises FeroError: Raised if the raw download is not available
        :return: The local filename of the download
        """
        if raw:
            url = self.raw_file
            if url is None:
                raise FeroError("Raw download for that datasource is not available.")

        else:
            url = self.download_url
            if url is None:
                self._client.get(f"/api/v2/data_source/{self.uuid}/create_download/")
                self.wait_until_download_complete()
                url = self.download_url

        # Download the file and write as a stream since it could be large
        req = requests.get(url, stream=True)

        def _write_chunks(fp):
            for chunk in req.iter_content(chunk_size=CHUNK_SIZE):
                fp.write(chunk)

        if file is None:
            download_filename = f'fero{"-raw" if raw else ""}-ds-{self.uuid}.csv'
            with open(download_filename, "wb") as fp:
                _write_chunks(fp)
        else:
            download_filename = file.name
            _write_chunks(file)
            file.seek(0)

        return download_filename


class UploadedFilesSchema(Schema):
    """A schema to store data related to files uploaded to fero."""

    class Meta:
        """
        Specify that unknown fields included on this schema should be excluded.

        See https://marshmallow.readthedocs.io/en/stable/quickstart.html#handling-unknown-fields.
        """

        unknown = EXCLUDE

    uuid = fields.UUID(required=True)

    INITIALIZED = "I"
    PARSING = "P"
    ANALYZING = "A"
    CREATING = "C"
    ERROR = "E"
    DONE = "D"
    DELETING = "R"  # R for removing
    USER_CONFIRMATION = "U"

    status = fields.String(
        validate=validate.OneOf(["I", "P", "A", "C", "R", "D", "E", "U"]), required=True
    )

    error_notices = fields.Dict(
        required=True, default=lambda: {"global_notices": [], "parsing_notices": []}
    )


class UploadedFileStatus:
    """An object to track the status of a file during its upload."""

    def __init__(self, client: "fero.Fero", id: str):
        """Create an `UploadedFileStatus` from the uplolad ID to track the file's upload status."""
        self._id = id
        self._client = client
        self._status_data = None
        self._schema = UploadedFilesSchema()

    @staticmethod
    def _check_status_complete(status: Optional[dict]) -> bool:
        """Check status of the latest uploaded file response.

        Returns true if complete, false if not complete and raises an error if the status is error.
        """
        if status is None or status["status"] not in [
            UploadedFilesSchema.ERROR,
            UploadedFilesSchema.DONE,
        ]:
            return False

        if status["status"] == UploadedFilesSchema.ERROR:
            errors = [
                f'"{str(e)}"'
                for e in status["error_notices"]["global_notices"]
                + status["error_notices"]["parsing_notices"]
            ]

            error_message = f"Unable to upload file.  The following error(s) occurred: {', '.join(errors)}"
            raise FeroError(error_message)

        return True

    def get_upload_status(self) -> Union[dict, None]:
        """Get current status of the uploaded files object.

        :return: Dict if it is completely processed, None if still processing
        :raises FeroError: Raised if fero was unable to process the file
        """
        raw_data = self._client.get(
            f"/api/v2/uploaded_files/{self._id}/", allow_404=True
        )
        data = None
        if raw_data is not None:
            data = self._schema.load(raw_data)

        return data

    def wait_until_complete(self) -> "UploadedFileStatus":
        """Continuously check the status of the upload until it succeeds or fails."""
        status = self.get_upload_status()
        while not self._check_status_complete(status):
            time.sleep(0.5)
            status = self.get_upload_status()

        return self
