"""This Module holds the Fero client class, which is used to communicate with the Fero API."""

import os
import io
import re
import requests
import backoff
from urllib.parse import urlparse
from azure.storage.blob import BlobClient
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union, TextIO, Type, Sequence
from . import FeroError
from .analysis import Analysis
from .asset import Asset
from .datasource import DataSource
from .process import Process
from .workspace import Workspace
from .common import FeroObject

FERO_CONF_FILE = ".fero"


class Fero:
    """The basic client used for communicating the the Fero API."""

    def __init__(
        self,
        hostname: Optional[str] = "https://app.ferolabs.com",
        username: Optional[str] = None,
        password: Optional[str] = None,
        fero_token: Optional[str] = None,
        verify: bool = True,
    ):
        """Create a base client for communicating with the Fero API.

        This class uses a JWT to query the api in all cases, however it will attempt to obtain a token if a username
        and password are provided.  It attempts to procure the token in the following order.

        1.  A token is provided in the constructor.
        2.  A username and password are provided in the constructor.
        3.  A token is provided in the "FERO_TOKEN" env variable.
        4.  A token is defined in the user's .fero file.
        5.  A username and password are provided in the environment variables "FERO_PASSWORD" and "FERO_USERNAME".
        6.  A username and password are provided in the .fero file.

        :param hostname: URL of the Fero server, defaults to "https://app.ferolabs.com/"
        :type hostname: Optional[str], optional
        :param username: username to user to login, defaults to None
        :type username: Optional[str], optional
        :param password: password to use to login, defaults to None
        :type password: Optional[str], optional
        :param fero_token: JWT token to use, defaults to None
        :type fero_token: Optional[str], optional
        :param verify: whether requests should verify ssl, defaults to True
        :type verify: bool
        :raises FeroError: Raised if a token cannot be obtained
        """
        self._fero_token = None
        self._fero_conf_content = None
        self._password = None
        self._username = None
        self._verify = verify
        self._impersonated = None

        self._hostname = hostname.rstrip("/")

        if fero_token:
            self._fero_token = fero_token

        if username and password:
            self._username = username
            self._password = password
            self._fero_token = self._get_token_as_user()

        if self._fero_token is None:
            self._fero_token = self._get_token_string_from_local()

        if self._fero_token is None:
            self._fero_token = self._jwt_from_local_user_pass()

        if self._fero_token is None:
            raise FeroError("Could not login into Fero")

    @property
    def fero_conf_contents(self) -> str:
        """Cache the file content."""
        if self._fero_conf_content is None:
            self._get_fero_conf_contents()

        return self._fero_conf_content

    @property
    def current_user(self) -> str:
        """Return the username of the current user."""
        return self._impersonated or self._username

    def _get_token_string_from_local(self) -> Union[str, None]:
        """Check the local system for the JWT token string."""
        if "FERO_TOKEN" in os.environ:
            return os.environ.get("FERO_TOKEN")
        jwt_match = re.search(r"FERO_TOKEN=([^\n]+)\n", self.fero_conf_contents)
        return jwt_match.group(1) if jwt_match else None

    def _get_fero_conf_contents(self) -> str:
        """Load the content of a .fero file in the user's home directory."""
        home = Path.home()
        fero_conf = home / FERO_CONF_FILE
        if fero_conf.is_file():
            with open(str(fero_conf)) as conf_file:
                self._fero_conf_content = conf_file.read()
        else:
            self._fero_conf_content = ""

    def _get_token_as_user(self) -> Union[str, None]:
        """Get the token with the username and password."""
        req = requests.post(
            f"{self._hostname}/api/token/auth/",
            json={"username": self._username, "password": self._password},
            verify=self._verify,
        )

        return req.json().get("token", None)

    def _jwt_from_local_user_pass(self) -> Union[str, None]:
        """Check the local system for username and password and queries fero for a JWT."""
        if "FERO_USERNAME" in os.environ and "FERO_PASSWORD" in os.environ:
            self._username = os.environ.get("FERO_USERNAME")
            self._password = os.environ.get("FERO_PASSWORD")

        if self._username is None or self._password is None:
            username_match = re.search(
                r"FERO_USERNAME=([^\n]+)\n", self.fero_conf_contents
            )
            password_match = re.search(
                r"FERO_PASSWORD=([^\n]+)\n", self.fero_conf_contents
            )

            if username_match and password_match:
                self._username = username_match.group(1)
                self._password = password_match.group(1)

        if self._username and self._password:
            return self._get_token_as_user()

        return None

    @staticmethod
    def _handle_response(
        response: requests.Response, allow_404: bool = False
    ) -> Optional[Union[dict, bytes]]:
        """Check and decode a request response and raise a relevant error if needed."""
        if 200 <= response.status_code < 300:
            if response.headers.get("content-type") == "application/json":
                return response.json()
            else:
                return response.content
        elif response.status_code == 404:
            if allow_404:
                return None
            else:
                raise FeroError("The requested resource was not found.")
        elif response.status_code in [401, 403]:
            raise FeroError("You are not authorized to access this resource.")
        elif response.headers.get("content-type") == "application/json":
            raise FeroError(
                f"There was an issue connecting to Fero: {response.json()}. - Status Code: {response.status_code}"
            )
        else:
            raise FeroError(
                f"There was an issue connecting to Fero: {response.reason}. - Status Code: {response.status_code}"
            )

    def _s3_upload(self, inbox_response, file_name, fp) -> None:
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
            verify=self._verify,
        )

        if res.status_code != 204:
            raise FeroError("Error Uploading File")

    @staticmethod
    def _azure_upload(inbox_response: dict, fp) -> None:
        blob_client = BlobClient.from_blob_url(
            f"https://{inbox_response['storage_name']}.blob.core.windows.net/{inbox_response['container']}/{inbox_response['blob']}?{inbox_response['sas_token']}"
        )
        blob_client.upload_blob(io.BytesIO(fp.read().encode()))

    def _paginated_get(
        self, url: str, object_class: Type[FeroObject], params: Dict[str, str]
    ) -> Iterator[FeroObject]:
        next_url = url

        while next_url is not None:
            parsed_url = urlparse(next_url)

            # we use the path here in case the server has custom ports behind a load balancer or
            # proxy
            next_url = parsed_url.path
            if params is None and parsed_url.query != "":
                next_url += f"?{parsed_url.query}"
            data = self.get(next_url, params=params)
            for result in data.get("results", []):
                yield object_class(self, result)

            next_url = data.get("next", None)
            # further params should come from the next_url
            params = None

    def upload_file(self, inbox_response, file_name, file_pointer):
        """Upload a file to object store specified by Fero inbox response."""
        if inbox_response["upload_type"] == "azure":
            self._azure_upload(inbox_response, file_pointer)
        else:
            self._s3_upload(inbox_response, file_name, file_pointer)

    def search_analyses(self, name: str = None) -> Iterator[Analysis]:
        """Search available analyses by name and return an iterator of matching objects.

        :param name: Name of analysis to filter by.
        :type name: str, optional
        :return: a list of analyses
        :rtype: Iterator[Analysis]
        """
        params = {}
        if name is not None:
            params["name"] = name
        return self._paginated_get("/api/analyses/", Analysis, params=params)

    def get_analysis(self, uuid: str) -> Analysis:
        """Get a Fero Analysis using the UUID.

        :param uuid: UUID of the analysis
        :type uuid: str
        :return: An Analysis object
        :rtype: Analysis
        """
        analysis_data = self.get(f"/api/analyses/{uuid}/")
        return Analysis(self, analysis_data)

    def search_assets(self, name: str = None) -> Iterator[Asset]:
        """Search available assets by name and return an iterator of matching objects.

        :param name: Name of asset to filter by.
        :type name: str, optional
        :return: a list of assets
        :rtype: Iterator[Asset]
        """
        params = {}
        if name is not None:
            params["name"] = name
        return self._paginated_get("/api/assets/", Asset, params=params)

    def get_asset(self, uuid: str) -> Asset:
        """Get a Fero Asset using the UUID.

        :param uuid: UUID of the asset
        :type uuid: str
        :return: An Asset object
        :rtype: Asset
        """
        asset_data = self.get(f"/api/assets/{uuid}/")
        return Asset(self, asset_data)

    def search_processes(self, name: str = None) -> Iterator[Process]:
        """Search available processes by name and return an iterator of matching objects.

        :param name: Name of analysis to filter by.
        :type name: str, optional
        :return: a list of processes
        :rtype: Iterator[Process]
        """
        params = {}
        if name is not None:
            params["name"] = name
        return self._paginated_get("/api/processes/", Process, params=params)

    def get_workspace(self, uuid: str) -> Workspace:
        """Get a Fero Workspace using the UUID.

        :param uuid: UUID of the workspace
        :type uuid: str
        :return: An Workspace object
        :rtype: Workspace
        """
        workspace_data = self.get(f"/api/workspaces/{uuid}/")
        return Workspace(self, workspace_data)

    def get_process(self, uuid: str) -> Process:
        """Get a Fero Process using the UUID.

        :param uuid: UUID of the process
        :type uuid: str
        :return: An Process object
        :rtype: Analysis
        """
        process_data = self.get(f"/api/processes/{uuid}/")
        return Process(self, process_data)

    def get_datasource(self, uuid: str) -> DataSource:
        """Get a Fero Data Source by UUID.

        :param uuid: UUID of requested object
        :type uuid: str
        :return: A data source object
        :rtype: DataSource
        """
        source_data = self.get(f"/api/v2/data_source/{uuid}/")
        return DataSource(self, source_data)

    def post(self, url: str, data: dict) -> Union[dict, bytes]:
        """Do a POST request with headers set."""
        return self._handle_response(
            requests.post(
                f"{self._hostname}{url}",
                json=data,
                headers={"Authorization": f"JWT {self._fero_token}"},
                verify=self._verify,
            ),
            allow_404=False,
        )

    def patch(self, url: str, data: dict) -> Union[dict, bytes]:
        """Do a PATCH request with headers set."""
        return self._handle_response(
            requests.patch(
                f"{self._hostname}{url}",
                json=data,
                headers={"Authorization": f"JWT {self._fero_token}"},
                verify=self._verify,
            ),
            allow_404=False,
        )

    def get(self, url: str, params=None, allow_404=False) -> Union[dict, bytes]:
        """Do a GET request with headers set."""
        return self._handle_response(
            requests.get(
                f"{self._hostname}{url}",
                params=params,
                headers={"Authorization": f"JWT {self._fero_token}"},
                verify=self._verify,
            ),
            allow_404=allow_404,
        )

    def get_preauthenticated(self, url, params=None) -> Union[dict, bytes]:
        """Do a GET request without adjusting the url or auth headers."""
        return self._handle_response(
            requests.get(
                url,
                params=params,
                verify=self._verify,
            )
        )

    def impersonate(self, username: str):
        """Impersonate a user.  Only available to admin users.

        :param username: The username to impersonate
        :type user_id: str
        :return: None
        """
        impersonated_token = self.post(
            "/api/token/impersonate/", {"user": username}
        ).get("token")
        self._admin_token = self._fero_token
        self._fero_token = impersonated_token
        self._impersonated = username

    def end_impersonation(self):
        """Disable client impersonation."""
        self._fero_token = self._admin_token
        self._admin_token = None
        self._impersonated = None


class UnsafeFeroForScripting(Fero):
    """SCRIPT USE ONLY: An unsafe client for scripting purposes."""

    @backoff.on_predicate(
        backoff.fibo, predicate=lambda ds: ds.status != "R", max_time=600
    )
    def _wait_until_datasource_is_ready(self, ds: DataSource) -> DataSource:
        if ds.status == "R":
            return ds
        return DataSource(self, self.get(f"/api/v2/data_source/{str(ds.uuid)}/"))

    def _upload_config(
        self,
        format_type=None,
        file_type=None,
        primary_datetime=None,
        primary_keys=None,
    ):
        # TODO: Get from an endpoint
        data = {
            "upload_format_configuration": {
                "format_type": format_type or "tabular",
                "file_options": {"kind": file_type or "CsvFileOptions"},
            }
        }

        if primary_datetime is not None:
            data["primary_datetime_col"] = primary_datetime
        if primary_keys is not None:
            data["primary_key_col"] = primary_keys

        return data

    def create_live_datasource(
        self,
        ds_name,
        ds_schema,
        format_type=None,
        file_type=None,
        primary_datetime=None,
        primary_keys=None,
    ):
        """SCRIPT USE ONLY: Create a live data source."""
        me = self.get("/api/me/")
        upload_ac = me["profile"]["default_upload_ac"]["name"]

        data = {
            "name": ds_name,
            "description": "",
            "access_control": upload_ac,
            "live_configuration": ds_schema,
            "default_upload_config": self._upload_config(
                file_type=file_type,
                format_type=format_type,
                primary_datetime=primary_datetime,
                primary_keys=primary_keys,
            ),
        }

        if primary_datetime is not None:
            data["primary_datetime_column"] = primary_datetime

        if primary_keys is not None:
            data["primary_keys"] = primary_keys

        ds = DataSource(
            self,
            self.post("/api/v2/data_source/", data),
        )
        return self._wait_until_datasource_is_ready(ds)

    def create_datasource_from_file(
        self,
        ds_name: str,
        file_name: str,
        file_schema: Dict[str, Any],
        file: TextIO,
        file_type: Optional[str] = "CsvFileOptions",
        format_type: Optional[str] = "tabular",
        overwrites: Optional[Dict[str, Any]] = None,
        primary_datetime: Optional[str] = None,
        primary_keys: Optional[Sequence[str]] = None,
    ) -> DataSource:
        """SCRIPT USE ONLY: Create a data source from a CSV file."""
        data = {
            "name": file_name,
            "uploaded_data_configuration": self._upload_config(
                file_type=file_type,
                format_type=format_type,
                primary_datetime=primary_datetime,
                primary_keys=primary_keys,
            ),
            "parsed_schema": file_schema,
        }

        uf_res = self.post(
            "/api/v2/uploaded_files/",
            data,
        )

        files_uuid = uf_res["uuid"]
        inbox_response = self.get(
            f"/api/v2/uploaded_files/{files_uuid}/inbox_url/?file_name={file_name}"
        )
        self.upload_file(inbox_response, file_name, file)

        me = self.get("/api/me/")
        upload_ac = me["profile"]["default_upload_ac"]["name"]

        ds = DataSource(
            self,
            self.post(
                "/api/v2/data_source/",
                {
                    "name": ds_name,
                    "access_control": upload_ac,
                    "uploaded_files_uuid": files_uuid,
                },
            ),
        )

        ds = self._wait_until_datasource_is_ready(ds)

        if overwrites is not None:
            ds = DataSource(
                self,
                self.post(
                    f"/api/v2/data_source/{str(ds.uuid)}/overwrite_schema/", overwrites
                ),
            )
            ds = self._wait_until_datasource_is_ready(ds)

        return ds

    def create_process_from_json_string(
        self, process_name, process_json_str
    ) -> Process:
        """SCRIPT USE ONLY: Create a process from a JSON string."""
        # can't pass a file through without modifying the client post
        api_id = self._handle_response(
            requests.post(
                f"{self._hostname}/api/process_upload/from_json/",
                data={"process_name": process_name},
                files={"process_revision_json": io.StringIO(process_json_str)},
                headers={"Authorization": f"JWT {self._fero_token}"},
                verify=False,
            ),
            allow_404=False,
        ).get("process_id")
        return self.get_process(api_id)

    def create_analysis(self, analysis_data):
        """SCRIPT USE ONLY: Create an analysis from a JSON string."""
        return Analysis(self, self.post("/api/analyses/", analysis_data))

    def create_workspace(self, name: str, description: str = ""):
        """SCRIPT USE ONLY: Create a workspace."""
        response = self.post(
            "/api/workspaces/", {"name": name, "description": description}
        )
        return Workspace(self, {**response, "description": description})

    def add_objects_to_workspace(
        self,
        workspace: Workspace,
        objects: list[Union[DataSource, Process, Analysis]],
    ) -> Workspace:
        """SCRIPT USE ONLY: Add objects to a workspace."""
        datasources = [str(obj.uuid) for obj in objects if isinstance(obj, DataSource)]
        processes = [str(obj.api_id) for obj in objects if isinstance(obj, Process)]
        analyses = [str(obj.uuid) for obj in objects if isinstance(obj, Analysis)]

        def _add_objects(object_name, object_uuids):
            data = self.post(
                f"/api/workspaces/{workspace.uuid}/update_objects/",
                {
                    "uuids": object_uuids,
                    "object_name": object_name,
                    "remove": False,
                    "add_dependencies": False,
                },
            )
            return data

        data = None
        if datasources:
            data = _add_objects("DataSourceV2", datasources)
        if processes:
            data = _add_objects("Process", processes)
        if analyses:
            data = _add_objects("Analysis", analyses)
        return Workspace(self, data)

    def set_hidden(self, object_name: str, uuid: str, hidden: bool):
        """SCRIPT USE ONLY: Set the hidden status of an object."""
        return self.patch(
            "/api/hide_objects/",
            {"hidden": hidden, "object_name": object_name, "uuid": uuid},
        )
