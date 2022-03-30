"""This Module holds the Fero client class, which is used to communicate with the Fero API."""

from fero.datasource import DataSource
import os
import io
import re
import requests
from urllib.parse import urlparse
from azure.storage.blob import BlobClient
from pathlib import Path
from typing import Dict, Iterator, Optional, Union, Type
from . import FeroError
from .analysis import Analysis
from .asset import Asset
from .process import Process
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
            raise FeroError("You are not authorized to access this resourced.")
        else:
            raise FeroError(
                f"There was an issue connecting to Fero. - Status Code: {response.status_code}"
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

    def search_processes(self, name: str = None) -> Iterator[Analysis]:
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

    def get_process(self, uuid: str) -> Analysis:
        """Get a Fero Process using the UUID.

        :param uuid: UUID of the analysis
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
