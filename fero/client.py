import os
import re
import requests
from pathlib import Path
from typing import Optional, Union, List
from . import FeroError
from .analysis import Analysis
from .asset import Asset

FERO_CONF_FILE = ".fero"


class Fero:
    def __init__(
        self,
        hostname: Optional[str] = "https://app.ferolabs.com",
        username: Optional[str] = None,
        password: Optional[str] = None,
        fero_token: Optional[str] = None,
        verify: bool = True,
    ):
        """Creates a base client for communicating with the Fero API.

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
        """Cache the file content"""
        if self._fero_conf_content is None:
            self._get_fero_conf_contents()

        return self._fero_conf_content

    def _get_token_string_from_local(self) -> Union[str, None]:
        """Checks the local system for the JWT token string"""
        if "FERO_TOKEN" in os.environ:
            return os.environ.get("FERO_TOKEN")
        jwt_match = re.search(r"FERO_TOKEN=([^\n]+)\n", self.fero_conf_contents)
        return jwt_match.group(1) if jwt_match else None

    def _get_fero_conf_contents(self) -> str:
        """Loads the content of a .fero file in the user's home directory"""
        home = Path.home()
        fero_conf = home / FERO_CONF_FILE
        if fero_conf.is_file():
            with open(str(fero_conf)) as conf_file:
                self._fero_conf_content = conf_file.read()
        else:
            self._fero_conf_content = ""

    def _get_token_as_user(self) -> Union[str, None]:
        """Gets the token with the username and password"""
        req = requests.post(
            f"{self._hostname}/api/token/auth/",
            json={"username": self._username, "password": self._password},
            verify=self._verify,
        )

        return req.json().get("token", None)

    def _jwt_from_local_user_pass(self) -> Union[str, None]:
        """Checks the local system for username and password and queries fero for a JWT"""
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
        """Check and decode a request response and raise a relevant error if needed"""
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

    def search_analyses(self, name: str = None) -> List[Analysis]:
        """Searches available analyses by name and returns a list of matching objects.

        :param name: Name of analysis to filter by.
        :type name: str, optional
        :return: a list of analyses
        :rtype: List[Analysis]
        """
        params = {}
        if name is not None:
            params["name"] = name
        analysis_data = self.get("/api/analyses/", params=params)
        return [Analysis(self, a) for a in analysis_data["results"]]

    def get_analysis(self, uuid: str) -> Analysis:
        """Gets a Fero Analysis using the UUID.

        :param uuid: UUID of the analysis
        :type uuid: str
        :return: An Analysis object
        :rtype: Analysis
        """
        analysis_data = self.get(f"/api/analyses/{uuid}/")
        return Analysis(self, analysis_data)

    def search_assets(self, name: str = None) -> List[Asset]:
        """Searches available assets by name and returns a list of matching objects.

        :param name: Name of asset to filter by.
        :type name: str, optional
        :return: a list of assets
        :rtype: List[Asset]
        """
        params = {}
        if name is not None:
            params["name"] = name
        asset_data = self.get("/api/assets/", params=params)
        return [Asset(self, a) for a in asset_data["results"]]

    def get_asset(self, uuid: str) -> Asset:
        """Gets a Fero Asset using the UUID.

        :param uuid: UUID of the asset
        :type uuid: str
        :return: An Asset object
        :rtype: Asset
        """
        asset_data = self.get(f"/api/assets/{uuid}/")
        return Asset(self, asset_data)

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
        """Do a GET request without adjusting the url or auth headers"""
        return self._handle_response(
            requests.get(url, params=params, verify=self._verify)
        )
