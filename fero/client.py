import os
import re
import requests
from pathlib import Path
from typing import Optional, Union
from . import FeroError

FERO_CONF_FILE = ".fero"


class Fero:
    def __init__(
        self,
        hostname: Optional[str] = "https://app.ferolabs.com",
        username: Optional[str] = None,
        password: Optional[str] = None,
        fero_token: Optional[str] = None,
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
        :param fero_token: , defaults to None
        :type fero_token: Optional[str], optional
        :raises FeroError: Raised if a token cannot be obtained
        """
        self._fero_token = None
        self._fero_conf_content = None
        self._password = None
        self._username = None

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
