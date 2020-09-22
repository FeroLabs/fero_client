import pytest
from unittest import mock
from pathlib import Path
from fero import Fero, FeroError


@pytest.fixture()
def patch_requests_post():

    with mock.patch("fero.client.requests.post") as post_function:
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"token": "fakeToken"}

        post_function.return_value = mock_response
        yield post_function


@pytest.fixture()
def mock_conf_path(monkeypatch, tmp_path_factory):

    path = tmp_path_factory.mktemp("home-")
    monkeypatch.setattr(Path, "home", mock.MagicMock(return_value=path))

    yield path


def test_fero_client_with_jwt():
    """Test that a client is created with a jwt provided"""

    client = Fero(fero_token="fakeToken")
    assert isinstance(client, Fero)
    assert client._fero_token == "fakeToken"


def test_fero_client_username_pass_provided(patch_requests_post):
    """Test that a client is created correctly with a provided username and password"""

    client = Fero(username="fero", password="pass")
    assert isinstance(client, Fero)

    patch_requests_post.assert_called_with(
        "https://app.ferolabs.com/api/token/auth/",
        json={"username": "fero", "password": "pass"},
    )
    assert client._fero_token == "fakeToken"


def test_fero_client_env_token(monkeypatch):
    """Test that a token is found in the env"""
    monkeypatch.setenv("FERO_TOKEN", "fakeToken")

    client = Fero()
    assert isinstance(client, Fero)
    assert client._fero_token == "fakeToken"


def test_fero_client_token_file(mock_conf_path):
    """Test that a token is found in the .fero file"""

    with open(str(mock_conf_path / ".fero"), "w") as fero_file:
        fero_file.write("FERO_TOKEN=fakeToken\n")

    client = Fero()
    assert isinstance(client, Fero)
    assert client._fero_token == "fakeToken"


def test_fero_client_user_pass_file(patch_requests_post, mock_conf_path):
    """Test that username and password are found in the .fero file"""
    with open(str(mock_conf_path / ".fero"), "w") as fero_file:
        fero_file.write("FERO_USERNAME=fero\nFERO_PASSWORD=pass\n")

    client = Fero()
    assert isinstance(client, Fero)
    assert client._fero_token == "fakeToken"

    patch_requests_post.assert_called_with(
        "https://app.ferolabs.com/api/token/auth/",
        json={"username": "fero", "password": "pass"},
    )


def test_fero_client_no_creds():
    """Test that an exception is raised if there is no way to get a token"""
    with pytest.raises(FeroError):
        Fero()
