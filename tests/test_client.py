from fero.analysis import Analysis
from fero.asset import Asset
import pytest
from unittest import mock
from pathlib import Path
from fero import Fero, FeroError


@pytest.fixture
def mock_response():

    response = mock.MagicMock()
    response.json.return_value = {"token": "fakeToken"}
    response.content
    response.headers = {"content-type": "application/json"}
    response.status_code = 200

    return response


@pytest.fixture()
def patch_requests_post(mock_response):
    with mock.patch("fero.client.requests.post") as post_function:

        post_function.return_value = mock_response
        yield post_function


@pytest.fixture
def patch_requests_get(mock_response):
    with mock.patch("fero.client.requests.get") as get_function:

        get_function.return_value = mock_response
        yield get_function


@pytest.fixture
def mock_conf_path(monkeypatch, tmp_path_factory):

    path = tmp_path_factory.mktemp("home-")
    monkeypatch.setattr(Path, "home", mock.MagicMock(return_value=path))

    yield path


@pytest.fixture
def patch_fero_get():
    with mock.patch.object(Fero, "get") as mock_fero_get:
        yield mock_fero_get


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
        verify=True,
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
        verify=True,
    )


def test_fero_client_no_creds():
    """Test that an exception is raised if there is no way to get a token"""
    with pytest.raises(FeroError):
        Fero()


def test_fero_client_get_wrapper(patch_requests_get):
    """Test that get requests have the correct headers added"""

    client = Fero(fero_token="fakeToken", hostname="http://test.com")
    client.get("/some/url/", params={"n": "42"})
    patch_requests_get.assert_called_with(
        "http://test.com/some/url/",
        params={"n": "42"},
        headers={"Authorization": "JWT fakeToken"},
        verify=True,
    )


def test_fero_client_post_wrapper(patch_requests_post):
    """Test that post requests have the correct headers added"""

    client = Fero(fero_token="fakeToken", hostname="http://test.com")
    client.post("/some/url/", {"n": "42"})
    patch_requests_post.assert_called_with(
        "http://test.com/some/url/",
        json={"n": "42"},
        headers={"Authorization": "JWT fakeToken"},
        verify=True,
    )


def test_fero_get_raises_error_404(patch_requests_get, mock_response):
    """Test that a FeroError with the expected message is raised if the url isn't found"""
    mock_response.status_code = 404
    patch_requests_get.return_value = mock_response
    client = Fero(fero_token="fakeToken", hostname="http://test.com")

    with pytest.raises(FeroError) as err:
        client.get("/some/url/")
        assert err.message == "The requested resource was not found"


def test_fero_get_raises_error_not_authorized(patch_requests_get, mock_response):
    """Test that a FeroError with the expected message is raised if the request isn't authorized"""
    mock_response.status_code = 403
    patch_requests_get.return_value = mock_response
    client = Fero(fero_token="fakeToken", hostname="http://test.com")

    with pytest.raises(FeroError) as err:
        client.get("/some/url/")
        assert err.message == "The requested resource was not found."

    mock_response.status_code = 401
    with pytest.raises(FeroError) as err:
        client.get("/some/url/")
        assert err.message == "The requested resource was not found."


def test_fero_get_raises_error_400(patch_requests_get, mock_response):
    """Test that a FeroError with the expected message is raised if a 400 is returned"""
    mock_response.status_code = 400
    patch_requests_get.return_value = mock_response
    client = Fero(fero_token="fakeToken", hostname="http://test.com")

    with pytest.raises(FeroError) as err:
        client.get("/some/url/")
        assert (
            err.message == "There was an issue connecting to Fero. - Status Code: 400"
        )


def test_get_analysis_success(patch_fero_get, analysis_data):
    """Test that an analysis is returned by get_analysis"""

    patch_fero_get.return_value = analysis_data
    client = Fero(fero_token="fakeToken", hostname="http://test.com")
    analysis = client.get_analysis("some-uuid")
    assert isinstance(analysis, Analysis)
    assert analysis.name == analysis_data["name"]
    patch_fero_get.assert_called_with("/api/analyses/some-uuid/")


def test_search_analyses(patch_fero_get, analysis_data):
    """Test that a list analyses is returned by search_analyses"""

    patch_fero_get.return_value = {"results": [analysis_data]}
    client = Fero(fero_token="fakeToken", hostname="http://test.com")
    analyses = client.search_analyses(analysis_data["name"])
    assert len(analyses) == 1
    assert isinstance(analyses[0], Analysis)
    assert analyses[0].name == analysis_data["name"]
    patch_fero_get.assert_called_with(
        "/api/analyses/", params={"name": analysis_data["name"]}
    )


def test_get_asset_success(patch_fero_get, asset_data):
    """Test that an asset is returned by get_asset"""

    patch_fero_get.return_value = asset_data
    client = Fero(fero_token="fakeToken", hostname="http://test.com")
    asset = client.get_asset("some-uuid")
    assert isinstance(asset, Asset)
    assert asset.name == asset_data["name"]
    patch_fero_get.assert_called_with("/api/assets/some-uuid/")


def test_search_assets(patch_fero_get, asset_data):
    """Test that a list of assets is returned by search_assets"""

    patch_fero_get.return_value = {"results": [asset_data]}
    client = Fero(fero_token="fakeToken", hostname="http://test.com")
    assets = client.search_assets(asset_data["name"])
    assert len(assets) == 1
    assert isinstance(assets[0], Asset)
    assert assets[0].name == asset_data["name"]
    patch_fero_get.assert_called_with(
        "/api/assets/", params={"name": asset_data["name"]}
    )
