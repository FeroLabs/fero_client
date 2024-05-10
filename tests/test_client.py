"""A module to test the `Fero` class."""

from fero.datasource import DataSource
from fero.analysis import Analysis
from fero.asset import Asset
from fero.process import Process
import pytest
from unittest import mock
from pathlib import Path
from fero import Fero, FeroError


@pytest.fixture
def mock_response():
    """Create a mocked successful API response."""
    response = mock.MagicMock()
    response.json.return_value = {"token": "fakeToken"}
    response.content
    response.headers = {"content-type": "application/json"}
    response.status_code = 200

    return response


@pytest.fixture()
def patch_requests_post(mock_response):
    """Create a mocked successful POST response."""
    with mock.patch("fero.client.requests.post") as post_function:
        post_function.return_value = mock_response
        yield post_function


@pytest.fixture
def patch_requests_get(mock_response):
    """Create a mocked successful GET response."""
    with mock.patch("fero.client.requests.get") as get_function:
        get_function.return_value = mock_response
        yield get_function


@pytest.fixture
def mock_conf_path(monkeypatch, tmp_path_factory):
    """Create a mocked successful POST response."""
    path = tmp_path_factory.mktemp("home-")
    monkeypatch.setattr(Path, "home", mock.MagicMock(return_value=path))

    yield path


@pytest.fixture
def patch_fero_get():
    """Create a mocked GET call from the fero client."""
    with mock.patch.object(Fero, "get") as mock_fero_get:
        yield mock_fero_get


def test_fero_client_with_jwt():
    """Test that a client is created with a jwt provided."""
    client = Fero(fero_token="fakeToken")
    assert isinstance(client, Fero)
    assert client._fero_token == "fakeToken"


def test_fero_client_username_pass_provided(patch_requests_post):
    """Test that a client is created correctly with a provided username and password."""
    client = Fero(username="fero", password="pass")
    assert isinstance(client, Fero)

    patch_requests_post.assert_called_with(
        "https://app.ferolabs.com/api/token/auth/",
        json={"username": "fero", "password": "pass"},
        verify=True,
    )
    assert client._fero_token == "fakeToken"


def test_fero_client_env_token(monkeypatch):
    """Test that a token is found in the env."""
    monkeypatch.setenv("FERO_TOKEN", "fakeToken")

    client = Fero()
    assert isinstance(client, Fero)
    assert client._fero_token == "fakeToken"


def test_fero_client_token_file(mock_conf_path):
    """Test that a token is found in the .fero file."""
    with open(str(mock_conf_path / ".fero"), "w") as fero_file:
        fero_file.write("FERO_TOKEN=fakeToken\n")

    client = Fero()
    assert isinstance(client, Fero)
    assert client._fero_token == "fakeToken"


def test_fero_client_user_pass_file(patch_requests_post, mock_conf_path):
    """Test that username and password are found in the .fero file."""
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
    """Test that an exception is raised if there is no way to get a token."""
    with pytest.raises(FeroError):
        Fero()


def test_fero_client_get_wrapper(patch_requests_get):
    """Test that get requests have the correct headers added."""
    client = Fero(fero_token="fakeToken", hostname="http://test.com")
    client.get("/some/url/", params={"n": "42"})
    patch_requests_get.assert_called_with(
        "http://test.com/some/url/",
        params={"n": "42"},
        headers={"Authorization": "JWT fakeToken"},
        verify=True,
    )


def test_fero_client_post_wrapper(patch_requests_post):
    """Test that post requests have the correct headers added."""
    client = Fero(fero_token="fakeToken", hostname="http://test.com")
    client.post("/some/url/", {"n": "42"})
    patch_requests_post.assert_called_with(
        "http://test.com/some/url/",
        json={"n": "42"},
        headers={"Authorization": "JWT fakeToken"},
        verify=True,
    )


def test_fero_get_raises_error_404(patch_requests_get, mock_response):
    """Test that a FeroError with the expected message is raised if the url isn't found."""
    mock_response.status_code = 404
    patch_requests_get.return_value = mock_response
    client = Fero(fero_token="fakeToken", hostname="http://test.com")

    with pytest.raises(FeroError) as err:
        client.get("/some/url/")
        assert err.message == "The requested resource was not found"


def test_fero_get_raises_error_not_authorized(patch_requests_get, mock_response):
    """Test that a FeroError with the expected message is raised if the request isn't authorized."""
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
    """Test that a `FeroError` with the expected message is raised if a 400 is returned."""
    mock_response.status_code = 400
    patch_requests_get.return_value = mock_response
    client = Fero(fero_token="fakeToken", hostname="http://test.com")

    with pytest.raises(FeroError) as err:
        client.get("/some/url/")
        assert (
            err.message == "There was an issue connecting to Fero. - Status Code: 400"
        )


def test_get_analysis_success(patch_fero_get, analysis_data):
    """Test that an analysis is returned by get_analysis."""
    patch_fero_get.return_value = analysis_data
    client = Fero(fero_token="fakeToken", hostname="http://test.com")
    analysis = client.get_analysis("some-uuid")
    assert isinstance(analysis, Analysis)
    assert analysis.name == analysis_data["name"]
    patch_fero_get.assert_called_with("/api/analyses/some-uuid/")


def test_search_analyses(patch_fero_get, analysis_data):
    """Test that the correct iterator of analyses is returned by `search_analyses`."""
    patch_fero_get.return_value = {"next": None, "results": [analysis_data]}
    client = Fero(fero_token="fakeToken", hostname="http://test.com")
    analyses = [a for a in client.search_analyses(analysis_data["name"])]
    assert len(analyses) == 1
    assert isinstance(analyses[0], Analysis)
    assert analyses[0].name == analysis_data["name"]
    patch_fero_get.assert_called_with(
        "/api/analyses/", params={"name": analysis_data["name"]}
    )


def test_search_analyses_paginated(patch_fero_get, analysis_data):
    """Test that a list analyses is returned by `search_analyses`."""
    patch_fero_get.side_effect = [
        {
            "next": "http://test.com/api/analyses/?token=1234",
            "results": [analysis_data],
        },
        {"next": None, "results": [analysis_data]},
    ]
    client = Fero(fero_token="fakeToken", hostname="http://test.com")
    analyses = [a for a in client.search_analyses(analysis_data["name"])]
    assert len(analyses) == 2
    assert isinstance(analyses[0], Analysis)
    assert analyses[0].name == analysis_data["name"]
    assert isinstance(analyses[1], Analysis)
    assert analyses[1].name == analysis_data["name"]
    patch_fero_get.assert_has_calls(
        [
            mock.call("/api/analyses/", params={"name": analysis_data["name"]}),
            mock.call("/api/analyses/?token=1234", params=None),
        ]
    )


def test_get_asset_success(patch_fero_get, asset_data):
    """Test that an asset is returned by `get_asset`."""
    patch_fero_get.return_value = asset_data
    client = Fero(fero_token="fakeToken", hostname="http://test.com")
    asset = client.get_asset("some-uuid")
    assert isinstance(asset, Asset)
    assert asset.name == asset_data["name"]
    patch_fero_get.assert_called_with("/api/assets/some-uuid/")


def test_search_assets(patch_fero_get, asset_data):
    """Test that an iterator of assets is returned by `search_assets`."""
    patch_fero_get.return_value = {"next": None, "results": [asset_data]}
    client = Fero(fero_token="fakeToken", hostname="http://test.com")
    assets = [a for a in client.search_assets(asset_data["name"])]
    assert len(assets) == 1
    assert isinstance(assets[0], Asset)
    assert assets[0].name == asset_data["name"]
    patch_fero_get.assert_called_with(
        "/api/assets/", params={"name": asset_data["name"]}
    )


def test_search_assets_paginated(patch_fero_get, asset_data):
    """Test that an iterator of assets is returned by `search_assets`."""
    patch_fero_get.side_effect = [
        {
            "next": "http://test.com/api/assets/?token=1234",
            "results": [asset_data],
        },
        {"next": None, "results": [asset_data]},
    ]
    client = Fero(fero_token="fakeToken", hostname="http://test.com")
    assets = [a for a in client.search_assets(asset_data["name"])]
    assert len(assets) == 2
    assert isinstance(assets[0], Asset)
    assert assets[0].name == asset_data["name"]
    assert isinstance(assets[1], Asset)
    assert assets[1].name == asset_data["name"]
    patch_fero_get.assert_has_calls(
        [
            mock.call("/api/assets/", params={"name": asset_data["name"]}),
            mock.call("/api/assets/?token=1234", params=None),
        ]
    )


def test_get_datasource_success(patch_fero_get, datasource_data):
    """Test that an data source is returned by `get_datasource`."""
    patch_fero_get.return_value = datasource_data
    client = Fero(fero_token="fakeToken", hostname="http://test.com")
    ds = client.get_datasource("9f79206e-94fc-4834-8f52-84008b12df86")
    assert isinstance(ds, DataSource)
    assert ds.name == datasource_data["name"]
    patch_fero_get.assert_called_with(
        "/api/v2/data_source/9f79206e-94fc-4834-8f52-84008b12df86/"
    )


def test_get_process_success(patch_fero_get, process_data):
    """Test that an process is returned by `get_process`."""
    patch_fero_get.return_value = process_data
    client = Fero(fero_token="fakeToken", hostname="http://test.com")
    process = client.get_process("some-uuid")
    assert isinstance(process, Process)
    assert process.name == process_data["name"]
    patch_fero_get.assert_called_with("/api/processes/some-uuid/")


def test_search_processes(patch_fero_get, process_data):
    """Test that the correct iterator of processes is returned by `search_processes`."""
    patch_fero_get.return_value = {"next": None, "results": [process_data]}
    client = Fero(fero_token="fakeToken", hostname="http://test.com")
    processes = [a for a in client.search_processes(process_data["name"])]
    assert len(processes) == 1
    assert isinstance(processes[0], Process)
    assert processes[0].name == process_data["name"]
    patch_fero_get.assert_called_with(
        "/api/processes/", params={"name": process_data["name"]}
    )


def test_search_processes_paginated(patch_fero_get, process_data):
    """Test that a list process is returned by `search_analyses`."""
    patch_fero_get.side_effect = [
        {
            "next": "http://test.com/api/processes/?token=1234",
            "results": [process_data],
        },
        {"next": None, "results": [process_data]},
    ]
    client = Fero(fero_token="fakeToken", hostname="http://test.com")
    processes = [a for a in client.search_processes(process_data["name"])]
    assert len(processes) == 2
    assert isinstance(processes[0], Process)
    assert processes[0].name == process_data["name"]
    assert isinstance(processes[1], Process)
    assert processes[1].name == process_data["name"]
    patch_fero_get.assert_has_calls(
        [
            mock.call("/api/processes/", params={"name": process_data["name"]}),
            mock.call("/api/processes/?token=1234", params=None),
        ]
    )
