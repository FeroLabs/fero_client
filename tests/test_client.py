from fero import Fero


def test_fero_client_with_jwt():
    """Test that a client is created with a jwt provided"""

    client = Fero(fero_token="fakeToken")
    assert isinstance(client, Fero)