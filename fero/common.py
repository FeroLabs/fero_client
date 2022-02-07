import fero
import time
from typing import Callable, TypeVar
from marshmallow import Schema

from fero.exceptions import FeroError


class FeroObject:

    schema_class: Schema = None

    def __init__(self, client: "fero.Fero", data: dict):
        self._client = client
        schema = self.schema_class()
        self._data = schema.load(data)

    def __getattr__(self, name: str):
        return self._data.get(name)


class FeroTimeoutError(FeroError):
    def __init__(self, last_response):
        self.last_response = last_response
        super().__init__("Request exceeded specified polling timeout")


D = TypeVar("D")


def poll(
    func: Callable[[], D], test: Callable[[D], bool], interval=0.5, timeout=300
) -> D:
    """Utility function to poll the api until a test condition is met

    :param func: function to do the polling request
    :type func: Callable[[], D]
    :param test: Function to test whether the polling should end
    :type test: Callable[[D], bool]
    :param interval: interval to poll at, defaults to 0.5
    :type interval: float, optional
    :param timeout: timeout value to stop polling, defaults to 300
    :type timeout: int, optional
    :raises FeroTimeoutError: [description]
    :return: [description]
    :rtype: D
    """
    data = func()
    start_time = time.time()
    while not test(data):
        if time.time() - start_time > timeout:
            raise FeroTimeoutError(data)
        time.sleep(interval)
        data = func()

    return data
