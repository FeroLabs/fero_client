"""This module defines classes and functions used in multiple places throughout the fero client."""

import fero
import time
import requests
from typing import Any, Callable, TypeVar
from marshmallow import Schema

from fero.exceptions import FeroError


class FeroObject:
    """
    A base class for fero-related objects.

    Data related to the object, formatted as a `schema_class`,
    is stored in the `_data` attribute and accessible through a normal
    `__getattr__` call (like a dictionary).
    """

    schema_class: Schema = None

    def __init__(self, client: "fero.Fero", data: dict):
        """
        Create a new `FeroObject` using provided `data`.

        :param client: The fero client object that gives access to this object
        :param data: A dictionary holding values for the fields of a `schema_class`
        """
        self._client = client
        schema = self.schema_class()
        self._data = schema.load(data)

    def __getattr__(self, name: str) -> Any:
        """
        Access an entry in this object's schema.

        :param name: the entry in the schema to access
        :return: the data stored in this entry of the schema
        """
        return self._data.get(name)


class FeroTimeoutError(FeroError):
    """An error raised when an api request times out."""

    def __init__(self, last_response: requests.Response):
        """
        Create a `FeroTimeoutError` citing the most recent response `last_response`.

        :param last_response: the last response from the api before timing out
        """
        self.last_response = last_response
        super().__init__("Request exceeded specified polling timeout")


D = TypeVar("D")


def poll(
    func: Callable[[], D], test: Callable[[D], bool], interval=0.5, timeout=300
) -> D:
    """Poll the api until a test condition is met.

    :param func: function to do the polling request
    :param test: Function to test whether the polling should end
    :param interval: interval to poll at, defaults to 0.5
    :param timeout: timeout value to stop polling, defaults to 300
    :raises FeroTimeoutError: raised after `timeout` seconds if no value passes `test`
    :return: the object to be returned from the function
    """
    data = func()
    start_time = time.time()
    while not test(data):
        if time.time() - start_time > timeout:
            raise FeroTimeoutError(data)
        time.sleep(interval)
        data = func()

    return data
