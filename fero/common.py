import fero
from marshmallow import Schema


class FeroObject:

    schema_class: Schema = None

    def __init__(self, client: "fero.Fero", data: dict):
        self._client = client
        schema = self.schema_class()
        self._data = schema.load(data)
