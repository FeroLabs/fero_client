from .exceptions import FeroError  # noqa
from .client import Fero  # noqa
from marshmallow import Schema, validates_schema, ValidationError


class FeroUnionSchema(Schema):

    unions = []

    @validates_schema
    def validate_all_unions(self, data, *kwarg):

        for union in self.unions:

            errors = union().validate(data)
            if not errors:
                return

        raise ValidationError("unable to match a union schema")
