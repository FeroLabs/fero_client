"""This module holds the `Workspace` class and its schema."""

from marshmallow import (
    Schema,
    fields,
    EXCLUDE,
)
from .common import FeroObject


class WorkspaceSchema(Schema):
    """A schema for a workspace."""

    class Meta:
        """
        Specify that unknown fields included on this schema should be excluded.

        See https://marshmallow.readthedocs.io/en/stable/quickstart.html#handling-unknown-fields.
        """

        unknown = EXCLUDE

    uuid = fields.UUID(required=True)
    name = fields.Str(required=True)
    description = fields.Str(default="", allow_none=True)
    modified = fields.DateTime(required=True)
    created_by = fields.Dict(required=True)
    analyses = fields.List(fields.Dict(), default=list)
    processes = fields.List(fields.Dict(), default=list)
    datasources = fields.List(fields.Dict(), default=list)


class Workspace(FeroObject):
    """An object representing a workspace."""

    schema_class = WorkspaceSchema

    def __repr__(self):
        """Represent the `Workspace` object by its name."""
        return f"<Workspace name={self.name}>"

    __str__ = __repr__
