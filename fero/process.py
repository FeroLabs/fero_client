from .common import FeroObject
from marshmallow import (
    Schema,
    fields,
    validate,
    EXCLUDE,
)

"""
class ProcessSerializer(FeroModelSerializer):

    latest_revision_version = serializers.IntegerField(read_only=True)

    product_type = serializers.CharField(read_only=True)

    username = serializers.CharField(read_only=True)

    process_type = serializers.CharField()
    kind = serializers.ReadOnlyField(default="process")

    latest_ready_snapshot = NestedSnapshotSerializer(read_only=True)

    data_config = SchemaSerializer(json_class=ProcessDataConfiguration, allow_null=False, read_only=True)
    primary_datetime_column = serializers.CharField(read_only=True, allow_null=True)
    shutdown_configuration = SchemaSerializer(json_class=ShutdownDefinition, allow_null=True, read_only=True)

    class Meta:
        model = Process
        fields = (
            "api_id",
            "name",
            "created",
            "modified",
            "latest_revision_version",
            "username",
            "process_type",
            "product_type",
            "kind",
            "latest_ready_snapshot",
            "data_config",
            "shutdown_configuration",
            "primary_datetime_column",
        )
        read_only_fields = (
            "api_id",
            "created",
            "modified",
            "latest_revision_version",
            "username",
            "product_type",
            "latest_ready_snapshot",
            "data_config",
            "shutdown_configuration",
            "primary_datetime_column",
        )
        write_on_create_only_fields = ("process_type",)


"""

# These do actually map to real names


class FeroProcessTypes:
    ADVANCED = "A"
    BATCH = "B"
    CONTINUOUS = "C"

    @classmethod
    def validator(cls):
        return validate.OneOf([cls.ADVANCED, cls.BATCH, cls.CONTINUOUS])


class SnapShotStatus:
    READY = "R"
    INITIALIZED = "I"
    PROCESSING = "P"
    ERROR = "E"

    @classmethod
    def validator(cls):
        return validate.OneOf([cls.READY, cls.INITIALIZED, cls.PROCESSING, cls.ERROR])


class NestedSnapshotSchema(Schema):

    uuid = fields.String(required=True)
    status = fields.String(
        required=True,
        validate=SnapShotStatus.validator(),
    )


class ProcessSchema(Schema):
    class Meta:
        unknown = EXCLUDE

    api_id = fields.String(required=True)
    name = fields.String(required=True)
    created = fields.DateTime(require=True)
    modified = fields.DateTime(require=True)
    latest_revision_version = fields.Integer(required=True)
    username = fields.String(required=True)
    process_type = fields.String(required=True, validate=FeroProcessTypes.validator())
    product_type = fields.String(Required=False, allow_none=True)
    kind = fields.String(required=True, validate=validate.OneOf(["process"]))
    latest_ready_snapshot = fields.Nested(NestedSnapshotSchema, allow_none=True)
    # TODO update this to a nested schema once the configuration settles more
    data_config = fields.Dict(required=True)


class Process(FeroObject):
    """High level object for interacting with a Process object defined in Fero.

    A Process is a set of configurations that describe an industrial process and allow
    Fero to combine, transform, and enrich raw data sources before the final data is
    used in an Analysis.
    """

    schema_class = ProcessSchema

    def __repr__(self):
        return f"<Process name={self.name}>"

    __str__ = __repr__
