from marshmallow import Schema, fields, validate, EXCLUDE


class CompanySchema(Schema):
    class Meta:
        unknown = EXCLUDE

    name = fields.String()


class AccessControlSchema(Schema):

    name = fields.String()
    readable_name = fields.String()
    uuid = fields.UUID()


class UserAccessSchema(Schema):

    readable_name = fields.String()
    name = fields.String()
    access_type = fields.String(validate=validate.OneOf(["w", "r", "e"]))


class ProfileSchema(Schema):
    class Meta:
        unknown = EXCLUDE

    company = fields.Nested(CompanySchema)
    can_provision_users = fields.Boolean()
    site_acl = fields.List(fields.String())
    write_accesses = fields.List(fields.Nested(UserAccessSchema))
    default_upload_ac = fields.Nested(AccessControlSchema)
    allow_data_v2 = fields.Boolean()


class UserSchema(Schema):
    class Meta:
        unknown = EXCLUDE

    username = fields.String()
    profile = fields.Nested(ProfileSchema)
