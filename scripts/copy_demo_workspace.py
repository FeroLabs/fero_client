import sys
from argparse import ArgumentParser
from pydantic_settings import BaseSettings
from typing import Optional

from fero.client import UnsafeFeroForScripting
from .common import copy_demo_workspace


class Settings(BaseSettings):
    copy_demo_source_admin_username: Optional[str] = None
    copy_demo_source_admin_password: Optional[str] = None
    copy_demo_source_hostname: str = "https://app.ferolabs.com"
    copy_demo_source_username: str = "demo_corp_admin"


def load_settings(env_file_name):
    settings = Settings(_env_file=env_file_name)
    source_vars = {
        k: v for k, v in dict(settings).items() if k.startswith("copy_demo_source_")
    }
    target_vars = {
        k: v for k, v in dict(settings).items() if k.startswith("copy_demo_target_")
    }
    missing_source_vars = [v for v in source_vars.values() if v is None]
    missing_target_vars = [v for v in target_vars.values() if v is None]
    if missing_source_vars or (missing_target_vars and any(target_vars.values())):
        print(f"Please provide the following environment variables ({env_file_name}):")
        with open(env_file_name, "w") as env_file:
            for field, value in dict(settings).items():
                if field.startswith("copy_demo_target_") and not any(
                    target_vars.values()
                ):
                    continue
                display_name = " ".join(
                    w.capitalize() for w in field.removeprefix("copy_demo_").split("_")
                )
                env_var_value = (
                    input(
                        f"Please enter {display_name} "
                        + (f"({value}) > " if value else "> "),
                    )
                    or value
                )
                env_file.write(f"{field}={env_var_value}\n")
    settings = Settings(_env_file=env_file_name)
    return settings


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="copy_demo_workspace",
        description="Copy a demo workspace to a user.",
    )
    parser.add_argument(
        "--env-file",
        required=True,
        help="Path to *.env file. (i.e., scripts/prod.env)",
    )
    parser.add_argument(
        "--workspace-uuid",
        required=True,
        help="UUID of the workspace to copy.",
    )
    parser.add_argument(
        "--target-username",
        required=True,
        help="User to copy the workspace to.",
    )
    parser.add_argument(
        "--target-admin-username",
        required=False,
        help="Admin username for the target environment. If not provided, the target workspace will be created with the same admin as the source workspace.",
    )
    parser.add_argument(
        "--target-admin-password",
        required="--target-admin-username" in sys.argv,
        help="Admin password for the target environment.",
    )
    parser.add_argument(
        "--target-hostname",
        required="--target-admin-username" in sys.argv,
        help="Hostname of the target environment.",
    )

    args = parser.parse_args()
    target_client = None
    settings = load_settings(args.env_file)
    if args.target_hostname:
        target_client = UnsafeFeroForScripting(
            hostname=args.target_hostname,
            username=args.target_admin_username,
            password=args.target_admin_password,
        )
    source_client = UnsafeFeroForScripting(
        hostname=settings.copy_demo_source_hostname,
        username=settings.copy_demo_source_admin_username,
        password=settings.copy_demo_source_admin_password,
    )
    copy_demo_workspace(
        source_client=source_client,
        source_user=settings.copy_demo_source_username,
        target_user=args.target_username,
        target_client=target_client,
        workspace_uuid=args.workspace_uuid,
    )
