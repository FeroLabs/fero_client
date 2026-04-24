"""This script copies a demo workspace to a user."""

from argparse import ArgumentParser
from fero.client import UnsafeFeroForScripting
from scripts.common import copy_demo_workspace

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="copy_demo_workspace",
        description="Copy a demo workspace to a user.",
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
        "--hostname",
        default="https://app.ferolabs.com",
        help="Hostname of the Fero instance. Defaults to https://app.ferolabs.com.",
    )
    parser.add_argument(
        "--source-username",
        required=True,
        help="Username of the source Fero instance.",
    )

    args = parser.parse_args()
    source_client = UnsafeFeroForScripting(
        hostname=args.hostname,
    )
    copy_demo_workspace(
        source_client=source_client,
        source_user=args.source_username,
        workspace_uuid=args.workspace_uuid,
        target_user=args.target_username,
    )
