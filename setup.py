"""File to define installation of the `fero` package."""

import os
import sys


from setuptools import setup
from setuptools.command.install import install
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")
VERSION = "2.2.13"


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version."""

    description = "verify that the git tag matches our version"

    def run(self):
        """Verify that the git tag matches our version with this custom command."""
        tag = os.getenv("CIRCLE_TAG") or os.getenv("GITHUB_REF_NAME")
        if not tag:
            sys.exit(
                "Set CIRCLE_TAG or GITHUB_REF_NAME to the release tag (e.g. v1.2.3) to verify."
            )
        if tag.lstrip("v") != VERSION:
            info = f"Git tag: {tag} does not match the version of this app: {VERSION}"
            sys.exit(info)


setup(
    name="fero",
    version=VERSION,
    description="Python client for accessing Fero API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FeroLabs/fero_client",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
    ],
    packages=["fero"],
    python_requires=">=3.9, <4",
    install_requires=[
        "requests",
        "pandas>=2.2.0,<4",
        "marshmallow>=4.3,<5",
        "azure-storage-blob",
        "pyarrow>=24.0.0,<24.1.0",
    ],
    project_urls={
        "Bug Reports": "https://github.com/pypa/sampleproject/issues",
        "Source": "https://github.com/pypa/sampleproject/",
    },
    cmdclass={
        "verify": VerifyVersionCommand,
    },
)
