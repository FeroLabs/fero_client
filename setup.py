import os
import sys


from setuptools import setup
from setuptools.command.install import install
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")
VERSION = "1.4.1"


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""

    description = "verify that the git tag matches our version"

    def run(self):
        tag = os.getenv("CIRCLE_TAG")

        if tag.lstrip("v") != VERSION:
            info = f"Git tag: {tag} does not match the version of this app: {VERSION}".format(
                tag, VERSION
            )
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
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    packages=["fero"],
    python_requires=">=3.5, <4",
    install_requires=["requests", "pandas", "marshmallow", "azure-storage-blob"],
    project_urls={
        "Bug Reports": "https://github.com/pypa/sampleproject/issues",
        "Source": "https://github.com/pypa/sampleproject/",
    },
    cmdclass={
        "verify": VerifyVersionCommand,
    },
)
