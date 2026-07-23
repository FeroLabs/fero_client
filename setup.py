"""File to define installation of the `fero` package."""

from setuptools import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")
VERSION = "2.3.0"


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
        "requests>=2.33.0",
        "pandas>=2.2.0,<4",
        "marshmallow>=3.13,<5",
        "azure-storage-blob>=12.25.1",
        "pyarrow>=19",
        "backoff>=2.2.1",
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
            "flake8-docstrings",
            "flake8-rst-docstrings",
            "black",
            "twine",
            "numpy>=1.26,<3",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/pypa/sampleproject/issues",
        "Source": "https://github.com/pypa/sampleproject/",
    },
)
