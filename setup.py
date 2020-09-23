from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="fero",
    version="1.0.0",
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
    package_dir={"fero": "fero"},
    packages=find_packages(where="fero"),
    python_requires=">=3.5, <4",
    install_requires=["requests", "pandas", "marshmallow"],
    project_urls={
        "Bug Reports": "https://github.com/pypa/sampleproject/issues",
        "Source": "https://github.com/pypa/sampleproject/",
    },
)
