"""Python setup.py for nano_umap package."""

import os

from setuptools import find_packages, setup  # type: ignore


def read(filepath: str) -> str:
    with open(
        os.path.join(os.path.dirname(__file__), filepath),
        encoding="utf-8",
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path: str) -> list[str]:
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="nano_umap",
    version=read("nano_umap/VERSION"),
    description="LLM Magics for Ipython created by kmkolasinski",
    url="https://github.com/kmkolasinski/nano_umap/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="kmkolasinski",
    packages=find_packages(exclude=["tests", ".github", "notebooks"]),
    install_requires=read_requirements("requirements.txt"),
    extras_require={"test": read_requirements("requirements-test.txt")},
)
