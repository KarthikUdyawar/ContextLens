"""
Setup script for the Contextlens package.
"""
from typing import List

from setuptools import find_packages, setup

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """Reads a file containing a list of requirements and returns list.

    Args:
        file_path (str): The path to the file containing the requirements.

    Returns:
        List[str]: A list of requirements read from the file.
    """
    requirements = []
    with open(file_path, encoding="utf-8") as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


def get_long_description(file_path: str = "README.md") -> str:
    """Reads the contents of a file and returns it as a string.

    Args:
        file_path (str, optional): The path to the file.
        Defaults to "README.md".

    Returns:
        str: The contents of the file as a string.
    """
    with open(file_path, encoding="utf-8") as file:
        long_description = file.read()
    return long_description


setup(
    name="Contextlens",
    version="1.0.0",
    author="Karthik Udyawar",
    author_email="karthikajitudy@gmail.com",
    license="Apache License",
    description="Unveiling Hidden Sentiments through Contextual Sentiment Analysis",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/KarthikUdyawar/Contextlens",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
