"""Script to check that the base image is consistent between the Dockerfile and
pyproject.toml."""

import re
from typing import Optional


def extract_image_from_dockerfile() -> Optional[str]:
    """Extract the image from the Dockerfile."""
    with open("Dockerfile", "r") as file:
        dockerfile_content = file.read()

    match = re.search(r"pytorch/pytorch:(\d+\.\d+\.\d+)", dockerfile_content)
    if match:
        return match.group(1)
    return None


def get_torch_version() -> Optional[str]:
    """Extract the torch version from the pyproject.toml file."""
    with open("pyproject.toml", "r") as file:
        pyproject_content = file.read()
    match = re.search(r"torch==(\d+\.\d+\.\d+)", pyproject_content)
    if match:
        return match.group(1)
    else:
        return None


def check_images_match() -> None:
    """Check that the base image is consistent between the Dockerfile and
    pyproject.toml."""
    docker_version = extract_image_from_dockerfile()
    torch_version = get_torch_version()

    if docker_version != torch_version:
        raise ValueError(
            f"The Dockerfile image {docker_version} is different from the "
            f" pyproject.toml version {torch_version}."
        )


if __name__ == "__main__":
    check_images_match()
