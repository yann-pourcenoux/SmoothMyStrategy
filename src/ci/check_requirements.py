"""Script to check that the base image is consistent between the Dockerfile and .gitlab-
ci.yml."""

import re


def extract_image_from_dockerfile() -> str:
    """Extract the image from the Dockerfile."""
    with open("Dockerfile", "r") as file:
        for line in file:
            if line.startswith("FROM "):
                return line.strip().split(" ")[1]
    return None


def extract_images_from_gitlab_ci_file() -> str:
    """Extract the images from the .gitlab-ci.yml file."""
    with open(".gitlab-ci.yml", "r") as file:
        content = file.read()
    pattern = r"image:\s*(nvidia/[^:\s]+:[^:\s]+)"
    matches = re.findall(pattern, content)

    if not matches:
        return None

    first_match = matches[0]
    assert all(
        match == first_match for match in matches
    ), "All nvidia images in .gitlab-ci.yml should be the same."
    return first_match


def check_images_match():
    """Check that the base image is consistent between the Dockerfile and .gitlab-
    ci.yml."""
    docker_image = extract_image_from_dockerfile()
    gitlab_ci_images = extract_images_from_gitlab_ci_file()

    if gitlab_ci_images is not None and docker_image != gitlab_ci_images:
        raise ValueError(
            f"The Dockerfile image {docker_image} is different from the .gitlab-ci.yml "
            f"image {gitlab_ci_images}."
        )


if __name__ == "__main__":
    check_images_match()
