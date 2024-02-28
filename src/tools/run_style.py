import subprocess

from loguru import logger


def main():
    return_codes = {
        "black": subprocess.run("python -m black --check src", shell=True).returncode,
        "isort": subprocess.run("python -m isort --check src", shell=True).returncode,
        "ruff": subprocess.run("python -m ruff src", shell=True).returncode,
        "docformatter": subprocess.run(
            "python -m docformatter --black --check -r src", shell=True
        ).returncode,
    }

    for name, code in return_codes.items():
        logger.info(f"{name}: {code}")

    if sum(return_codes.values()) != 0:
        raise ValueError("Some errors happened.")


if __name__ == "__main__":
    main()
