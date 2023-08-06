import subprocess


def main():
    return_codes = {
        "black": subprocess.run("black --check ."),
        "isort": subprocess.run("isort --check ."),
        "ruff": subprocess.run("ruff ."),
        "mypy": subprocess.run("mypy ."),
        "docformatter": subprocess.run("docformatter --black --check -r ."),
    }

    for name, code in return_codes.items():
        print(f"{name}: {code}")

    if sum(return_codes.values()) != 0:
        raise ValueError("Some errors happened.")
