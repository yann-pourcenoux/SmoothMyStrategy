set -euxo pipefail

python -m ruff src
python -m black --check src
python -m isort --check src
python -m docformatter --black --check -r src
