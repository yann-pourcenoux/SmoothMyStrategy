set -euxo pipefail

source tools/ci/before_script.sh

python -m ruff src
python -m ruff format --check src
python -m isort --check src
python -m docformatter --black --check -r src
