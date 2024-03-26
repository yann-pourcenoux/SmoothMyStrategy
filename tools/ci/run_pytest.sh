set -euxo pipefail

GIT_PYTHON_GIT_EXECUTABLE=`pwd`
echo $GIT_PYTHON_GIT_EXECUTABLE

mkdir -p data

python src/data/download.py --files src/data/config/NAS_100.json
pytest src
