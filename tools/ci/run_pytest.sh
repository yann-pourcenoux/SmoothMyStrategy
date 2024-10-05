set -euxo pipefail

source tools/ci/before_script.sh

GIT_PYTHON_GIT_EXECUTABLE=`pwd`
echo $GIT_PYTHON_GIT_EXECUTABLE

mkdir -p data

python src/data/download.py --files src/data/config/test_data.json
pytest --cov --cov-report term --cov-report xml:coverage.xml src

