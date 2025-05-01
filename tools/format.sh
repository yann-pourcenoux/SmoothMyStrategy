#!/bin/bash

echo "Formatting code..."
echo "Ruff check..."
ruff check --fix .
echo "Ruff format..."
ruff format .
echo "Isort..."
isort .
echo "Docformatter..."
docformatter -r -i .
echo "Done!"
