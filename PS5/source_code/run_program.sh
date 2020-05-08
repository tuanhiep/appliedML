#!/bin/bash

echo "APPLIED MACHINE LEARNING PROJECT : Medical prediction for patients of kidney disease "

echo "Starting the program..."

# do features selection
python -W ignore features_selection/main_features_selection.py -data csv/kidney.csv
# do classification
python -W ignore classification/main_classification.py

echo "End the program !"
