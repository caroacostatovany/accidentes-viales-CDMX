"""
Main
"""

import os

from pipelines.ingestion import ingest
from pipelines.transformation import transform
from pipelines.feature_engineering import feature_engineering
from pipelines.modeling import modeling
from pipelines.model_evaluation import metrics
from pipelines.bias_fairness import bias_main

path = os.getcwd() # Debe ser el path del repo
magic_loop = True

print("You are located in", path)
ingest(path, 'incidentes-viales-c5.csv')
transform(path)
feature_engineering(path, magic_loop=magic_loop)
if magic_loop:
    modeling(path)
    metrics(path)
    bias_main(path)
