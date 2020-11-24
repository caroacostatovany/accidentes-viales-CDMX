"""
Main
"""

import os

from pipelines.ingestion import ingest
from pipelines.transformation import transform
from pipelines.feature_engineering import feature_engineering

path = os.getcwd() # Debe ser el path del repo

print("You are located in", path)
ingest(path, 'incidentes-viales-c5.csv')
transform(path)
feature_engineering(path)