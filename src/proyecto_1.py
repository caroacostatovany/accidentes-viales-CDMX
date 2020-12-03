"""
Main
"""

import os

from pipelines.ingestion import ingest
from pipelines.transformation import transform
from pipelines.feature_engineering import feature_engineering
from pipelines.magic_loop_v2 import modeling

path = os.getcwd() # Debe ser el path del repo

print("You are located in", path)
ingest(path, 'incidentes-viales-c5.csv')
transform(path)
feature_engineering(path, magic_loop=True)
modeling(path)
