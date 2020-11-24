"""
Main
"""

from pipelines.ingestion import ingest
from pipelines.transformation import transform
from pipelines.feature_engineering import feature_engineering

path = ""

ingest(path)
transform()
feature_engineering()