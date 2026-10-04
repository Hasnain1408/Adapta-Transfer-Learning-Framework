"""Dataset loading and format parsing infrastructure.

Provides dataset loaders for multiple scientific and tabular file formats:
- DatasetFormatLoader: Base interface for single-format file loaders.
- DatasetLoader: Composite router dispatching files to concrete loaders by extension.
- CSVDatasetLoader: Comma-separated value (.csv) loader.
- XLSXDatasetLoader: Excel spreadsheet (.xlsx) loader.
- MATDatasetLoader: MATLAB matrix (.mat) loader with dual-domain support.
- ARFFDatasetLoader: Weka attribute-relation file format (.arff) loader.
"""

from .arff_dataset_loader import ARFFDatasetLoader
from .csv_dataset_loader import CSVDatasetLoader
from .dataset_loader import DatasetFormatLoader, DatasetLoader
from .mat_dataset_loader import MATDatasetLoader
from .xlsx_dataset_loader import XLSXDatasetLoader

__all__ = [
    "ARFFDatasetLoader",
    "CSVDatasetLoader",
    "DatasetFormatLoader",
    "DatasetLoader",
    "MATDatasetLoader",
    "XLSXDatasetLoader",
]
