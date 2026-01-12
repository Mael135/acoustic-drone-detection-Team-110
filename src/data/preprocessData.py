# import torch
# import torchaudio
# import torchaudio.transforms as transforms
# import pandas as pd
from pathlib import Path

# from src.features.transforms import SpectrogramTransform
# from src.config import load_and_merge_configs

# config = load_and_merge_configs("config/default.yaml")

# spectrogramTransform = SpectrogramTransform(config)

def parse_filename_metadata(filename: str):
    parts = Path(filename).stem.split('__')
    return parts


print(parse_filename_metadata("data/raw/2025-12-15__room__background__ambience__g50__22k05__5m__01.wav"))