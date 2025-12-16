import numpy as np
from src.config import load_and_merge_configs
from typing import Optional

import torch
import torchaudio

class SpectrogramTransform:
    def __init__(self, config):
        data_config = config["data"]
        spec_config = config["spectrogram"]
        
        self.sample_rate =              data_config["sample_rate"]
        self.duration: int =            data_config.get("duration", None)

        self.n_fft: int =               spec_config["n_fft"]
        self.win_length: int =          spec_config["win_length"]
        self.hop_length: int =          spec_config["hop_length"]
        self.n_mels: int =              spec_config["n_mels"]
        self.f_min: float =             spec_config.get("f_min", 0.0)
        self.f_max: Optional[float] =   spec_config.get("f_max", None)
        self.eps: float =               spec_config.get("eps", 1e-10)

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels
            f_min=self.f_min,
            f_max=self.f_max
            power=2.0
            center=True
            pad_mode="reflect"
            norm=None
            mel_scale="htk"
        )

    def __call__(self, waveform):
        mel_spec = self.mel_spectrogram(waveform)
        db_mel_spec = self.amplitude_to_db(mel_spec)
        return db_mel_spec
