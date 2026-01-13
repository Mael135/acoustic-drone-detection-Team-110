import torch
import torchaudio
import torchaudio.transforms as transforms
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.features.transforms import SpectrogramTransform
from src.config import load_default_config

def parse_filename_metadata(filename: str):
    date, location, label, noise, gain, sr, duration, num = Path(filename).stem.split('__')
    is_drone = True if label == 'drone' else False
    if '_' in noise and len(noise.split('_')) == 4:
        parts = noise.split('_')
        noise_level = parts[1] + ' ' + parts[2]
        noise_type =  parts[3]
    else:
        noise_type = noise
        noise_level = ''
    gain = int(gain.replace('g', ''))
    sr = int(1000 * float(sr.replace('k', '.')))
    duration = duration.split('m')
    if len(duration) > 1:
        m, s = duration
        duration = (60 * int(m)) + int(s) if s != '' else (60 * int(m))
    else:
        duration = int(duration[0])
    num = int(num)
    return {"is_drone": is_drone, 
            "date": date,
            "location": location, 
            "noise_level": noise_level, 
            "noise_type": noise_type, 
            "gain": gain, 
            "sr": sr, 
            "duration": duration, 
            "num": num
            }

def process_data():
    RAW_DIR = Path("data/raw")
    PROCESSED_DIR = Path("data/processed")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    config = load_default_config()
    spectrogram_transform = SpectrogramTransform(config)
    
    CHUNK_DURATION = config['data']['duration']

    audio_files = list(RAW_DIR.rglob("*.wav"))
    print(f"Found {len(audio_files)} raw audio files. Starting processing...")
    drone_spec_number = 0
    no_drone_spec_number = 0
    metadata_rows = []
    expanded_metadata_rows = []
    for audio_path in tqdm(audio_files):
        meta = parse_filename_metadata(audio_path.name)
        if not meta:
            continue
        waveform, sr = torchaudio.load(audio_path)
        if (meta["sr"] != 0 and sr != meta["sr"]):
            print("whoops, sr incorrect!!!!")

        chunk_samples = int(CHUNK_DURATION * sr)
        total_samples = waveform.shape[1]
        num_chunks = total_samples / chunk_samples
        for i in range (int(num_chunks)):
            start = i * chunk_samples
            end = start + chunk_samples 
            chunk_waveform = waveform[:, start:end]
            spectrogram = spectrogram_transform(chunk_waveform)
            if (meta['is_drone']):
                prefix = 'drone'
                num = drone_spec_number
                drone_spec_number += 1
            else:
                prefix = 'ambience'
                num = no_drone_spec_number
                no_drone_spec_number += 1
            out_filename = f"{prefix}_{num:06d}.pt"
            out_path = PROCESSED_DIR / out_filename
            torch.save(spectrogram, out_path)
            chunk_meta = {}
            chunk_meta["filename"] = out_filename
            chunk_meta["start_time"] = i * CHUNK_DURATION
            chunk_meta["target"] = 1 if meta["is_drone"] else 0
            metadata_rows.append(chunk_meta)
            expanded_chunk_meta = meta.copy()
            expanded_chunk_meta["filename"] = out_filename
            expanded_metadata_rows.append(expanded_chunk_meta)
    df = pd.DataFrame(metadata_rows)
    cols = ['filename', 'target'] + [c for c in df.columns if c not in ['filename', 'target']]
    df = df[cols]

    expanded_df = pd.DataFrame(expanded_metadata_rows)
    cols = ['filename', 'is_drone'] + [c for c in expanded_df.columns if c not in ['filename', 'is_drone']]
    expanded_df = expanded_df[cols]

    csv_path = PROCESSED_DIR / "metadata.csv"
    df.to_csv(csv_path, index=False)
    expanded_csv_path = PROCESSED_DIR / "expanded_metadata.csv"
    expanded_df.to_csv(expanded_csv_path, index=False)


    print(f"Processing complete. Metadata saved to {csv_path}")
    print(f"Total processed chunks: {len(df)}")


        

process_data()