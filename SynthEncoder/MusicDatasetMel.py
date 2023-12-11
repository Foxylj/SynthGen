import torch
import librosa
import numpy as np
import os
from tqdm import tqdm

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, target_dir, sr=22050, duration=30, n_mels=128):
        self.data = []
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.max_length = sr * duration

        file_indices = range(150)

        with tqdm(total=len(file_indices), desc='Loading Dataset', unit='file') as pbar:
            for i in file_indices:
                input_file = os.path.join(input_dir, f'mix_{i}.wav')
                target_file = os.path.join(target_dir, f'synth_{i}.wav')

                # Load audio files and compute Mel spectrogram
                input_audio, _ = librosa.load(input_file, sr=self.sr, duration=self.duration, mono=True)
                target_audio, _ = librosa.load(target_file, sr=self.sr, duration=self.duration, mono=True)

                input_mel = librosa.feature.melspectrogram(y=input_audio, sr=self.sr, n_mels=self.n_mels)
                target_mel = librosa.feature.melspectrogram(y=target_audio, sr=self.sr, n_mels=self.n_mels)

                input_db = librosa.power_to_db(input_mel)
                target_db = librosa.power_to_db(target_mel)

                self.data.append((torch.tensor(input_db, dtype=torch.float32), 
                                  torch.tensor(target_db, dtype=torch.float32)))

                pbar.update(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
