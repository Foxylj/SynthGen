import torch
import librosa
import numpy as np
import os
from tqdm import tqdm

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, mix_dir, synth_dir, sr=22050, n_mels=128, max_length=1292):
        self.data = []
        self.sr = sr
        self.n_mels = n_mels
        self.duration = 30  # Duration in seconds
        self.max_length = max_length

        file_names = os.listdir(mix_dir)
        total_files = len(file_names)

        with tqdm(total=total_files, desc='Loading Dataset', unit='file') as pbar:
            for idx in range(total_files):
                mix_file = f'mix_{idx}.wav'
                synth_file = f'synth_{idx}.wav'

                mix_path = os.path.join(mix_dir, mix_file)
                synth_path = os.path.join(synth_dir, synth_file)

                mix, _ = librosa.load(mix_path, sr=self.sr, duration=self.duration)
                synth, _ = librosa.load(synth_path, sr=self.sr, duration=self.duration)

                mix_mel = librosa.feature.melspectrogram(mix, sr=self.sr, n_mels=self.n_mels)
                synth_mel = librosa.feature.melspectrogram(synth, sr=self.sr, n_mels=self.n_mels)

                mix_db = librosa.power_to_db(mix_mel, ref=np.max)
                synth_db = librosa.power_to_db(synth_mel, ref=np.max)

                mix_db_padded = self._pad_spectrogram(mix_db, self.max_length)
                synth_db_padded = self._pad_spectrogram(synth_db, self.max_length)

                self.data.append((mix_db_padded, synth_db_padded))

                # Update the progress bar
                pbar.update(1)


    def __getitem__(self, idx):
        mix_db, synth_db = self.data[idx]
        return torch.tensor(mix_db, dtype=torch.float32), torch.tensor(synth_db, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def _pad_spectrogram(self, spectrogram, max_length):
        padding_length = max_length - spectrogram.shape[1]
        if padding_length > 0:
            return np.pad(spectrogram, ((0, 0), (0, padding_length)), 'constant')
        return spectrogram
