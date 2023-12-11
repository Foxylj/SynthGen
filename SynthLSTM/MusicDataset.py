import os
import torch
import librosa
import numpy as np

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, mix_dir, synth_dir, sr=22050, n_mels=128, num_samples=150):
        self.mix_dir = mix_dir
        self.synth_dir = synth_dir
        self.sr = sr
        self.duration = 30  # Duration in seconds
        self.n_mels = n_mels
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        mix_file = f'mix_{idx}.wav'
        synth_file = f'synth_{idx}.wav'
        
        mix_path = os.path.join(self.mix_dir, mix_file)
        synth_path = os.path.join(self.synth_dir, synth_file)

        # Load audio files
        mix, _ = librosa.load(mix_path, sr=self.sr, duration=self.duration)
        synth, _ = librosa.load(synth_path, sr=self.sr, duration=self.duration)

        # Convert to Mel spectrogram
        mix_mel = librosa.feature.melspectrogram(y=mix, sr=self.sr, n_mels=self.n_mels)
        synth_mel = librosa.feature.melspectrogram(y=synth, sr=self.sr, n_mels=self.n_mels)

        # Convert to decibels
        mix_db = librosa.power_to_db(mix_mel, ref=np.max)
        synth_db = librosa.power_to_db(synth_mel, ref=np.max)

        max_length = 1292  # This should be set to the length of your longest spectrogram

        # Pad the spectrograms to have the same length
        mix_db_padded = self._pad_spectrogram(mix_db, max_length)
        synth_db_padded = self._pad_spectrogram(synth_db, max_length)

        return torch.tensor(mix_db_padded, dtype=torch.float32), torch.tensor(synth_db_padded, dtype=torch.float32)


    def _pad_spectrogram(self, spectrogram, max_length):
        padding_length = max_length - spectrogram.shape[1]
        if padding_length > 0:
            return np.pad(spectrogram, ((0, 0), (0, padding_length)), 'constant')
        return spectrogram

