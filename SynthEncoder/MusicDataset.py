import torch
import librosa
import numpy as np
import os
from tqdm import tqdm

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, target_dir, sr=22050, duration=30):
        self.data = []
        self.sr = sr
        self.duration = duration
        self.max_length = sr * duration  # Calculate max_length

        file_indices = range(150)  # Assuming the range is from 0 to 149
        total_files = len(file_indices)

        with tqdm(total=total_files, desc='Loading Dataset', unit='file') as pbar:
            for i in file_indices:
                input_file = os.path.join(input_dir, f'mix_{i}.wav')
                target_file = os.path.join(target_dir, f'synth_{i}.wav')

                # Load and normalize audio files
                input_audio, _ = librosa.load(input_file, sr=self.sr, duration=self.duration)
                target_audio, _ = librosa.load(target_file, sr=self.sr, duration=self.duration)

                input_audio = self.normalize_audio(input_audio)
                target_audio = self.normalize_audio(target_audio)

                # Pad audio sequences to the same length
                input_audio = self._pad_audio(input_audio, self.max_length)
                target_audio = self._pad_audio(target_audio, self.max_length)

                self.data.append((torch.tensor(input_audio, dtype=torch.float32),
                                  torch.tensor(target_audio, dtype=torch.float32)))

                pbar.update(1)  # Update progress bar for each file

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_audio, target_audio = self.data[idx]
        input_audio = input_audio.unsqueeze(0)  # Add channel dimension
        target_audio = target_audio.unsqueeze(0)
        return input_audio, target_audio

    def _pad_audio(self, audio, max_length):
        padding_length = max_length - len(audio)
        if padding_length > 0:
            return np.pad(audio, (0, padding_length), 'constant')
        return audio

    def normalize_audio(self, audio):
        # Normalize the audio to the range [-1, 1]
        audio_max = np.max(np.abs(audio))
        if audio_max > 0:
            audio = audio / audio_max
        return audio
