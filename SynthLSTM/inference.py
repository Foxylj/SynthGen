import torch
import librosa
import numpy as np
import soundfile as sf
from SynthLSTM import SynthLSTM

def load_model(model_path, input_size, hidden_size, output_size, num_layers):
    model = SynthLSTM(input_size, hidden_size, output_size, num_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_audio(file_path, sr=22050, n_mels=128, duration=30):
    # Load and convert to Mel spectrogram
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return torch.tensor(mel_db, dtype=torch.float32)

def inverse_mel_spectrogram(mel_db, sr=22050, n_fft=2048, hop_length=512):
    # Convert decibel Mel spectrogram back to linear Mel spectrogram
    mel_spec = librosa.db_to_power(mel_db)

    # Inverse Mel to audio - using Griffin-Lim
    audio = librosa.feature.inverse.mel_to_audio(mel_spec, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return audio

def synthesize_audio(input_file, output_file, model, device):
    # Preprocess the input audio
    input_data = preprocess_audio(input_file).unsqueeze(0).transpose(1, 2)  # Add batch dimension and transpose
    input_data = input_data.to(device)

    # Inference
    with torch.no_grad():
        output = model(input_data)
    
    # Convert output to waveform
    output_data = output.squeeze().transpose(0, 1).cpu().numpy()  # Remove batch dimension and transpose
    synthesized_audio = inverse_mel_spectrogram(output_data)

    # Save to file
    sf.write(output_file, synthesized_audio, 22050)

if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    model_path = './models/lstm_base.pth'
    model = load_model(model_path, 128, 128, 128, 10)
    model.to(device)

    for i in range(5):
        # Synthesize audio
        input_wav = './mix/mix_{}.wav'.format(i)
        output_wav = './result/synthgen_{}.wav'.format(i)
        synthesize_audio(input_wav, output_wav, model, device)
