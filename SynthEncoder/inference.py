import torch
import torch.nn as nn
import librosa
import soundfile as sf
from EncoderDecoder import AudioEncoderDecoder  # Make sure this import points to your model definition

def load_model(model_path, input_channels, hidden_size, output_channels):
    # Initialize the model
    model = AudioEncoderDecoder(input_channels, hidden_size, output_channels)

    # Load the trained model parameters
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_audio(file_path, sr=22050, duration=30):
    # Load the audio file
    audio, _ = librosa.load(file_path, sr=sr, duration=duration, mono=True)

    # Add a channel dimension and convert to tensor
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return audio_tensor

def save_audio(audio_tensor, file_path, sr=22050):
    # Convert the tensor to numpy array and remove batch and channel dimensions
    audio_numpy = audio_tensor.squeeze(0).squeeze(0).numpy()

    # Save the audio
    sf.write(file_path, audio_numpy, sr)

def infer_and_save(input_file, output_file, model_path, input_channels, hidden_size, output_channels, device):
    # Load the model
    model = load_model(model_path, input_channels, hidden_size, output_channels)
    model.to(device)

    # Preprocess the input audio
    audio_tensor = preprocess_audio(input_file).to(device)

    # Perform inference
    with torch.no_grad():
        output_tensor = model(audio_tensor)

    # Save the output audio
    save_audio(output_tensor.cpu(), output_file)

# Example usage
if __name__ == "__main__":
    input_channels = 1  # Mono audio
    hidden_size = 64
    output_channels = 1  # Mono audio
    model_save_path = './models/mono_64_4.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # input_wav_file = '/scratch1/lijingy/EE641/Project/SynthLSTM/mix/mix_0.wav'
    input_wav_file = './mix/mix_0.wav'
    output_wav_file = './generated/synth_gen.wav'

    infer_and_save(input_wav_file, output_wav_file, model_save_path, input_channels, hidden_size, output_channels, device)
