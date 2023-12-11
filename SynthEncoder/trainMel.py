import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import soundfile as sf
import os
import librosa
from EncoderDecoderMel import AudioEncoderDecoder
from MusicDatasetMel import MusicDataset

def normalized_mse_loss(output, target):
    # Assuming output and target are 4D tensors: [batch, channel, freq, time]
    norm_output = torch.norm(output, dim=(2, 3), keepdim=True)
    norm_target = torch.norm(target, dim=(2, 3), keepdim=True)

    norm_output = torch.where(norm_output == 0, torch.ones_like(norm_output), norm_output)
    norm_target = torch.where(norm_target == 0, torch.ones_like(norm_target), norm_target)

    output_normalized = output / norm_output
    target_normalized = target / norm_target

    return nn.MSELoss()(output_normalized, target_normalized)

def save_epoch_output(output_tensor, epoch, file_path, sr=22050):
    # Convert the tensor to numpy array and convert back to audio
    mel_spectrogram = output_tensor[0].squeeze(0).detach().cpu().numpy()
    audio = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sr)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    epoch_file = os.path.join(file_path, f'epoch_{epoch + 1}.wav')
    sf.write(epoch_file, audio, sr)

def train(model, train_loader, num_epochs, learning_rate, device, output_dir, model_save_path):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = normalized_mse_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        save_epoch_output(outputs, epoch, output_dir)

    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Model saved as {model_save_path}")

if __name__ == "__main__":
    input_channels = 1  # Mono audio - Mel Spectrogram has one channel
    hidden_size = 64
    output_channels = 1  # Mono audio
    learning_rate = 0.001
    num_epochs = 25
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = './models/mono_64_4.pth'
    output_dir = './training_outputs'

    input_dir = './mix'
    target_dir = './synth'
    train_dataset = MusicDataset(input_dir, target_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = AudioEncoderDecoder(input_channels, hidden_size, output_channels)

    train(model, train_loader, num_epochs, learning_rate, device, output_dir, model_save_path)
