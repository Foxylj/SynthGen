import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import soundfile as sf
import os
from EncoderDecoder import AudioEncoderDecoder
from MusicDataset import MusicDataset

def normalized_mse_loss(output, target):
    # Compute norms
    norm_output = torch.norm(output, dim=1, keepdim=True)
    norm_target = torch.norm(target, dim=1, keepdim=True)

    # Avoid division by zero
    norm_output = torch.where(norm_output == 0, torch.ones_like(norm_output), norm_output)
    norm_target = torch.where(norm_target == 0, torch.ones_like(norm_target), norm_target)

    # Normalize output and target
    output_normalized = output / norm_output
    target_normalized = target / norm_target

    # Compute MSE loss
    return nn.MSELoss()(output_normalized, target_normalized)



def save_epoch_output(output_tensor, epoch, file_path, sr=22050):
    # Detach the tensor from the computation graph and convert to numpy array
    audio_numpy = output_tensor[0].squeeze(0).detach().cpu().numpy()

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the audio for the epoch
    epoch_file = os.path.join(file_path, f'epoch_{epoch + 1}.wav')
    sf.write(epoch_file, audio_numpy, sr)

def train(model, train_loader, num_epochs, learning_rate, device, output_dir, model_save_path):
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = normalized_mse_loss(outputs, targets)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save a sample output at the end of each epoch
        save_epoch_output(outputs, epoch, output_dir)

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Model saved as {model_save_path}")

# Example usage
if __name__ == "__main__":
    input_channels = 1  # Mono audio
    hidden_size = 64
    output_channels = 1  # Mono audio
    learning_rate = 0.001
    num_epochs = 25
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model_save_path = './models/mono_64_4.pth'
    output_dir = './training_outputs'

    # Load your dataset here
    input_dir = './mix'
    target_dir = './synth'

    train_dataset = MusicDataset(input_dir, target_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model
    model = AudioEncoderDecoder(input_channels, hidden_size, output_channels)

    # Train the model
    train(model, train_loader, num_epochs, learning_rate, device, output_dir, model_save_path)
