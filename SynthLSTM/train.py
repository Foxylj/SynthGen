import torch
from torch.utils.data import DataLoader
from SynthLSTM import SynthLSTM
from MusicDatasetRAM import MusicDataset
# from MusicDataset import MusicDataset

def train_model(model, data_loader, learning_rate, epochs, device, save_path):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for mix, synth in data_loader:
            mix = mix.to(device).transpose(1, 2)  # Reshape to [batch_size, seq_len, feature_size]
            synth = synth.to(device).transpose(1, 2)

            # Forward pass
            outputs = model(mix)
            loss = criterion(outputs, synth)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Calculate average loss for the epoch
        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss}')

    # Save the model after training
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    # Hyperparameters
    feature_size = 128  # Number of Mel bands
    seq_len = 1292  # Sequence length (number of frames in spectrogram)
    hidden_size = 128
    output_size = feature_size  # Output should match the number of Mel bands
    num_layers = 10
    learning_rate = 0.001
    epochs = 100
    batch_size = 4

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    mix_dir = './mix'
    synth_dir = './synth'
    music_dataset = MusicDataset(mix_dir, synth_dir)
    print("Data Loading Complete!")
    music_dataloader = DataLoader(music_dataset, batch_size=batch_size, shuffle=True)
    print(device)

    # Model
    model = SynthLSTM(feature_size, hidden_size, output_size, num_layers).to(device)

    # Train the model
    train_model(model, music_dataloader, learning_rate, epochs, device, save_path='synth_lstm_model.pth')