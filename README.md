# SynthGen
## Advanced Music Synthesizer Track Generation and Modification

### Contributors
- Lijing Yang, lijingy@usc.edu
- Haosong Liu, haosongl@usc.edu
- Buyuan Huang, boyuanhu@usc.edu

## Project Overview
SynthGen stands at the intersection of music and machine learning, dedicated to the innovative generation and modification of synthesizer tracks from mixed music. The project is structured around three primary research directions:

1. **LSTM-Based Mel Spectrogram Modeling**: Leveraging LSTM models to generate synthesizer tracks, with Mel Spectrograms providing the crucial audio input.

2. **Encoder/Decoder for Raw Audio Processing**: This segment employs a sophisticated Encoder/Decoder model designed to intricately process raw audio data, enabling precise track modifications.

3. **Transformer-Based Multimodal Model**: Incorporating the latest from Meta's MusicGen, this aspect utilizes a transformer-based model finely tuned for multimodal operations, merging both audio and textual data to create and modify synthesizer tracks.

Through these avenues, SynthGen aims to revolutionize the creation and transformation of synthesizer tracks, harnessing the power of advanced machine learning techniques.

### Models
- **LSTM Model**: This model uses Mel Spectrograms to generate synthesizer tracks, effectively capturing the temporal dynamics inherent in audio.

- **Encoder/Decoder Model**: Operates on a convolutional neural network framework, this model directly interacts with raw audio data. Its primary objective is to isolate and refine synthesizer components within mixed tracks.

- **Transformer-Based Multimodal Model**: This model stands as a key component of the project, meticulously calibrated to process both audio and textual data, leveraging the innovative technology from Meta's MusicGen. It is fundamentally composed of three critical elements: the Encodec audio encoder/decoder, which is specialized in compressing and decompressing music, a text encoder adept at interpreting textual inputs, and a powerful single-stage transformer language model (LM). 

## Training

### SynthLSTM:
- *Details to be provided about training SynthLSTM.*

### SynthEncoder:
- *Details to be provided about training SynthEncoder.*

### SynthGen:
- **Prerequisites**: SynthGen requires Python 3.9 and PyTorch 2.0.0. Additionally, [AudioCraft](https://github.com/facebookresearch/audiocraft/tree/main) must be installed.
- **Data Preparation**: 
   - First, create a folder named `dataset` within the SynthGen directory.
   - Within `dataset`, create three sub-folders: `mix`, `synth`, and `description`.
   - Place your Mixed WAV tracks in `mix`, Isolated Synthesizer WAV tracks in `synth`, and Text Description files in `description`.

- **Model Checkpoints**:
   - Create a `checkpoint` folder within the SynthGen directory.
   - This folder will be used to store the model's training checkpoints.

- **Training the Model**:
   - Once you have prepared your dataset and checkpoint folder, you can initiate the training.
   - Run the training script with the following command:
     ```
     $ python trainer.py
     ```
   - The checkpoint with the lowest loss will be saved in the `checkpoint` directory.
- **Performing Inference**:
   - After the training is complete, you can run inference using the trained model.
   - For this, use the `audiocraft_inference.py` script located in the `inference` folder:
     ```
     $ python audiocraft_inference.py
     ```
   - This script allows you to generate new outputs based on the trained model.
