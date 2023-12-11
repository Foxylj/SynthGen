# SynthGen Model Card

## Model details

- **Model Name**: SynthGen v1.0
- **Developer**: Lijing Yang
- **Release Date**: TBD
- **Model Type**: LSTM (Long Short-Term Memory)
- **Model type:** Drawing from Facebook's advanced multimodal model, MusicGen, it features an EnCodec model for proficient audio tokenization and an auto-regressive language model based on the transformer architecture for music modeling. 

## Intended use
- **Primary intended use:** Generation of synthesizer tracks from mixed music tracks and description. 

- **Primary intended users:** Music producers, researchers, and hobbyists interested in automated music production.

- **Out-of-scope use cases:** The model is not designed for speech synthesis, non-musical audio processing, or real-time audio generation. The model is not suitable for environments lacking hardware accelerators.

## Training Data
- **Dataset Description**: 
  - The model was trained on the Musdb dataset, which contains a variety of music tracks. The dataset was split into mixed tracks and corresponding isolated synthesizer tracks.
  - The model is also trained using a text prompt, which is a description of the process we intend the model to generate.
- **Preprocessing Steps**: 
  - The audio is segmented into 10-second chunks, and to gather more data points, we employ a method of sliding through the music dataset using 5-second sliding windows.

## Evaluation Data
- **Dataset Description**: 
  - A subset of the Musdb dataset, separate from the training set, was used for evaluation.
- **Preprocessing Steps**: 
  - Similar preprocessing as training data was applied.

## Performance Measures
- **Evaluation Metrics**: 
  - The primary metric for performance evaluation is determined by calculating the cross-entropy loss between the target encoded by Encodec and the logits from the Transformer decoder.
- **Performance Results**: 
  - The model has exhibited a downward trend in cross-entropy loss. However, due to current limitations in computing resources, we are unable to identify the most optimal model at this moment. Moving forward, we plan to secure larger resources for training to address this issue.

## Ethical Considerations
- **Potential Biases**: 
  - No explicit biases identified in the dataset, but the diversity of music genres may affect model generalization.
- **Impact on Stakeholders**: 
  - Aimed at assisting music producers, but care should be taken to not misrepresent the model's capabilities.

## Limitations and Biases
- **Known Limitations**: 
  - Due to the large size of the model, both its deployment and training have been exceptionally challenging. In this project, the model has not converged well due to limited computational resources, even though there has been a consistent reduction in loss. Additionally, the uniformity of text inputs during training has caused SynthGen to lose its uniqueness compared to the LSTM, encode/decode models we used.
- **Biases**: 
  - Given the dataset's composition, the model's performance might be genre-specific and may not generalize well across different styles of music.

## Maintenance
- **Model Updates**: 
  - Future updates will focus on improving convergence and reducing MSE loss. Alternatives to MSE as a loss function may be explored.
- **Contact Information**: 
  - For inquiries or feedback, please contact Lijing Yang at lijingy@usc.edu 
  - Or Questions and comments about SynthGen can be sent via the [GitHub repository](https://github.com/Foxylj/SynthGen) of the project, or by opening an issue.
