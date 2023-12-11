# SynthGen Encoder/Decoder Model Card

## Model Details
- **Model Name**: SynthGen Encoder/Decoder v1.0
- **Developer**: Haosong Liu
- **Release Date**: TBD
- **Model Type**: Convolutional Neural Network (Encoder/Decoder)
- **Model Description**: 
  - This model uses an Encoder/Decoder architecture with convolutional layers, designed to modify mixed audio tracks with the aim of lowering MSE loss. The model processes raw audio data and attempts to isolate synthesizer tracks.

## Intended Use
- **Primary Use**: Modification of mixed music tracks to extract synthesizer components.
- **Intended Users**: Researchers and developers in the field of automated music processing.
- **Out-of-Scope Use Cases**: 
  - Not intended for real-time audio processing or speech synthesis tasks.

## Training Data
- **Dataset Description**: 
  - Trained on the Musdb dataset, comprising mixed tracks and isolated synthesizer tracks.
- **Preprocessing Steps**: 
  - Raw audio data were segmented and normalized for consistent input format.

## Evaluation Data
- **Dataset Description**: 
  - Utilized a separate subset of the Musdb dataset for evaluation.
- **Preprocessing Steps**: 
  - Similar to training data, involving segmentation and normalization.

## Performance Measures
- **Evaluation Metrics**: 
  - Normalized Mean Squared Error (MSE) was the primary metric.
- **Performance Results**: 
  - The model tended to modify the input track rather than generating new content, primarily to lower MSE loss.

## Ethical Considerations
- **Potential Biases**: 
  - The diversity in music genres could impact the model's performance and its ability to generalize.
- **Impact on Stakeholders**: 
  - Designed to aid in music analysis, but its limitations should be clearly communicated to users.

## Limitations and Biases
- **Known Limitations**: 
  - The model's focus on lowering MSE loss led to it modifying input tracks rather than generating new ones. Regular MSE loss resulted in very quiet output tracks, necessitating the use of normalized MSE loss.
- **Biases**: 
  - The model's performance may be genre-specific due to the dataset's composition.

## Maintenance
- **Model Updates**: 
  - Future development may explore alternative loss functions, such as perceptual loss, to overcome current limitations.
- **Contact Information**: 
  - For further information or feedback, please contact Haosong Liu at haosongl@usc.edu
  - Or Questions and comments about SynthGen can be sent via the [GitHub repository](https://github.com/Foxylj/SynthGen) of the project, or by opening an issue.

## Caveats and Recommendations
- **Best Practices**: 
  - Experimentation with different loss functions and audio preprocessing techniques is recommended.
- **Usage Recommendations**: 
  - Currently best suited for research purposes in the area of audio processing. Not recommended for production-level music generation or enhancement tasks.
