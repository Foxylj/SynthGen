import pdb
import torch
import torchaudio
import torch.nn as nn
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio

def main(model_name="facebook/musicgen-melody",
         synthgen="./../checkpoint/SynthGen_checkpoint.pth",
         duration=10):
    #torch.set_default_device('cuda')
    model = MusicGen.get_pretrained(model_name)
    model.set_generation_params(duration=duration)
    model.lm.load_state_dict(torch.load(synthgen)["lm_model_state_dict"])
    model.lm = model.lm.to(torch.float32)
    with torch.no_grad():

        mix_wav, mix_sr = torchaudio.load("./../dataset/mix/mix_0.wav")
        
        mix_wav=convert_audio(mix_wav, mix_sr, model.compression_model.sample_rate, model.compression_model.channels).unsqueeze(0).to('cuda')
        mix_wav=mix_wav[:,:,:model.compression_model.sample_rate*10]

        descriptions = ['Generate a synthesizer track for the given music.']
        mix_codes, _ = model.compression_model.encode(mix_wav)
        attributes, prompt_tokens = model._prepare_tokens_and_attributes(descriptions, prompt=None,melody_wavs=mix_wav)
        tokenized = model.lm.condition_provider.tokenize(attributes)
        condition_tensors = model.lm.condition_provider(tokenized)
        lm_output = model.lm.compute_predictions(codes=mix_codes, conditions=[], condition_tensors=condition_tensors)
        logits = lm_output.logits
        mask = lm_output.mask

        """m = nn.Softmax(dim=0)
        output_code=torch.tensor([])
        for dim2 in logits[0]:
            row=torch.tensor([]).to('cuda')
            for dim3 in dim2:
                if row.shape[0]==0:row=torch.tensor([torch.argmax(m(dim3))]).to('cuda')
                else:
                    temp=torch.tensor([torch.argmax(dim3)]).to('cuda')
                    row=torch.cat((row,temp)).to('cuda')
            if output_code.shape[0]==0:output_code=row.unsqueeze(0).to('cuda')
            else:
                output_code=torch.cat((output_code,row.unsqueeze(0)),dim=0).to('cuda')
        wav=model.compression_model.decode(output_code.unsqueeze(0))"""
        tokens = model._generate_tokens(attributes, prompt_tokens)
        wav=model.generate_audio(tokens)

        for idx, one_wav in enumerate(wav):
            # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
            audio_write('outcome/inference', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

if __name__ == "__main__":
    main()