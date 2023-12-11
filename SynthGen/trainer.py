## Some of the code refer to: https://github.com/chavinlo/musicgen_trainer/blob/main/train.py
import os
import pdb
import utils
import torch
import wandb
import scipy
import random
import librosa
import torchaudio

import typing as tp
import torch.nn as nn
from torch.optim import AdamW
from audiocraft.models import MusicGen
from torch.utils.data import DataLoader,Dataset

from audiocraft.data.audio_dataset import AudioDataset
from audiocraft.data.audio import audio_read, audio_write
from audiocraft.data.audio_utils import convert_audio
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout

SAVE_PATH = "models/"
DATA_PATH = "dataset/"
os.makedirs(SAVE_PATH, exist_ok=True)

def one_hot_encode(tensor, num_classes=2048):
    shape = tensor.shape
    one_hot = torch.zeros((shape[0], shape[1], num_classes))

    for i in range(shape[0]):
        for j in range(shape[1]):
            index = tensor[i, j].item()
            one_hot[i, j, index] = 1

    return one_hot

def preprocess_audio(mix_path,synth_path, model: MusicGen, duration: int = 10,sliding_window=5):
    
    original_mix_wav, mix_sr = torchaudio.load(mix_path)
    original_synth_wav, synth_sr = torchaudio.load(synth_path)
    assert mix_sr==synth_sr

    
    original_mix_wav=convert_audio(original_mix_wav, mix_sr, model.compression_model.sample_rate, model.compression_model.channels)
    original_synth_wav=convert_audio(original_synth_wav, synth_sr, model.compression_model.sample_rate, model.compression_model.channels)

    if original_mix_wav.shape[1] < model.sample_rate * duration or original_synth_wav.shape[1] < model.sample_rate * duration: return None

    duration = int(model.sample_rate * duration)
    start_sample = 0

    mix_codes_list=torch.Tensor()
    synth_codes_list=torch.Tensor()
    mix_wav_list=torch.Tensor()
    synth_wav_list=torch.Tensor()
    while start_sample+duration<original_mix_wav.shape[1]:
        mix_wav = original_mix_wav[:, start_sample : start_sample + duration]
        synth_wav = original_synth_wav[:, start_sample : start_sample + duration]

        mix_wav = mix_wav.cuda()
        mix_wav = mix_wav.unsqueeze(1)
        synth_wav = synth_wav.cuda()
        synth_wav = synth_wav.unsqueeze(1)

        with torch.no_grad(): 
            gen_mix_audio = model.compression_model.encode(mix_wav)
            gen_synth_audio = model.compression_model.encode(synth_wav)

        mix_codes, scale1 = gen_mix_audio
        synth_codes, scale2 = gen_synth_audio

        assert scale1 is None
        assert scale2 is None
        mix_codes_list= mix_codes if mix_codes_list.shape[0]==0 else torch.cat((mix_codes_list,mix_codes),dim=0)
        synth_codes_list=synth_codes if synth_codes_list.shape[0]==0 else torch.cat((synth_codes_list,synth_codes),dim=0)
        mix_wav_list=mix_wav if mix_wav_list.shape[0]==0 else torch.cat((mix_wav_list,mix_wav),dim=0)
        synth_wav_list=synth_wav if synth_wav_list.shape[0]==0 else torch.cat((synth_wav_list,synth_wav),dim=0)
        start_sample+=model.sample_rate * sliding_window

    return mix_codes_list,synth_codes_list,mix_wav_list,synth_wav_list

def process_data_from_path_list(mix_path_list,synth_path_list,description_path_list,model,duration=10):
    mix_codes_list=torch.Tensor()
    synth_codes_list=torch.Tensor()
    mix_wav_list=torch.Tensor()
    synth_wav_list=torch.Tensor()
    description_list=[]
    for mix_path,synth_path,description_path in zip(mix_path_list,synth_path_list,description_path_list):
        mix_codes,synth_codes,mix_wav,synth_wav=preprocess_audio(mix_path,synth_path, model, duration)

        mix_codes_list= mix_codes if mix_codes_list.shape[0]==0 else torch.cat((mix_codes_list,mix_codes),dim=0)
        synth_codes_list=synth_codes if synth_codes_list.shape[0]==0 else torch.cat((synth_codes_list,synth_codes),dim=0)
        mix_wav_list=mix_wav if mix_wav_list.shape[0]==0 else torch.cat((mix_wav_list,mix_wav),dim=0)
        synth_wav_list=synth_wav if synth_wav_list.shape[0]==0 else torch.cat((synth_wav_list,synth_wav),dim=0)
        for _ in range(mix_codes.shape[0]):description_list.append(open(description_path, "r").read().strip())
    return mix_codes_list,synth_codes_list,mix_wav_list,synth_wav_list,description_list
class Customize_Dataset(Dataset):
    def __init__(self,data_dir,model,duration=10):
        music_list = [d for d in os.listdir(os.path.join(data_dir, "mix/"))]
        self.data_map=[]
        for music in music_list:
            name, ext = os.path.splitext(music)
            idx=name.split("_")[-1]
            description_path=os.path.join(data_dir,"description/","des_"+idx+".txt")
            mix_path=os.path.join(data_dir,"mix/","mix_"+idx+".wav")
            synth_path=os.path.join(data_dir,"synth/","synth_"+idx+".wav")
            mix_codes_list,synth_codes_list,mix_wav_list,synth_wav_list,description_list=process_data_from_path_list([mix_path],[synth_path],[description_path],model,duration=duration)
            for i in range(len(description_list)):
                temp={
                    "mix_codes_list":mix_codes_list[i],
                    "synth_codes_list":synth_codes_list[i],
                    "mix_wav_list":mix_wav_list[i],
                    "synth_wav_list":synth_wav_list[i],
                    "description_list":description_list[i]
                }
                self.data_map.append(temp)
    
    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, index):
        mix_codes_list=self.data_map[index]["mix_codes_list"].squeeze(0)
        synth_codes_list=self.data_map[index]["synth_codes_list"].squeeze(0)
        mix_wav_list=self.data_map[index]["mix_wav_list"].squeeze(0)
        synth_wav_list=self.data_map[index]["synth_wav_list"].squeeze(0)
        description_list=self.data_map[index]["description_list"]
        return mix_codes_list,synth_codes_list,mix_wav_list,synth_wav_list,description_list  
    
def save_checkpoint(model,epoch,loss):
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'lm_model_state_dict': model.lm.state_dict(),
    }
    torch.save(checkpoint, 'checkpoint/SynthGen_checkpoint.pth')

def save_dataset(dataset,path="./dataset/preprocessed_dataset"):
    torch.save(dataset, os.path.join(path,"dataset.pt"))

def train(model_name="facebook/musicgen-melody",
          lr=1e-4,
          epochs=10,
          seed=42,
          train_duration=10,
          batch_size=3,
          use_scaler=True,
          save_models=True,
          use_preprocessed_dataset=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    utils.set_seed(seed=seed,device=device)

    model = MusicGen.get_pretrained(model_name)
    model.lm = model.lm.to(torch.float32)

    if use_preprocessed_dataset:
        dataset=torch.load("./dataset/preprocessed_dataset/dataset.pt",map_location="cpu")
    else:
        dataset=Customize_Dataset(DATA_PATH,model,duration=train_duration)
        save_dataset(dataset)

    dataloader_train = DataLoader(dataset=dataset,batch_size=batch_size)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.lm.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=1e-6,
    )
    epoch_loss=[]
    #torch.cuda.empty_cache()
    for epoch in range(epochs):
        model.lm.train()
        sum_loss=0
        for idx , batch in enumerate(dataloader_train):
            mix_codes_list,synth_codes_list,mix_wav_list,synth_wav_list,description_list =batch
            mix_codes_list,synth_codes_list,mix_wav_list,synth_wav_list=mix_codes_list.to(device),synth_codes_list.to(device),mix_wav_list.to(device),synth_wav_list.to(device)
            mix_wav_list=mix_wav_list.unsqueeze(1)
            description_list=list(description_list)
            optimizer.zero_grad()            
            attributes, prompt_tokens = model._prepare_tokens_and_attributes(description_list, prompt=None,melody_wavs=mix_wav_list)
            assert prompt_tokens is None

            tokenized = model.lm.condition_provider.tokenize(attributes)
            condition_tensors = model.lm.condition_provider(tokenized)
            with torch.cuda.amp.autocast():
                lm_output = model.lm.compute_predictions(codes=mix_codes_list, conditions=[], condition_tensors=condition_tensors)
                logits = lm_output.logits
                mask = lm_output.mask

                one_hot_code=torch.Tensor()
                for codes in synth_codes_list:
                    codes = one_hot_encode(codes, num_classes=2048)
                    one_hot_code= codes.unsqueeze(0) if one_hot_code.shape[0]==0 else torch.cat((one_hot_code,codes.unsqueeze(0)),dim=0)
                
                one_hot_code = one_hot_code.cuda()
                logits = logits.cuda()
                mask = mask.cuda()
                mask = mask.view(mask.shape[0],-1)

                loss=0
                for m,l,c in zip(mask,logits.view(logits.shape[0],-1, 2048),one_hot_code.view(one_hot_code.shape[0],-1, 2048)):
                    masked_logits = l[m]
                    masked_codes = c[m]
                    loss += criterion(masked_logits, masked_codes)
                loss/=logits.shape[0]

            (scaler.scale(loss) if use_scaler else loss).backward()
            sum_loss+=loss

            if use_scaler:scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 0.5)

            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        epoch_loss.append(sum_loss/len(dataloader_train))
        if save_models and (min(epoch_loss)==epoch_loss[-1]): save_checkpoint(model,epoch,epoch_loss)
        if epoch%2==0: print(f"Epoch [{epoch+1}/{epochs}], Loss: {sum_loss/len(dataloader_train)}")

    """for idx, one_wav in enumerate(pred):
                # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
                audio_write('outcome/electro_1_from_triner', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)"""



if __name__ == "__main__":
    train()