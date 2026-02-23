import os
import torch
import numpy
import torchaudio
from asteroid.models import BaseModel

def separate_audio(mix_path, output_dir="task5/", model_name="JorisCos/ConvTasNet_Libri2Mix_sepclean_8k"):
    """
    Loads a mixed audio file, separates it using Asteroid, and saves the output tracks.
    """
    print(f"Loading Asteroid model: {model_name}...")
    model = BaseModel.from_pretrained(model_name)
    model.eval()

    print(f"Processing: {mix_path}")
    mix_tensor, sr = torchaudio.load(mix_path)
    
    # Resample if necessary to match model's expected sample rate
    if sr != model.sample_rate:
        print(f"Resampling from {sr}Hz to {model.sample_rate}Hz...")
        resampler = torchaudio.transforms.Resample(sr, model.sample_rate)
        mix_tensor = resampler(mix_tensor)
    
    # Downmix to mono if stereo
    if mix_tensor.shape[0] > 1:
        mix_tensor = torch.mean(mix_tensor, dim=0, keepdim=True)

    # Add batch dimension [batch, channels, time]
    mix_tensor = mix_tensor.unsqueeze(0)
        
    # Separate
    with torch.no_grad():
        est_sources = model(mix_tensor)
        
    # Remove batch dimension -> [n_sources, time]
    est_sources = est_sources.squeeze(0) 
    
    base_name = os.path.splitext(os.path.basename(mix_path))[0]
    
    # Save files
    for i in range(est_sources.shape[0]):
        out_path = os.path.join(output_dir, f"{base_name}_est_speaker_{i+1}.wav")
        torchaudio.save(out_path, est_sources[i].unsqueeze(0), int(model.sample_rate))
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    # Specify your mixed file here
    #MIXED_FILE = "task5/speakers_12_merged_Audacity.wav"
    #MIXED_FILE = "task5/speakers_13_merged_Audacity.wav" 
    MIXED_FILE = "task5/speakers_23_merged_Audacity.wav" 
    
    if os.path.exists(MIXED_FILE):
        separate_audio(MIXED_FILE)
    else:
        print(f"Error: {MIXED_FILE} not found.")