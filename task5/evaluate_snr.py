import os
import torch
import torchaudio

def calculate_snr(estimate_path, reference_path):
    """
    Calculates the SNR between an estimated audio file and its ground truth.
    """
    # Load audio files
    est_tensor, est_sr = torchaudio.load(estimate_path)
    ref_tensor, ref_sr = torchaudio.load(reference_path)
    
    # 1. Match Sample Rates
    if est_sr != ref_sr:
        print(f"Sample rate mismatch. Resampling estimate to {ref_sr}Hz...")
        resampler = torchaudio.transforms.Resample(est_sr, ref_sr)
        est_tensor = resampler(est_tensor)
        
    # 2. Downmix to mono if necessary (for fair comparison)
    if est_tensor.shape[0] > 1: est_tensor = torch.mean(est_tensor, dim=0, keepdim=True)
    if ref_tensor.shape[0] > 1: ref_tensor = torch.mean(ref_tensor, dim=0, keepdim=True)

    # 3. Trim to the exact same length (model framing can alter length slightly)
    min_len = min(est_tensor.shape[-1], ref_tensor.shape[-1])
    est = est_tensor[..., :min_len]
    ref = ref_tensor[..., :min_len]
    
    # 4. Calculate SNR
    noise = est - ref
    signal_power = torch.sum(ref ** 2)
    noise_power = torch.sum(noise ** 2) + 1e-8 # Prevent division by zero
    
    snr = 10 * torch.log10(signal_power / noise_power)
    
    return snr.item()

if __name__ == "__main__":
    # Manually pair your files here after listening to them
    #ESTIMATED_FILE = "task5/speakers_12_merged_Audacity_est_speaker_1.wav"
    #GROUND_TRUTH_FILE = "speaker1_me_20sec.mp3"

    #ESTIMATED_FILE = "task5/speakers_12_merged_Audacity_est_speaker_2.wav"
    #GROUND_TRUTH_FILE = "task5/speaker2_Anton_20sec.mp3"
    
    #ESTIMATED_FILE = "task5/speakers_13_merged_Audacity_est_speaker_2.wav"
    #GROUND_TRUTH_FILE = "task5/speaker1_me_20sec.mp3"

    #ESTIMATED_FILE = "task5/speakers_13_merged_Audacity_est_speaker_1.wav"
    #GROUND_TRUTH_FILE = "task5/speaker3_Vika_20s.mp3"

    #ESTIMATED_FILE = "task5/speakers_23_merged_Audacity_est_speaker_2.wav"
    #GROUND_TRUTH_FILE = "task5/speaker2_Anton_20sec.mp3"

    ESTIMATED_FILE = "task5/speakers_23_merged_Audacity_est_speaker_1.wav"
    GROUND_TRUTH_FILE = "task5/speaker3_Vika_20s.mp3"

    if os.path.exists(ESTIMATED_FILE) and os.path.exists(GROUND_TRUTH_FILE):
        snr_value = calculate_snr(ESTIMATED_FILE, GROUND_TRUTH_FILE)
        print(f"---")
        print(f"Estimate:  {ESTIMATED_FILE}")
        print(f"Reference: {GROUND_TRUTH_FILE}")
        print(f"Resulting SNR: {snr_value:.2f} dB")
        print(f"---")
    else:
        print("Error: Could not find one or both of the specified audio files.")