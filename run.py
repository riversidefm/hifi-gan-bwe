import os
import librosa
import torch
import soundfile as sf
from tqdm import tqdm
from hifi_gan_bwe import BandwidthExtender

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
model = BandwidthExtender.from_pretrained("hifi-gan-bwe-10-42890e3-vctk-48kHz").to(device)

samples_dir = "/home/arno/ai-research/samples"
audios = os.listdir(samples_dir)
audios = [audio for audio in audios if "mpsenet" in audio]
sorted(audios)

for audio in tqdm(audios):
    x, sr = librosa.load(f"{samples_dir}/{audio}", sr=16000)
    x = torch.tensor(x, device=device)
    with torch.no_grad():
        y = model(x, sr)
    sf.write(f"{samples_dir}/{audio.split('16khz')[0]}48khz.wav", y, 48000)
