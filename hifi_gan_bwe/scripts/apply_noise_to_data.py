import json
import os
from pydantic import BaseModel
import torch
from tqdm import tqdm
from hifi_gan_bwe_common.hifi_gan_bwe import datasets
from hifi_gan_bwe_common.hifi_gan_bwe.scripts.train import (
    DatasetSplit,
    DatasetType,
    load_dataset,
)
from scipy.io.wavfile import write as write_wav


class Config(BaseModel):
    sample_rate: int = 48_000
    seed: int = 1337
    num_epochs: int = 5
    seq_length_sec: float = 2.0  # interval sizes to cut from the audio files

    noise_path: str = "/data/projects/audio-enhancement/datasets/DNS-Challenge/"
    dataset_path: str = (
        "/data/home/eliran/datasets/riverside-audio/vad-wavs/train/manifest.json"
    )
    dataset_type: DatasetType = DatasetType.RIVERSIDE
    dataset_split: DatasetSplit = DatasetSplit.TRAINING

    dataset_tgt_path: str = (
        "/data/projects/audio-enhancement/mp-senet/clean-noisy-pairings/riverside-5epochs-2sec"
    )


def main(config: Config):
    torch.manual_seed(config.seed)


    os.makedirs(config.dataset_tgt_path, exist_ok=False)
    noised_dataset_tgt_path = os.path.join(config.dataset_tgt_path, "noisy")
    clean_dataset_tgt_path = os.path.join(config.dataset_tgt_path, "clean")
    for p in [
        noised_dataset_tgt_path,
        clean_dataset_tgt_path,
    ]:
        os.makedirs(p, exist_ok=False)

    noise_set = datasets.DNSDataset(config.noise_path, seq_length=int(config.seq_length_sec * config.sample_rate))
    data_set = load_dataset(
        dataset_type=config.dataset_type,
        dataset_split=config.dataset_split,
        path=config.dataset_path,
        seq_length_sec=config.seq_length_sec,
        eval_set_seq_length=config.seq_length_sec,  # eval set is unused when generating noise
    )
    loader = torch.utils.data.DataLoader(
        data_set,
        collate_fn=datasets.Preprocessor(
            noise_set=noise_set, training=True, return_original_audio=True
        ),
        batch_size=datasets.BATCH_SIZE,
        shuffle=False,
        drop_last=True,
    )
    batch_idx = 0
    for epoch in tqdm(range(config.num_epochs)):
        for batch in loader:
            _, _, audio_tensor_noisy, audio_tensor_clean = batch

            for audio_tensor, dst_dir in [
                (audio_tensor_noisy, noised_dataset_tgt_path),
                (audio_tensor_clean, clean_dataset_tgt_path),
            ]:
                for j, audio_sample in enumerate(audio_tensor):
                    audio_sample = audio_sample.squeeze().detach().cpu().numpy()
                    filename = f"{epoch}_{batch_idx}_{j}.wav"
                    dst_path = os.path.join(dst_dir, filename)
                    write_wav(dst_path, config.sample_rate, audio_sample)
            batch_idx += 1

    config_dict = config.dict()
    with open(os.path.join(config.dataset_tgt_path, "config.json"), "w") as f:
        f.write(json.dumps(config_dict))


if __name__ == "__main__":
    main(config=Config())
