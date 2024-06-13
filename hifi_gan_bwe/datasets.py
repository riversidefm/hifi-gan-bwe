""" datasets for bandwidth extension training

This module defines custom torch datasets representing the VCTK and DNS
Challenge audio datasets. All source audio files are in the 16-bit PCM WAV
format and sampled at 48kHz. Audio clips are loaded by mmap'ing the wav files
and selecting a sub-sequence of each audio file.

As described in the paper, the VCTK dataset is partitioned by selecting the
first 99 speakers (in speaker id order) for training and leaving the
remaining speakers for valiation/evaluation. At training time, audio
clips are augmented with noise samples selected randomly from the
DNS Challenge noise dataset.

Each training batch (x, r, y) consists of a source clip x at a downsampled
sample rate (one of 8kHz, 16kHz, and 24kHz, randomly selected), the
downsampled rate r, and the original audio y, which is used as a training
target.

The first sample for each validation speaker is used for model evaluation
(samples for listening, spectrogram images, etc.), and these can be accessed
using VCTKDataset.eval_set.

https://datashare.ed.ac.uk/handle/10283/2950
https://github.com/microsoft/DNS-Challenge

"""

from abc import abstractmethod
import mmap
import typing as T
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio

from torch.utils.data import Dataset
import librosa

A = T.TypeVar("A")
B = T.TypeVar("B")

SAMPLE_RATE = 48000
RESAMPLE_RATES = [8000, 16000, 24000]
BATCH_SIZE = 4
SEQ_LENGTH = int(1.0 * SAMPLE_RATE)
SIGNAL_RMS_MIN = -30
SIGNAL_RMS_MAX = -12
NOISE_SNR_MIN = 15
NOISE_SNR_MAX = 60
TRAIN_SPEAKERS = 99


class WavDataset(Dataset):
    """pytorch dataset for a collection of WAV files

    This class provides efficient access to random subsequences of a set of
    audio files using mmap.

    Args:
        paths: the list of wav file paths to sample
        seq_length: the number of contiguous samples to load from each file

    """

    def __init__(
        self,
        paths: T.Iterator[Path],
        seq_length: int = SEQ_LENGTH,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__()

        self._paths = [
            p
            for p in paths
            if p.stat().st_size // 2 > seq_length
            and librosa.get_samplerate(p) == sample_rate
        ]
        self._seq_length = seq_length
        self._sample_rate = sample_rate

    @property
    def paths(self) -> T.List[Path]:
        return self._paths.copy()

    @property
    @abstractmethod
    def eval_set(self) -> T.List[np.ndarray]:
        pass

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.load(index, seq_length=self._seq_length)

    def load(self, index: int, seq_length: int = -1) -> np.ndarray:
        with self._paths[index].open("rb") as file:
            with mmap.mmap(file.fileno(), length=0, prot=mmap.PROT_READ) as data:
                assert int.from_bytes(data[22:24], "little") == 1
                assert int.from_bytes(data[34:36], "little") == 16

                sample_rate = int.from_bytes(data[24:28], "little")
                matching_sample_rate = sample_rate == self._sample_rate

                if not matching_sample_rate:
                    raise ValueError(
                        f"Sample rate mismatch: {self._sample_rate} vs {sample_rate}"
                    )

                # attach the mmap'd file to a numpy buffer
                audio = np.frombuffer(data[:], dtype=np.int16)[22:]

                # take a subsequence of the file if requested, so that
                # we don't load the whole file into memory if not needed
                if seq_length != -1:
                    offset = np.random.randint(max(1, len(audio) - seq_length))
                    audio = audio[offset : offset + seq_length]

                # convert the audio from PCM-16 to float32
                audio = audio.astype(np.float32) / 32767.0
                return audio

    def _load(self, index: int, seq_length: int = -1) -> np.ndarray:
        # load wav using librosa
        path = self._paths[~index]
        duration_ms = librosa.get_duration(path=path) * 1000
        seq_length_ms = seq_length / self._sample_rate * 1000

        cut_kwargs = {}
        if seq_length != -1:
            offset_ms = np.random.randint(max(1, duration_ms - seq_length_ms))
            cut_kwargs = {
                "offset": offset_ms / 1000.0,
                "duration": seq_length_ms / 1000.0,
            }

        audio = librosa.load(
            self._paths[index], sr=self._sample_rate, mono=True, **cut_kwargs
        )[0]

        if len(cut_kwargs) > 0:
            audio = audio[:seq_length]

        return audio


class VCTKDataset(WavDataset):
    """VCTK speech dataset wrapper"""

    def __init__(self, path: str, training: bool):
        paths = sorted((Path(path) / "wav48").glob("*"))
        paths = paths[:TRAIN_SPEAKERS] if training else paths[TRAIN_SPEAKERS:]
        super().__init__(
            paths=(p for s in paths for p in s.glob("*.wav")),
            seq_length=SEQ_LENGTH,
        )

    @property
    def eval_set(self) -> T.List[np.ndarray]:
        speaker_paths = group_by(self.paths, lambda p: p.parent.name)
        return [self.load(self.paths.index(p[0])) for p in speaker_paths.values()]


class DNSDataset(WavDataset):
    """DNS Challenge noise dataset wrapper"""

    def __init__(self, path: str, seq_length: int = SEQ_LENGTH):
        noise_path = Path(path) / "datasets_fullband" / "noise_fullband"
        super().__init__(
            paths=noise_path.glob("*.wav"),
            seq_length=BATCH_SIZE * seq_length,
        )


class Preprocessor:
    """pytorch data loader preprocessor for HiFi-GAN+ training

    Args:
        noise_set: noise dataset, used for augmentation during training
        training: True for training, False for inference
        device: the pytorch device to load batches onto

    """

    def __init__(
        self,
        noise_set: WavDataset,
        training: bool,
        device: str = "cuda",
        return_original_audio: bool = False,
    ):
        self._device = device
        self._training = training
        self._noise_set = noise_set
        self._return_original_audio = return_original_audio

    def __call__(
        self, batch: T.List[np.ndarray]
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # batch the training examples using the default batcher
        # and copy to the GPU, where resampling is much faster
        orig_y = torch.utils.data.dataloader.default_collate(batch).unsqueeze(1)
        y = orig_y.to(self._device)

        # only augment during training
        if self._training:
            y = self._augment(y)

        r = np.random.choice(RESAMPLE_RATES)
        x = torchaudio.functional.resample(y, SAMPLE_RATE, r)

        if self._return_original_audio:
            return x, r, y, orig_y
        return x, r, y

    def _augment(self, y: torch.Tensor) -> torch.Tensor:
        # perform random amplitude augmentation
        signal_rms = torch.sqrt((y**2).mean(-1, keepdim=True))
        target_rms = torch.from_numpy(
            np.random.uniform(
                SIGNAL_RMS_MIN,
                SIGNAL_RMS_MAX,
                size=[BATCH_SIZE, 1, 1],
            )
        ).to(y.device)
        target_rms = 10 ** (target_rms / 20)
        gain = target_rms / (signal_rms + 1e-5)
        y *= gain
        signal_rms *= gain

        # load a noise sample
        noise_index = np.random.randint(len(self._noise_set))
        noise = torch.from_numpy(self._noise_set[noise_index]).to(y.device)
        noise = noise.reshape([BATCH_SIZE, 1, -1])
        noise_rms = torch.sqrt((noise**2).mean(-1, keepdim=True))

        # compute a random SNR, adjust the noise gain to match it,
        # and mix the noise with the audio
        source_snr = signal_rms / (noise_rms + 1e-5)
        target_snr = torch.from_numpy(
            np.random.uniform(
                NOISE_SNR_MIN,
                NOISE_SNR_MAX,
                size=[BATCH_SIZE, 1, 1],
            )
        ).to(y.device)
        target_snr = 10 ** (target_snr / 20)
        noise *= source_snr / target_snr
        y += noise

        # scale the signal to avoid clipping
        # this is critical for bandwidth extension, since we are
        # trying to reproduce the target signal exactly and
        # these models use a final tanh activation
        signal_peak = y.abs().max(-1, keepdim=True).values
        gain = 1.0 / torch.maximum(signal_peak, torch.tensor(1.0))
        y *= gain

        return y


def group_by(seq: T.Iterable[A], key: T.Callable[[A], B]) -> T.Dict[B, T.List[A]]:
    """groups an input sequence into a dictionary of lists of elements

    Args:
        seq: the sequence to group
        key: a lambda used to group each element

    Returns:
        groups: a dictionary of elements grouped by key(element)

    """

    groups = defaultdict(list)
    for value in seq:
        groups[key(value)].append(value)
    return groups
