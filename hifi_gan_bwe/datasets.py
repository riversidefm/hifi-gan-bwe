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

from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
import mmap
import typing as T
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset
import librosa
import torchaudio

A = T.TypeVar("A")
B = T.TypeVar("B")

SAMPLE_RATE = 48000
RESAMPLE_RATES = [8000, 16000, 24000]
BATCH_SIZE = 4
SEQ_LENGTH = int(1.0 * SAMPLE_RATE)
SEQ_LENGTH_SEC = SEQ_LENGTH / SAMPLE_RATE
SIGNAL_RMS_MIN = -30
SIGNAL_RMS_MAX = -12
NOISE_SNR_MIN = 15
NOISE_SNR_MAX = 60
TRAIN_SPEAKERS = 99


class _SR(int, Enum):
    SR_44_1K = 44100
    SR_48K = 48000


class _SampleWidth(int, Enum):
    SW_16 = 16
    SW_32 = 32


def _get_seq_from_buffer(
    buffer: mmap.mmap, seq_length: int, sample_width: _SampleWidth
) -> np.ndarray:
    if sample_width == _SampleWidth.SW_16:
        dtype = np.int16
    elif sample_width == _SampleWidth.SW_32:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # attach the mmap'd file to a numpy buffer
    buffer = np.frombuffer(buffer[:], dtype=dtype)[22:]

    # take a subsequence of the file if requested, so that
    # we don't load the whole file into memory if not needed
    if seq_length != -1:
        offset = np.random.randint(max(1, len(buffer) - seq_length))
        buffer = buffer[offset : offset + seq_length]

    return buffer.astype(np.float32) / ((1 << (sample_width.value - 1)) - 1)


def _create_resampler(
    orig_freq: _SR,
    new_freq: _SR,
    lowpass_filter_width: int = 64,
    rolloff: float = 0.9475937167399596,
    resampling_method: str = "sinc_interp_kaiser",
    beta: float = 14.769656459379492,
):
    """
    Create a resampler with the given parameters
    # default "kaiser_best" resampling from https://pytorch.org/audio/0.12.1/tutorials/audio_resampling_tutorial.html
    """
    return torchaudio.transforms.Resample(
        orig_freq=orig_freq.value,
        new_freq=new_freq.value,
        lowpass_filter_width=lowpass_filter_width,
        rolloff=rolloff,
        resampling_method=resampling_method,
        beta=beta,
    ).train(False)


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
        seq_length: int,
        sample_rate: _SR,
        min_duration_samples: int = 0,
        allow_invalid_samples_as_none: bool = False,
        allow_resample: bool = False,
        min_sample_rate: T.Optional[_SR] = None,
        allowed_sample_widths: T.Optional[T.Set[_SampleWidth]] = None,
    ):
        super().__init__()
        # These params determine the output's sample rate and sequence length
        # When using different yet allowed sample rates, the resampler will be used to convert the audio and the sequence length will be adjusted accordingly
        self._seq_length = seq_length
        self._sample_rate = sample_rate

        # Sample widths
        if allowed_sample_widths is None:
            allowed_sample_widths = {_SampleWidth.SW_16}
        self._allowed_sample_widths = allowed_sample_widths

        # Sample rates
        self._allowed_sample_rates = {sample_rate}
        if allow_resample:
            if min_sample_rate is None or min_sample_rate >= sample_rate:
                raise ValueError(
                    f"min_sample_rate must be provided and less than sample_rate if upsample is allowed"
                )
            self._allowed_sample_rates = {
                sr for sr in _SR if sr.value >= min_sample_rate.value
            }
            self._sr_to_resampler = {
                sr: _create_resampler(orig_freq=sr, new_freq=sample_rate)
                for sr in self._allowed_sample_rates
            }

        # Filtering out audio files that are too short or have an unsupported sample rate
        sr_to_min_duration_samples = {
            sr: int(min_duration_samples * sr / sample_rate)
            for sr in self._allowed_sample_rates
        }
        self._paths = []
        for p in paths:
            try:
                sr = librosa.get_samplerate(p)
            except Exception:
                continue
            if sr not in self._allowed_sample_rates:
                continue

            sr_seq_length, sr_min_duration_samples = (
                self._get_sr_adapted_seq_length(sr, self._seq_length),
                sr_to_min_duration_samples[sr],
            )
            if (p.stat().st_size // 2) <= max(sr_seq_length, sr_min_duration_samples):
                continue

            self._paths.append(p)

        self._allow_invalid_samples_as_none = allow_invalid_samples_as_none

    @property
    def paths(self) -> T.List[Path]:
        return self._paths.copy()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def default_seq_length(self) -> int:
        return self._seq_length

    @lru_cache(maxsize=100)
    def _get_sr_adapted_seq_length(self, sr: _SR, seq_length: int) -> int:
        if seq_length == -1:
            return -1
        return int(seq_length * sr / self._sample_rate)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> T.Optional[np.ndarray]:
        try:
            return self.load(index, seq_length=self._seq_length)
        except Exception as e:
            if self._allow_invalid_samples_as_none:
                return None
            raise e

    def load(self, index: int, seq_length: int = -1) -> np.ndarray:
        with self._paths[index].open("rb") as file:
            with mmap.mmap(file.fileno(), length=0, prot=mmap.PROT_READ) as data:
                assert int.from_bytes(data[22:24], "little") == 1

                sample_width = _SampleWidth(int.from_bytes(data[34:36], "little"))
                assert sample_width in self._allowed_sample_widths

                sample_rate = _SR(int.from_bytes(data[24:28], "little"))
                matching_sample_rate = sample_rate in self._allowed_sample_rates

                if not matching_sample_rate:
                    raise ValueError(
                        f"Sample rate mismatch: {self._sample_rate} vs {self._allowed_sample_rates}"
                    )

                need_resample = sample_rate != self._sample_rate

                sr_seq_length = self._get_sr_adapted_seq_length(sample_rate, seq_length)
                audio = _get_seq_from_buffer(
                    buffer=data, seq_length=sr_seq_length, sample_width=sample_width
                )
                if need_resample:
                    resampler = self._sr_to_resampler[sample_rate]
                    audio = (
                        resampler(torch.tensor(audio).unsqueeze(0)).squeeze().numpy()
                    )

                # convert the audio from PCM-16 to float32
                return audio


class BWEDataset(WavDataset, ABC):
    def __init__(self, eval_set_seq_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_set_seq_length = eval_set_seq_length

    @property
    @abstractmethod
    def eval_paths(self) -> T.List[Path]:
        pass

    @property
    def eval_set(self) -> T.List[np.ndarray]:
        return [
            self.load(self.paths.index(p), self.eval_set_seq_length)
            for p in self.eval_paths
        ]


class VCTKDataset(BWEDataset):
    """VCTK speech dataset wrapper"""

    def __init__(self, path: str, training: bool, eval_set_seq_length: int):
        if eval_set_seq_length != -1:
            raise ValueError("currently VCTK does not support eval set sequence length")
        paths = sorted((Path(path) / "wav48").glob("*"))
        paths = paths[:TRAIN_SPEAKERS] if training else paths[TRAIN_SPEAKERS:]
        super().__init__(
            paths=(p for s in paths for p in s.glob("*.wav")),
            seq_length=SEQ_LENGTH,
            eval_set_seq_length=eval_set_seq_length,
            sample_rate=SAMPLE_RATE,
        )

    @property
    def eval_paths(self) -> T.List[Path]:
        speaker_paths = group_by(self.paths, lambda p: p.parent.name)
        return [p[0] for p in speaker_paths.values()]


class DNSDataset(WavDataset):
    """DNS Challenge noise dataset wrapper"""

    def __init__(
        self,
        path: str,
        seq_length_sec: int,
        sample_rate: int = SAMPLE_RATE,
        num_samples: T.Optional[int] = None,
    ):
        noise_path = Path(path) / "datasets_fullband" / "noise_fullband"
        seq_length = int(seq_length_sec * sample_rate)
        super().__init__(
            paths=list(noise_path.glob("*.wav"))[:num_samples],
            seq_length=seq_length,
            sample_rate=sample_rate,
        )


class Preprocessor:
    """pytorch data loader preprocessor for HiFi-GAN+ training

    Args:
        noise_set: noise dataset, used for augmentation during training
        training: True for training, False for inference
        device: the pytorch device to load batches onto
        diverse_noises: whether to use a different noise sample for each batch. otherwise will try to split a single noise sample among the batch samples (original behavior)
        perform_amplitude_augmentation: whether to perform random amplitude augmentation to the audio, according to noise_snr min and max
    """

    def __init__(
        self,
        noise_set: WavDataset,
        training: bool,
        device: str = "cuda",
        target_sample_rate: int = SAMPLE_RATE,
        return_original_audio: bool = False,
        diverse_noises: bool = False,
        signal_rms_min: float = SIGNAL_RMS_MIN,
        signal_rms_max: float = SIGNAL_RMS_MAX,
        noise_snr_min: float = NOISE_SNR_MIN,
        noise_snr_max: float = NOISE_SNR_MAX,
        perform_amplitude_augmentation: bool = True,
    ):
        self._device = device
        self._training = training
        self._noise_set = noise_set
        self._return_original_audio = return_original_audio
        self._target_sample_rate = target_sample_rate
        self._diverse_noises = diverse_noises
        self._signal_rms_min = signal_rms_min
        self._signal_rms_max = signal_rms_max
        self._noise_snr_min = noise_snr_min
        self._noise_snr_max = noise_snr_max
        self._perform_amplitude_augmentation = perform_amplitude_augmentation

    def __call__(
        self,
        batch: T.List[np.ndarray],
        return_only_noisy_audio: bool = False,
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # batch the training examples using the default batcher
        # and copy to the GPU, where resampling is much faster
        orig_y = torch.utils.data.dataloader.default_collate(batch).unsqueeze(1)
        y = orig_y.to(self._device)

        # only augment during training
        if self._training:
            y = self._augment(y)
            if return_only_noisy_audio:
                return y
        r = np.random.choice(RESAMPLE_RATES)
        x = torchaudio.functional.resample(y, SAMPLE_RATE, r)

        if self._return_original_audio:
            return x, r, y, orig_y
        return x, r, y

    def _augment(self, y: torch.Tensor) -> torch.Tensor:
        batch_size = y.shape[0]
        signal_rms = torch.sqrt((y**2).mean(-1, keepdim=True))

        if self._perform_amplitude_augmentation:
            # perform random amplitude augmentation
            target_rms = torch.from_numpy(
                np.random.uniform(
                    self._signal_rms_min,
                    self._signal_rms_max,
                    size=[batch_size, 1, 1],
                )
            ).to(y.device)
            target_rms = 10 ** (target_rms / 20)
            gain = target_rms / (signal_rms + 1e-5)
            y *= gain
            signal_rms *= gain

        # load a noise sample
        if self._diverse_noises:
            noise_indices = [
                np.random.randint(len(self._noise_set)) for _ in range(batch_size)
            ]
            noise = [
                np.resize(
                    librosa.resample(
                        self._noise_set[noise_index],
                        orig_sr=self._noise_set._sample_rate,
                        target_sr=self._target_sample_rate,
                        axis=-1,
                    ),
                    (1, y.shape[-1]),
                )
                for noise_index in noise_indices
            ]
        else:
            noise_index = np.random.randint(len(self._noise_set))
            noise = np.resize(
                librosa.resample(
                    self._noise_set[noise_index],
                    orig_sr=self._noise_set._sample_rate,
                    target_sr=self._target_sample_rate,
                    axis=-1,
                ),
                (1, y.shape[-1] * batch_size),
            )

        noise = torch.from_numpy(np.array(noise)).to(y.device)
        noise = noise.reshape([batch_size, 1, -1])
        noise_rms = torch.sqrt((noise**2).mean(-1, keepdim=True))
        # compute a random SNR, adjust the noise gain to match it,
        # and mix the noise with the audio
        source_snr = signal_rms / (noise_rms + 1e-5)
        target_snr = torch.from_numpy(
            np.random.uniform(
                self._noise_snr_min,
                self._noise_snr_max,
                size=[batch_size, 1, 1],
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


if __name__ == "__main__":
    import os

    audio_dir = "/data/projects/audio-enhancement/datasets/riverside-high-quality-vad-wavs-2/valid/vad_segments"
    num_samples = 1000
    audio_paths = [
        Path(p.path) for i, p in enumerate(os.scandir(audio_dir)) if i < num_samples
    ]

    dataset = WavDataset(
        paths=audio_paths,
        seq_length=48000 * 1,
        sample_rate=_SR.SR_48K,
        allow_resample=True,
        min_sample_rate=_SR.SR_44_1K,
        allowed_sample_widths=set(_SampleWidth),
    )
    print(len(dataset))
