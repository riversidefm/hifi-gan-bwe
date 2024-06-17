from hifi_gan_bwe_common.hifi_gan_bwe.datasets import BWEDataset, group_by
from riverside_datasets.audio.riverside_audio_dataset import RiversideAudioDataset


import typing as T
from pathlib import Path


def bwe_dataset_from_riverside_audio_dataset(
    riverside_audio_dataset: RiversideAudioDataset,
    eval_set_seq_length: int,
) -> BWEDataset:
    class _RiversideBWEDataset(BWEDataset):
        def __init__(
            self,
            riverside_audio_dataset: RiversideAudioDataset,
            eval_set_seq_length: int,
        ):
            super().__init__(
                paths=riverside_audio_dataset.paths,
                sample_rate=riverside_audio_dataset._sample_rate,
                seq_length=riverside_audio_dataset._seq_length,
                eval_set_seq_length=eval_set_seq_length,
            )
            self.audio_path_to_sample = riverside_audio_dataset.audio_path_to_sample

        @property
        def eval_paths(self) -> T.List[Path]:
            samples = list(self.audio_path_to_sample.values())
            session_to_metadata = group_by(samples, lambda m: m.metadata.session_id)
            inverse_mapping = {v: k for k, v in self.audio_path_to_sample.items()}

            return [
                inverse_mapping[p[0]]
                for p in session_to_metadata.values()
                if len(p) > 1
            ]

    return _RiversideBWEDataset(
        riverside_audio_dataset=riverside_audio_dataset,
        eval_set_seq_length=eval_set_seq_length,
    )
