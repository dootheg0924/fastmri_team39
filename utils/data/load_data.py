import h5py
from utils.data.transforms import DataTransform
import re
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
import numpy as np


_ACCELERATION_TAG = re.compile(r"(?:^|_)acc([48])(?:_|\.|$)", re.IGNORECASE)


def acceleration_from_filename(filename):
    """Return 4/8 from an explicit filename tag, or ``None`` when ambiguous."""
    matches = _ACCELERATION_TAG.findall(Path(filename).name)
    if len(matches) != 1:
        return None
    return int(matches[0])


class BalancedAccelerationSampler(Sampler):
    """Yield an exact 50/50, alternating acc4/acc8 slice stream per epoch.

    ``oversample`` (the default) visits every majority slice once and cycles
    through reshuffled minority slices until both groups have equal length.
    ``undersample`` draws an equal-size subset from both groups. A private
    generator keyed by ``seed + epoch`` makes an epoch reproducible across
    warm-start and resume without consuming the global training RNG.
    """

    def __init__(self, dataset, mode="oversample", seed=430):
        if mode not in {"oversample", "undersample"}:
            raise ValueError(
                "Acceleration balance mode must be 'oversample' or 'undersample'."
            )
        self.mode = mode
        self.seed = int(seed)
        self.epoch = 0
        self.indices = {4: [], 8: []}
        unknown = []
        for index, (path, _) in enumerate(dataset.kspace_examples):
            acceleration = acceleration_from_filename(path.name)
            if acceleration is None:
                unknown.append(path.name)
            else:
                self.indices[acceleration].append(index)
        if unknown:
            examples = ", ".join(sorted(set(unknown))[:5])
            raise ValueError(
                "Balanced acceleration sampling requires one explicit _acc4_ or "
                f"_acc8_ filename tag per volume; unrecognized examples: {examples}"
            )
        if not self.indices[4] or not self.indices[8]:
            raise ValueError(
                "Balanced acceleration sampling requires both acc4 and acc8 slices; "
                f"found acc4={len(self.indices[4])}, acc8={len(self.indices[8])}."
            )

    @property
    def samples_per_acceleration(self):
        counts = (len(self.indices[4]), len(self.indices[8]))
        return max(counts) if self.mode == "oversample" else min(counts)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    @staticmethod
    def _draw(pool, count, generator):
        drawn = []
        while len(drawn) < count:
            order = torch.randperm(len(pool), generator=generator).tolist()
            drawn.extend(pool[i] for i in order[: count - len(drawn)])
        return drawn

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        count = self.samples_per_acceleration
        drawn = {
            acceleration: self._draw(self.indices[acceleration], count, generator)
            for acceleration in (4, 8)
        }
        first, second = ((4, 8) if self.epoch % 2 == 0 else (8, 4))
        alternating = []
        for i in range(count):
            alternating.extend((drawn[first][i], drawn[second][i]))
        return iter(alternating)

    def __len__(self):
        return 2 * self.samples_per_acceleration

    def summary(self):
        return {
            "mode": self.mode,
            "source_acc4": len(self.indices[4]),
            "source_acc8": len(self.indices[8]),
            "sampled_per_acceleration": self.samples_per_acceleration,
            "samples_per_epoch": len(self),
        }


class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []

        if not forward:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

        kspace_files = list(Path(root / "kspace").iterdir())
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)

            self.kspace_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]


    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]
        if not self.forward and image_fname.name != kspace_fname.name:
            raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            
        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)


def create_data_loaders(data_path, args, shuffle=False, isforward=False):
    if not isforward:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward
    )

    sampler = None
    if shuffle and getattr(args, "balance_accelerations", False):
        if args.batch_size != 1:
            raise ValueError(
                "Acceleration-balanced hard routing requires batch_size=1."
            )
        sampler = BalancedAccelerationSampler(
            data_storage,
            mode=getattr(args, "acceleration_balance_mode", "oversample"),
            seed=getattr(args, "seed", 430),
        )
        shuffle = False

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=getattr(args, 'num_workers', 0),
        pin_memory=getattr(args, 'pin_memory', False),
        persistent_workers=getattr(args, 'num_workers', 0) > 0,
    )
    return data_loader
