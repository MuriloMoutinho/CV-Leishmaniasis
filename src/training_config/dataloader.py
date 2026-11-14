from torch.utils.data import DataLoader, ConcatDataset


def create_dataloader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

def set_transform(dataset, transform):
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            set_transform(ds, transform)
    else:
        dataset.transform = transform
