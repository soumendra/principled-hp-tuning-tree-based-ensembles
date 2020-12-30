from src.datasets.dataset import Dataset  # type:ignore

z = Dataset()

for name in z.dataset_names:
    z.fetch_dataset(name)
