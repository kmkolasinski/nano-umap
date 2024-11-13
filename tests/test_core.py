import nano_umap
import numpy as np


def test_nano_umap() -> None:

    reducer = nano_umap.NanoUMAPV4()
    data = np.random.randn(2000, 128)
    embedding = reducer.fit_transform(data)
