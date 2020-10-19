import numpy as np
import xarray as xr

from sgkit.preprocessing import filter_partial_calls
from sgkit.testing import simulate_genotype_call_dataset


def test_filter_partial_calls():
    calls = np.array([[[0, 0], [0, 1], [1, 0]], [[-1, 0], [0, -1], [-1, -1]]])
    ds = simulate_genotype_call_dataset(*calls.shape)
    dims = ds["call_genotype"].dims
    ds["call_genotype"] = xr.DataArray(calls, dims=dims)
    ds["call_genotype_mask"] = xr.DataArray(calls < 0, dims=dims)

    ds2 = filter_partial_calls(ds)

    calls_filtered = ds2["call_genotype_complete"]
    mask_filtered = ds2["call_genotype_complete_mask"]

    np.testing.assert_array_equal(
        calls_filtered,
        np.array([[[0, 0], [0, 1], [1, 0]], [[-1, -1], [-1, -1], [-1, -1]]]),
    )

    np.testing.assert_array_equal(
        mask_filtered,
        np.array([[[0, 0], [0, 0], [0, 0]], [[1, 1], [1, 1], [1, 1]]], dtype=np.bool),
    )
