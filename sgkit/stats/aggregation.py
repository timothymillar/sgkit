import dask.array as da
import numpy as np
import xarray as xr
import numba
from xarray import DataArray, Dataset


@numba.guvectorize([
    'void(numba.int8[:], numba.uint8[:], numba.uint8[:])',
    'void(numba.int16[:], numba.uint8[:], numba.uint8[:])',
    'void(numba.int32[:], numba.uint8[:], numba.uint8[:])',
    'void(numba.int64[:], numba.uint8[:], numba.uint8[:])',
    ], '(n),(k)->(k)', nopython=True)
def count_alleles(g, _, out):
    out[:] = 0
    n_allele = len(g)
    for i in range(n_allele):
        a = g[i]
        if a >= 0:
            out[a] += 1


def count_call_alleles(ds: Dataset) -> DataArray:
    """Compute per sample allele counts from genotype calls.

    Parameters
    ----------
    ds : Dataset
        Genotype call dataset such as from
        `sgkit.create_genotype_call_dataset`.

    Returns
    -------
    call_allele_count : DataArray
        Allele counts with shape (variants, samples, alleles) and values
        corresponding to the number of non-missing occurrences
        of each allele.

    Examples
    --------

    >>> import sgkit as sg
    >>> from sgkit.testing import simulate_genotype_call_dataset
    >>> ds = simulate_genotype_call_dataset(n_variant=4, n_sample=2, seed=1)
    >>> ds['call_genotype'].to_series().unstack().astype(str).apply('/'.join, axis=1).unstack() # doctest: +NORMALIZE_WHITESPACE
    samples 0   1
    variants
    0       1/0	1/0
    1       1/0	1/1
    2       0/1	1/0
    3       0/0	0/0

    >>> sg.count_call_alleles(ds).values # doctest: +NORMALIZE_WHITESPACE
    array([[[1, 1],
            [1, 1]],
    <BLANKLINE>
           [[1, 1],
            [0, 2]],
    <BLANKLINE>
           [[1, 1],
            [1, 1]],
    <BLANKLINE>
           [[2, 0],
            [2, 0]]], dtype=uint8)
    """
    n_alleles = ds.dims['alleles']
    G = da.asarray(ds['call_genotype'])
    shape = (G.chunks[0], G.chunks[1], n_alleles)
    K = da.empty(n_alleles, dtype=np.uint8)
    return xr.DataArray(
        da.map_blocks(count_alleles, G, K, chunks=shape, drop_axis=2, new_axis=2),
        dims=('variants', 'samples', 'alleles'),
        name='call_allele_count'
    )


def count_variant_alleles(ds: Dataset) -> DataArray:
    """Compute allele count from genotype calls.

    Parameters
    ----------
    ds : Dataset
        Genotype call dataset such as from
        `sgkit.create_genotype_call_dataset`.

    Returns
    -------
    variant_allele_count : DataArray
        Allele counts with shape (variants, alleles) and values
        corresponding to the number of non-missing occurrences
        of each allele.

    Examples
    --------

    >>> import sgkit as sg
    >>> from sgkit.testing import simulate_genotype_call_dataset
    >>> ds = simulate_genotype_call_dataset(n_variant=4, n_sample=2, seed=1)
    >>> ds['call_genotype'].to_series().unstack().astype(str).apply('/'.join, axis=1).unstack() # doctest: +NORMALIZE_WHITESPACE
    samples 0   1
    variants
    0       1/0	1/0
    1       1/0	1/1
    2       0/1	1/0
    3       0/0	0/0

    >>> sg.count_variant_alleles(ds).values # doctest: +NORMALIZE_WHITESPACE
    array([[2, 2],
           [1, 3],
           [2, 2],
           [4, 0]], dtype=uint64)
    """
    return (
        count_call_alleles(ds)
        .sum(dim='samples')
        .rename('variant_allele_count')
    )
