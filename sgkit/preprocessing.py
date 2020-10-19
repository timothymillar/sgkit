from typing import Hashable

import xarray as xr
from xarray import Dataset

from sgkit import variables
from sgkit.utils import conditional_merge_datasets


def filter_partial_calls(
    ds: Dataset,
    *,
    call_genotype: Hashable = variables.call_genotype,
    merge: bool = True,
) -> Dataset:
    """Replace partial genotype calls with missing values.

    Parameters
    ----------
    ds
        Genotype call dataset such as from
        :func:`sgkit.create_genotype_call_dataset`.
    call_genotype
        Input variable name holding call_genotype as defined by
        :data:`sgkit.variables.call_genotype_spec`
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    Dataset containing `call_genotype_complete` and
    `call_genotype_complete_mask` in which partial genotype calls are
    replaced with compleately missing genotype calls.

    Examples
    --------

    >>> import sgkit as sg
    >>> from sgkit.testing import simulate_genotype_call_dataset
    >>> ds = simulate_genotype_call_dataset(n_variant=4, n_sample=2, seed=1, missing_pct=0.3)
    >>> sg.display_genotypes(ds) # doctest: +NORMALIZE_WHITESPACE
    samples    S0   S1
    variants
    0         ./0  ./.
    1         ./0  1/1
    2         0/1  ./0
    3         ./0  0/0

    >>> ds2 = filter_partial_calls(ds)
    >>> ds2['call_genotype'] = ds2['call_genotype_complete']
    >>> ds2['call_genotype_mask'] = ds2['call_genotype_complete_mask']
    >>> sg.display_genotypes(ds2) # doctest: +NORMALIZE_WHITESPACE
    samples    S0   S1
    variants
    0         ./.  ./.
    1         ./.  1/1
    2         0/1  ./.
    3         ./.  0/0
    """
    variables.validate(ds, {call_genotype: variables.call_genotype_spec})
    G = ds[call_genotype]
    P = (G < 0).any(axis=-1)
    F = xr.where(P, -1, G)  # type: ignore[no-untyped-call]
    new_ds = Dataset(
        {
            variables.call_genotype_complete: F,
            variables.call_genotype_complete_mask: F < 0,
        }
    )
    return conditional_merge_datasets(ds, variables.validate(new_ds), merge)
