from typing import Hashable

import dask.array as da
from xarray import Dataset

from sgkit import variables
from sgkit.utils import (
    conditional_merge_datasets,
    create_dataset,
    define_variable_if_absent,
)

from .aggregation import count_call_alleles


def estimate_kinship(
    ds: Dataset,
    *,
    call_allele_count: Hashable = variables.call_allele_count,
    merge: bool = True,
) -> Dataset:
    """Estimate kinship based on allelic matching as described in Weir and Goudet 2017 [1].

    Parameters
    ----------
    ds
        Genotype call dataset.
    call_allele_count
        Input variable name holding call_allele_count as defined by
        :data:`sgkit.variables.call_allele_count_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`count_call_alleles`.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.allelic_matching_sample_pairwise_beta_spec`
    of estimated pairwise coancestry relative to the average coancestry of all pairs of
    individuals in the dataset.
    The dimensions are named ``samples_0`` and ``samples_1``.

    References
    ----------
    [1] - Bruce, S. Weir, and Jérôme Goudet 2017.
    "A Unified Characterization of Population Structure and Relatedness."
    Genetics 206 (4): 2085-2103.
    """
    ds = define_variable_if_absent(
        ds, variables.call_allele_count, call_allele_count, count_call_alleles
    )
    variables.validate(ds, {call_allele_count: variables.call_allele_count_spec})

    ac = da.array(ds[call_allele_count])
    af = ac / ac.sum(axis=-1, keepdims=True)

    # pairwise probability of IBS between pair of random alleles
    p_ibs = (af[:, None, :, :] * af[:, :, None, :]).sum(axis=-1)
    called = ~da.isnan(p_ibs)

    # average between-individual matching in population is the
    # mean of p_ibs excluding the diagonal (self sharing) elements
    num = p_ibs.sum(axis=-1).sum(axis=-1) - da.diagonal(p_ibs, axis1=-2, axis2=-1).sum(
        axis=-1
    )
    denom = called.sum(axis=-1).sum(axis=-1) - da.diagonal(
        called, axis1=-2, axis2=-1
    ).sum(axis=-1)
    avg = num / denom

    # allele sharing estimate for kinship
    num2 = da.nansum(p_ibs - avg[..., None, None], axis=0)
    denom2 = da.nansum(1 - avg[..., None, None], axis=0)
    k = num2 / denom2

    new_ds = create_dataset(
        {
            variables.allelic_matching_sample_pairwise_beta: (
                ("samples_0", "samples_1"),
                k,
            )
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)
