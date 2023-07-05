from math import lgamma
from typing import Hashable, Optional

import dask.array as da
import numpy as np
from xarray import Dataset

from sgkit import variables
from sgkit.accelerate import numba_guvectorize, numba_jit
from sgkit.stats.aggregation import infer_variant_allele_unique
from sgkit.stats.conversion_numba_fns import _comb_with_replacement
from sgkit.typing import ArrayLike
from sgkit.utils import (
    conditional_merge_datasets,
    create_dataset,
    define_variable_if_absent,
)


@numba_jit
def increment_allele_counts(allele_count: ArrayLike) -> None:
    n_allele = len(allele_count)
    for i in range(n_allele):
        if i == n_allele:
            # final genotype
            raise (ValueError, "Final genotype")
        ci = allele_count[i]
        if ci == 0:
            pass
        else:
            allele_count[i] = 0
            allele_count[i + 1] += 1
            allele_count[0] = ci - 1
            return


@numba_jit(nogil=True)
def _genotype_log_likelihood(
    allele_depth: ArrayLike,
    allele_count: ArrayLike,
    error_matrix: ArrayLike,
    ploidy: int,
    n_allele: int,
) -> float:
    llk = 0.0
    for i in range(n_allele):
        depth_i = allele_depth[i]
        if depth_i > 0:
            # probability of drawing allele i from genotype
            allele_prob = 0.0
            for j in range(n_allele):
                allele_prob += error_matrix[i, j] * allele_count[j] / ploidy
            llk += np.log(allele_prob) * depth_i
    return llk


@numba_guvectorize(  # type: ignore
    [
        "void(int64[:], int64, int64, float64[:,:], uint8[:], float64[:])",
    ],
    "(a),(),(),(a,a),(g)->(g)",
)
def _genotype_log_likelihoods(
    allele_depth: ArrayLike,
    n_allele: ArrayLike,
    ploidy: ArrayLike,
    error_matrix: ArrayLike,
    _: ArrayLike,
    out: ArrayLike,
) -> None:
    allele_count = np.zeros(len(allele_depth), np.int64)
    allele_count[0] = ploidy  # first genotype
    out[:] = np.nan
    n_genotype = _comb_with_replacement(n_allele, ploidy)
    for i in range(n_genotype):
        llk = _genotype_log_likelihood(
            allele_depth, allele_count, error_matrix, ploidy, n_allele
        )
        out[i] = llk
        increment_allele_counts(allele_count)


@numba_jit(nogil=True)
def log_allele_count_multinomial_prior(
    allele_count: ArrayLike, frequencies: ArrayLike, n_allele: int, ploidy: int
) -> float:
    log_num = lgamma(ploidy + 1)
    log_denom = 0.0
    for i in range(n_allele):
        count = allele_count[i]
        if count > 0:
            log_freq = np.log(frequencies[i])
            log_num += log_freq * count
            log_denom += lgamma(count + 1)
    return log_num - log_denom


@numba_jit(nogil=True)
def log_allele_count_dirmul_prior(
    allele_count: ArrayLike, alphas: ArrayLike, n_allele: int, ploidy: int
) -> float:
    # Dirichlet-multinomial
    # left side of equation in log space
    sum_alphas = alphas.sum()
    num = lgamma(ploidy + 1) + lgamma(sum_alphas)
    denom = lgamma(ploidy + sum_alphas)
    left = num - denom

    # right side of equation
    prod = 0.0  # log(1.0)
    for i in range(n_allele):
        count = allele_count[i]
        if count > 0:
            alpha = alphas[i]
            num = lgamma(count + alpha)
            denom = lgamma(count + 1) + lgamma(alpha)
            prod += num - denom

    # return as log probability
    return left + prod


@numba_guvectorize(  # type: ignore
    [
        "void(float64[:], int64, int64, float64, uint8[:], float64[:])",
    ],
    "(a),(),(),(),(g)->(g)",
)
def _genotype_log_priors(
    frequencies: ArrayLike,
    n_allele: ArrayLike,
    ploidy: ArrayLike,
    inbreeding: ArrayLike,
    _: ArrayLike,
    out: ArrayLike,
) -> None:
    allele_count = np.zeros(len(frequencies), np.int64)
    allele_count[0] = ploidy  # first genotype
    out[:] = np.nan
    n_genotype = _comb_with_replacement(n_allele, ploidy)
    if inbreeding == 0.0:
        for i in range(n_genotype):
            lprior = log_allele_count_multinomial_prior(
                allele_count, frequencies, n_allele, ploidy
            )
            out[i] = lprior
            increment_allele_counts(allele_count)
    else:
        alphas = frequencies * ((1 - inbreeding) / inbreeding)
        for i in range(n_genotype):
            lprior = log_allele_count_dirmul_prior(
                allele_count, alphas, n_allele, ploidy
            )
            out[i] = lprior
            increment_allele_counts(allele_count)


@numba_jit(nogil=True)
def add_log_prob(x: float, y: float):
    if x == y == -np.inf:
        return -np.inf
    if x > y:
        return x + np.log1p(np.exp(y - x))
    else:
        return y + np.log1p(np.exp(x - y))


@numba_guvectorize(  # type: ignore
    [
        "void(float64[:], float64[:])",
    ],
    "(n)->(n)",
)
def nan_norm_logs(array: ArrayLike, out: ArrayLike) -> None:
    total = -np.inf
    for i in range(len(array)):
        val = array[i]
        if not np.isnan(val):
            total = add_log_prob(total, val)
    for i in range(len(array)):
        out[i] = array[i] - total


def genotype_log_likelihoods(
    ds: Dataset,
    *,
    error_matrix: Hashable,  # (variants * alleles * alleles)
    allele_depth: Hashable = variables.call_allele_depth,  # (variants * samples * alleles)
    allele_unique: Hashable = variables.variant_allele_unique,
    mixed_ploidy: bool = False,
    call_ploidy: Hashable = variables.call_ploidy,
    merge: bool = True,
) -> Dataset:
    """Calculate log-likelihoods for all possible genotypes given
    observed allele depths.

    Parameters
    ----------
    ds
        Dataset containing allele depths.
    error_matrix
        Variable containing matrix of pairwise error probabilities for each combination
        of alleles. See the note for details.
    allele_depth
        Variable containing observed allele depths for each sample as defined by
        :data:`sgkit.variables.call_allele_depth_spec`.
    allele_unique
        Variable containing the number of unique alleles at each variant as defined by
        :data:`sgkit.variables.variant_allele_unique_spec`. If absent, this variable will be
        computed automatically using :func:`infer_variant_allele_unique`.
    mixed_ploidy
        Specify if genotypes should be called with variable ploidy levels.
    call_ploidy
        Variable containing the ploidy level of each call_genotype as defined by
        :data:`sgkit.variables.call_ploidy_spec`. This variable is only used when calling
        a mixed-ploidy genotypes.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.call_genotype_log_likelihood_spec`.
    """
    allele_depth = da.array(ds[allele_depth].data)
    _, _, n_allele = allele_depth.shape
    # get ploidy of each call and maximum ploidy of dataset
    if mixed_ploidy:
        variables.validate(ds, {call_ploidy: variables.call_ploidy_spec})
        call_ploidy = da.array(ds[call_ploidy].values)
        max_ploidy = ds.dims.get("ploidy", call_ploidy.max().compute())
    else:
        call_ploidy = ds.dims.get("ploidy", 2)
        max_ploidy = call_ploidy
    # get the number of alleles at each locus
    ds = define_variable_if_absent(
        ds,
        variables.variant_allele_unique,
        allele_unique,
        infer_variant_allele_unique,
    )
    variables.validate(ds, {allele_unique: variables.variant_allele_unique_spec})
    u_alleles = da.array(ds[variables.variant_allele_unique].data)[:, None]
    error_spec = variables.ArrayLikeSpec("", "", kind="f", ndim={2, 3})
    variables.validate(ds, {error_matrix: error_spec})
    error_matrix = da.array(ds[error_matrix].data)
    max_genotypes = _comb_with_replacement(n_allele, max_ploidy)
    G = np.empty(max_genotypes, np.uint8)
    llks = _genotype_log_likelihoods(
        allele_depth,
        u_alleles,
        call_ploidy,
        error_matrix,
        G,
    )
    dims = ["variants", "samples", "genotypes"]
    new_ds = create_dataset(
        {
            variables.call_genotype_log_likelihood: (dims, llks),
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)


def genotype_log_priors(
    ds: Dataset,
    *,
    inbreeding_prior: Optional[Hashable] = None,  # (variants * samples)
    allele_prior: Optional[Hashable] = None,  # (variants * samples * alleles)
    allele_unique: Hashable = variables.variant_allele_unique,
    mixed_ploidy: bool = False,
    call_ploidy: Hashable = variables.call_ploidy,
    merge: bool = True,
) -> Dataset:
    """Calculate log-prior probabilities for all possible genotypes.

    Parameters
    ----------
    ds
        Dataset containing prior information.
    inbreeding_prior
        Variable containing expected inbreeding coefficients for each call genotype.
        This may have shape (samples,) to set priors at the samples level, or it
        may have shape (variants, samples) to set a prior for each individual call.
    allele_prior
        Variable containing expected allele frequencies.
        This may have shape (variants, alleles) to set a population prior for all samples
        or shape (variants, samples, alleles) to provide individual priors.
        By default, a flat prior is used for allele frequencies.
    allele_unique
        Variable containing the number of unique alleles at each variant as defined by
        :data:`sgkit.variables.variant_allele_unique_spec`. If absent, this variable will be
        computed automatically using :func:`infer_variant_allele_unique`.
    mixed_ploidy
        Specify if genotypes should be called with variable ploidy levels.
    call_ploidy
        Variable containing the ploidy level of each call_genotype as defined by
        :data:`sgkit.variables.call_ploidy_spec`. This variable is only used when calling
        a mixed-ploidy genotypes.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.call_genotype_log_prior_spec`.
    """
    # get the number of alleles at each locus and the dimension size
    ds = define_variable_if_absent(
        ds,
        variables.variant_allele_unique,
        allele_unique,
        infer_variant_allele_unique,
    )
    variables.validate(ds, {allele_unique: variables.variant_allele_unique_spec})
    u_allele = da.array(ds[allele_unique].data)[:, None]
    n_allele = ds.dims.get("alleles") or u_allele.max().compute()
    # get ploidy of each call and maximum ploidy of dataset
    if mixed_ploidy:
        variables.validate(ds, {call_ploidy: variables.call_ploidy_spec})
        call_ploidy = da.array(ds[call_ploidy].values)
        max_ploidy = ds.dims.get("ploidy") or call_ploidy.max().compute()
    else:
        call_ploidy = ds.dims.get("ploidy", 2)
        max_ploidy = call_ploidy
    # get inbreeding prior
    if inbreeding_prior is None:
        inbreeding = 0.0
    else:
        # May be shape (samples,) or (variants * samples)
        inbreeding_spec = variables.ArrayLikeSpec(
            "",
            "",
            kind="f",
            dims=(
                {"variants", None},
                "samples",
            ),
        )
        variables.validate(ds, {inbreeding_prior: inbreeding_spec})
        inbreeding = da.array(ds[inbreeding_prior].data)
    if allele_prior is None:
        frequencies = da.full(u_allele, 1 / u_allele, float)
    else:
        frequencies = da.array(ds[allele_prior].data)
        if np.ndim(frequencies) == 2:
            # must be variants * alleles so add samples dim
            variables.validate(
                ds, {allele_prior: variables.variant_allele_frequency_spec}
            )
            frequencies = frequencies[:, None, :]
        else:
            variables.validate(ds, {allele_prior: variables.call_allele_frequency_spec})
    max_genotypes = _comb_with_replacement(n_allele, max_ploidy)
    G = np.empty(max_genotypes, np.uint8)
    lpriors = _genotype_log_priors(
        frequencies,
        u_allele,
        call_ploidy,
        inbreeding,
        G,
    )
    # broadcast priors to (variants * samples * genotypes) if needed
    shape = (ds.dims["variants"], ds.dims["samples"], max_genotypes)
    if lpriors.shape != shape:
        lpriors = da.broadcast_to(lpriors, shape)
    dims = ["variants", "samples", "genotypes"]
    new_ds = create_dataset(
        {
            variables.call_genotype_log_prior: (dims, lpriors),
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)


def genotype_posteriors(
    ds: Dataset,
    *,
    genotype_log_prior: Hashable = variables.call_genotype_log_prior,
    genotype_log_likelihood: Hashable = variables.call_genotype_log_likelihood,
    merge: bool = True,
) -> Dataset:
    """Calculate call genotype posterior distributions.

    Parameters
    ----------
    ds
        Dataset containing genotype priors and likelihoods.
    genotype_log_prior
        Variable containing log-transformed genotype prior probabilities
        as defined by :data:`sgkit.variables.call_genotype_log_prior_spec`.
    genotype_log_likelihood
        Variable containing log-transformed genotype likelihoods as defined
        by :data:`sgkit.variables.call_genotype_log_likelihood_spec`.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.call_genotype_log_prior_spec`.

    - `call_genotype_log_probability` (variants, samples, genotypes): Log-transformed
        call genotype posterior as defined by
        :data:`sgkit.variables.call_genotype_log_probability_spec`.

    - `call_genotype_probability` (variants, samples, genotypes): Call genotype posterior
        probabilities as defined by :data:`sgkit.variables.call_genotype_probability_spec`.
    """
    variables.validate(
        ds,
        {
            genotype_log_prior: variables.call_genotype_log_prior_spec,
            genotype_log_likelihood: variables.call_genotype_log_likelihood_spec,
        },
    )
    lprior = da.array(ds[genotype_log_prior].data)
    llk = da.array(ds[genotype_log_likelihood].data)
    lposterior = nan_norm_logs(lprior + llk)
    posterior = np.exp(lposterior)
    dims = ["variants", "samples", "genotypes"]
    new_ds = create_dataset(
        {
            variables.call_genotype_log_probability: (dims, lposterior),
            variables.call_genotype_probability: (dims, posterior),
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)
