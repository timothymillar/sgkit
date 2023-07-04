import math

import numpy as np

from sgkit.accelerate import numba_guvectorize, numba_jit
from sgkit.typing import ArrayLike


@numba_jit(nogil=True)
def _biallelic_genotype_index(genotype: ArrayLike) -> int:
    index = 0
    for i in range(len(genotype)):
        a = genotype[i]
        if a < 0:
            if a < -1:
                raise ValueError("Mixed-ploidy genotype indicated by allele < -1")
            return -1
        if a > 1:
            raise ValueError("Allele value > 1")
        index += a
    return index


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:], int64[:])",
        "void(int16[:], int64[:])",
        "void(int32[:], int64[:])",
        "void(int64[:], int64[:])",
    ],
    "(k)->()",
)
def biallelic_genotype_call_index(
    genotype: ArrayLike, out: ArrayLike
) -> int:  # pragma: no cover
    out[0] = _biallelic_genotype_index(genotype)


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:,:], uint64[:], uint64[:])",
        "void(int16[:,:], uint64[:], uint64[:])",
        "void(int32[:,:], uint64[:], uint64[:])",
        "void(int64[:,:], uint64[:], uint64[:])",
    ],
    "(n, k),(g)->(g)",
)
def _count_biallelic_genotypes(
    genotypes: ArrayLike, _: ArrayLike, out: ArrayLike
) -> ArrayLike:  # pragma: no cover
    out[:] = 0
    for i in range(len(genotypes)):
        index = _biallelic_genotype_index(genotypes[i])
        if index >= 0:
            out[index] += 1


# implementation from github.com/PlantandFoodResearch/MCHap
# TODO: replace with math.comb when supported by numba
@numba_jit(nogil=True)
def _comb(n: int, k: int) -> int:
    if k > n:
        return 0
    r = 1
    for d in range(1, k + 1):
        gcd_ = math.gcd(r, d)
        r //= gcd_
        r *= n
        r //= d // gcd_
        n -= 1
    return r


_COMB_REP_LOOKUP = np.array(
    [[math.comb(max(0, n + k - 1), k) for k in range(11)] for n in range(11)]
)
_COMB_REP_LOOKUP[0, 0] = 0  # special case


@numba_jit(nogil=True)
def _comb_with_replacement(n: int, k: int) -> int:
    if (n < _COMB_REP_LOOKUP.shape[0]) and (k < _COMB_REP_LOOKUP.shape[1]):
        return _COMB_REP_LOOKUP[n, k]
    n = n + k - 1
    return _comb(n, k)


@numba_guvectorize(["void(int64, int64, int64[:])"], "(),()->()")  # type: ignore
def comb_with_replacement(
    n: ArrayLike, k: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover
    out[0] = _comb_with_replacement(n, k)


@numba_jit(nogil=True)
def _sorted_genotype_index(genotype: ArrayLike) -> int:
    # Warning: genotype alleles must be sorted in ascending order!
    if genotype[0] < 0:
        if genotype[0] < -1:
            raise ValueError("Mixed-ploidy genotype indicated by allele < -1")
        return -1
    index = 0
    for i in range(len(genotype)):
        a = genotype[i]
        index += _comb_with_replacement(a, i + 1)
    return index


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:], int64[:])",
        "void(int16[:], int64[:])",
        "void(int32[:], int64[:])",
        "void(int64[:], int64[:])",
    ],
    "(k)->()",
)
def sorted_genotype_call_index(
    genotype: ArrayLike, out: ArrayLike
) -> int:  # pragma: no cover
    # Warning: genotype alleles must be sorted in ascending order!
    out[0] = _sorted_genotype_index(genotype)


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:,:], uint64[:], uint64[:])",
        "void(int16[:,:], uint64[:], uint64[:])",
        "void(int32[:,:], uint64[:], uint64[:])",
        "void(int64[:,:], uint64[:], uint64[:])",
    ],
    "(n, k),(g)->(g)",
)
def _count_sorted_genotypes(
    genotypes: ArrayLike, _: ArrayLike, out: ArrayLike
) -> ArrayLike:  # pragma: no cover
    # Warning: genotype alleles must be sorted in ascending order!
    out[:] = 0
    for i in range(len(genotypes)):
        index = _sorted_genotype_index(genotypes[i])
        if index >= 0:
            out[index] += 1


@numba_guvectorize(  # type: ignore
    [
        "void(int64, int64, int8[:], int8[:])",
        "void(int64, int64, int16[:], int16[:])",
        "void(int64, int64, int32[:], int32[:])",
        "void(int64, int64, int64[:], int64[:])",
    ],
    "(),(),(k)->(k)",
)
def _index_as_genotype(
    index: int, ploidy: int, _: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover
    """Convert the integer index of a genotype to a
    genotype call following the VCF specification
    for fields of length G.

    Parameters
    ----------
    index
        Index of genotype following the sort order described in the
        VCF spec. An index less than 0 is invalid and will return an
        uncalled genotype.
    _
        Dummy variable of length ploidy. The dtype of this variable is
        used as the dtype of the returned genotype array.

    Returns
    -------
    genotype
        Integer alleles of the genotype call.
    """
    if index < 0:
        out[:ploidy] = -1
        out[ploidy:] = -2
        return
    remainder = index
    for index in range(ploidy):
        # find allele n for position k
        p = ploidy - index
        n = -1
        new = 0
        prev = 0
        while new <= remainder:
            n += 1
            prev = new
            new = _comb_with_replacement(n, p)
        n -= 1
        remainder -= prev
        out[p - 1] = n
    out[ploidy:] = -2


@numba_guvectorize(  # type: ignore
    [
        "void(float64[:], int64, float64, int64[:])",
        "void(float32[:], int64, float64, int64[:])",
    ],
    "(p),(),()->()",
)
def _convert_probabilities_to_index(
    probability: ArrayLike, n_genotypes: int, threshold: float, out: ArrayLike
) -> None:  # pragma: no cover
    # use only the valid part of the distribution
    n_probs = len(probability)
    if n_probs > n_genotypes:
        probs = probability[0:n_genotypes]
    else:
        probs = probability
    if np.any(np.isnan(probs)):
        # probabilities must not contain nans
        out[0] = -1
    else:
        index = np.argmax(probs)
        prob = probs[index]
        if prob >= threshold:
            if (probs == prob).sum() > 1:
                # max probability is not unique
                out[0] = -1
            else:
                out[0] = index
        else:
            # max probability is below threshold
            out[0] = -1
