import dask.array as da
import numpy as np
from numba import guvectorize, vectorize
from xarray import Dataset

from sgkit import variables
from sgkit.typing import ArrayLike
from sgkit.utils import conditional_merge_datasets, create_dataset


@vectorize("int64(int64, int64)", nopython=True, cache=True)  # type: ignore
def _greatest_common_denominatior(x: int, y: int) -> int:  # pragma: no cover
    while y != 0:
        t = x % y
        x = y
        y = t
    return x


@vectorize("int64(int64, int64)", nopython=True, cache=True)  # type: ignore
def _comb(n: int, k: int) -> int:  # pragma: no cover
    if k > n:
        return 0
    r = 1
    for d in range(1, k + 1):
        gcd = _greatest_common_denominatior(r, d)
        r //= gcd
        r *= n
        r //= d // gcd
        n -= 1
    return r


@vectorize("int64(int64, int64)", nopython=True, cache=True)  # type: ignore
def _comb_with_replacement(n: int, k: int) -> int:  # pragma: no cover
    n = n + k - 1
    return _comb(n, k)


@guvectorize(  # type: ignore
    [
        "void(int8[:], int64[:])",
        "void(int16[:], int64[:])",
        "void(int32[:], int64[:])",
        "void(int64[:], int64[:])",
    ],
    "(k)->()",
    nopython=True,
    cache=True,
)
def _genotype_as_index(g: ArrayLike, out: ArrayLike) -> None:  # pragma: no cover
    """Convert genotypes to the index of their array position
    following the VCF specification for fields of length G.

    Parameters
    ----------
    g
        Genotype call of shape (ploidy,) containing alleles encoded as
        type `int` with values of -1 indicating a missing allele and
        values of -2 indicating non alleles.

    Returns
    -------
    i
        Index of genotype following the sort order described in the
        VCF spec.
    """
    out[0] = 0
    for i in range(len(g)):
        a = g[i]
        if a >= 0:
            out[0] += _comb_with_replacement(a, i + 1)
        elif a == -1:
            raise ValueError("Partial genotypes cannot be converted to an index.")


@guvectorize(  # type: ignore
    [
        "void(int8, int8, int8[:], int8[:])",
        "void(int16, int16, int8[:], int8[:])",
        "void(int32, int32, int8[:], int8[:])",
        "void(int64, int64, int8[:], int8[:])",
    ],
    "(),(),(k)->(k)",
    nopython=True,
    cache=True,
)
def _index_as_genotype(
    i: int, k: int, _: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover
    """Convert the index of a genotype sort position to the
    genotype call indicated by that index following the VCF
    specification for fields of length G.

    Parameters
    ----------
    i
        Index of genotype following the sort order described in the
        VCF spec. An index less than 0 is invalid and will return an
        uncalled genotype.
    k
        Ploidy of the genotype call.
    _
        Dummy variable of type `uint8` and shape (ploidy,) used to define
        the size of the resulting array.
        Parameter k may be smaller than the length of this array in the
        case of mixed ploidy data.

    Returns
    -------
    g
        Genotype call of shape (ploidy,) containing alleles encoded as
        type `int` with values of -1 indicating a missing allele and
        values of -2 indicating non alleles.
    """
    out[:] = -2
    if i < 0:
        # handle non-call
        out[:k] = -1
        return
    remainder = i
    for i in range(k):
        # find allele n for position k
        p = k - i
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


@guvectorize(  # type: ignore
    [
        "void(float64[:], uint8[:], float64, int8[:])",
        "void(float32[:], uint8[:], float64, int8[:])",
    ],
    "(p),(k),()->(k)",
    nopython=True,
    cache=True,
)
def _convert_probability_to_call(
    gp: ArrayLike, _: ArrayLike, threshold: float, out: ArrayLike
) -> None:  # pragma: no cover
    """Generalized U-function for converting genotype probabilities to hard calls

    Parameters
    ----------
    gp
        Genotype probabilities of shape (genotypes,) containing unphased, biallelic
        probabilities in the order homozygous reference, heterozygous, homozygous alternate.
    _
        Dummy variable of type `uint8` and shape (ploidy,) used to define
        the ploidy of the resulting array
    threshold
        Probability threshold that must be met or exceeded by at least one genotype
        probability in order for any calls to be made -- all values will be -1 (missing)
        otherwise. Setting this value to less than 0 disables any effect it has.
    out
        Hard calls array of shape (ploidy,).
    """
    # Ignore singleton array inputs used for metadata inference by dask
    if gp.shape[0] == 1 and out.shape[0] == 1:
        return
    if gp.shape[0] != 3 or out.shape[0] != 2:
        raise NotImplementedError(
            "Hard call conversion only supported for diploid, biallelic genotypes."
        )
    out[:] = -1  # (ploidy,)
    # Return no call if any probability is absent
    if np.any(np.isnan(gp)):
        return
    i = np.argmax(gp)
    # Return no call if max probability does not exceed threshold
    if threshold > 0 and gp[i] < threshold:
        return
    # Return no call if max probability is not unique
    if (gp[i] == gp).sum() > 1:
        return
    # Homozygous reference
    if i == 0:
        out[:] = 0
    # Heterozygous
    elif i == 1:
        out[0] = 1
        out[1] = 0
    # Homozygous alternate
    else:
        out[:] = 1


def convert_probability_to_call(
    ds: Dataset,
    call_genotype_probability: str = variables.call_genotype_probability,
    threshold: float = 0.9,
    merge: bool = True,
) -> Dataset:
    """

    Parameters
    ----------
    ds
        Dataset containing genotype probabilities, such as from :func:`sgkit.io.bgen.read_bgen`.
    call_genotype_probability
        Genotype probability variable to be converted as defined by
        :data:`sgkit.variables.call_genotype_probability_spec`.
    threshold
        Probability threshold in [0, 1] that must be met or exceeded by at least one genotype
        probability in order for any calls to be made -- all values will be -1 (missing)
        otherwise. Setting this value to less than or equal to 0 disables any effect it has.
        Default value is 0.9.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing the following variables:

    - `call_genotype` (variants, samples, ploidy): Converted hard calls.
        Defined by :data:`sgkit.variables.call_genotype_spec`.

    - `call_genotype_mask` (variants, samples, ploidy): Mask for converted hard calls.
        Defined by :data:`sgkit.variables.call_genotype_mask_spec`.
    """
    if not (0 <= threshold <= 1):
        raise ValueError(f"Threshold must be float in [0, 1], not {threshold}.")
    variables.validate(
        ds, {call_genotype_probability: variables.call_genotype_probability_spec}
    )
    if ds.dims["genotypes"] != 3:
        raise NotImplementedError(
            f"Hard call conversion only supported for diploid, biallelic genotypes; "
            f"num genotypes in provided probabilities array = {ds.dims['genotypes']}."
        )
    GP = da.asarray(ds[call_genotype_probability])
    # Remove chunking in genotypes dimension, if present
    if len(GP.chunks[2]) > 1:
        GP = GP.rechunk((None, None, -1))
    K = da.empty(2, dtype=np.uint8)
    GT = _convert_probability_to_call(GP, K, threshold)
    new_ds = create_dataset(
        {
            variables.call_genotype: (("variants", "samples", "ploidy"), GT),
            variables.call_genotype_mask: (("variants", "samples", "ploidy"), GT < 0),
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)
