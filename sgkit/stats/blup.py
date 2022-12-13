from typing import Hashable, Optional, Sequence, Union

import dask.array as da
import numpy as np
import xarray as xr
from xarray import Dataset


def _design_BLUP(trait, heritability, inverse_relationship, fixed=None):
    assert trait.ndim == 1
    n_samples = len(trait)
    assert np.shape(heritability) == ()
    assert inverse_relationship.shape == (n_samples, n_samples)
    # convert to dask
    trait = da.asarray(trait)
    inverse_relationship = da.asarray(inverse_relationship)
    if fixed is not None:
        fixed = da.asarray(fixed)
        assert fixed.ndim == 2
        assert fixed.shape[0] == n_samples
    # compute design matrices
    # note that we zero-out samples with missing phenotypes rather
    # than removing them from a given axis (as in commonly notation)
    # this avoids computing an additional dimension for phenotyped samples
    alpha = (1 - heritability) / heritability
    not_nan = ~np.isnan(trait)
    Z = da.eye(n_samples) * not_nan[:, None]
    ZtZ = Z
    Y = da.nan_to_num(trait)[:, None]
    ZtY = Y
    # case of no fixed effects
    if fixed is None:
        A = ZtZ + inverse_relationship * alpha
        B = ZtY
        return A, B
    # case of fixed effects
    X = fixed * not_nan[:, None]
    XtX = X.T @ X
    XtZ = X.T @ Z
    ZtX = XtZ.T
    XtY = X.T @ Y
    A = da.vstack(
        [
            da.hstack([XtX, XtZ]),
            da.hstack([ZtX, ZtZ + inverse_relationship * alpha]),
        ]
    )
    B = da.vstack([XtY, ZtY])
    return A, B


def blup(
    ds: Dataset,
    *,
    traits: Hashable,
    heritability: Hashable,
    inverse_relationship: Hashable,
    fixed: Optional[Union[Hashable, Sequence[Hashable]]] = None,
    merge: bool = True,
    **kwargs: Optional[dict],
) -> Dataset:
    # traits
    traits = da.asarray(ds[traits].data)
    if traits.ndim == 1:
        # samples as inner dimension
        traits = traits[None, :]
    else:
        traits = traits.T
    n_traits, n_samples = traits.shape

    # heritability
    heritability = da.asarray(ds[heritability].data)
    if heritability.shape == ():
        heritability = heritability[None]
    assert heritability.shape == (n_traits,)
    # inverse relationship
    inverse_relationship = da.asarray(ds[inverse_relationship].data)
    assert inverse_relationship.shape == (n_samples, n_samples)
    # fixed effects
    if fixed is not None:
        if isinstance(fixed, Hashable):
            fixed = [fixed]
        fixed_arrays = []
        for variable in fixed:
            array = da.asarray(ds[variable].data)
            if array.ndim == 1:
                array = array[:, None]
            assert array.shape[0] == n_samples
            fixed_arrays.append(array)
        fixed = da.concatenate(fixed_arrays, axis=1)
        n_fixed = fixed.shape[1]
        assert fixed.shape == (n_samples, n_fixed)
    else:
        n_fixed = 0
    # solve blups
    Xs = []
    for trait, h2 in zip(traits, heritability):
        A, B = _design_BLUP(trait, h2, inverse_relationship, fixed)
        A = A.rechunk(-1)
        B = B.rechunk(-1)
        Xs.append(da.linalg.solve(A, B, **kwargs))
    Xs = da.squeeze(da.asarray(Xs), axis=-1).T
    new = xr.Dataset()
    new["sample_blups"] = ["samples", "traits"], Xs[n_fixed:]
    if n_fixed > 0:
        new["fixed_blups"] = ["fixed_effects", "traits"], Xs[0:n_fixed]
    return new
