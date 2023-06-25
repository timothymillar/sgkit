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
    if heritability == 0.0:
        alpha = 1
    else:
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
    inverse_relationship: Hashable,
    heritability: Optional[Hashable] = None,
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
    if heritability is None:
        heritability = np.array([0.0])
    else:
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


def blup_step_reference(trait, fixed, H, Hinv, sig2_e, sig2_a):
    not_nan = ~np.isnan(trait)
    # n = number of records = number of phenotyped samples
    # p = number of levels of fixed effects
    # q = number of levels of random effects = total samples
    n = not_nan.sum()
    q, p = fixed.shape
    assert len(trait) == q
    assert trait.ndim == 1
    q = len(trait)
    assert Hinv.shape == (q, q)
    if fixed is not None:
        assert fixed.ndim == 2
        assert fixed.shape[0] == q
    # compute design matrices
    # note that we zero-out samples with missing phenotypes rather
    # than removing them from a given axis (as in commonly notation)
    # this avoids computing an additional dimension for phenotyped samples
    Z = np.eye(q)[not_nan, :]
    assert Z.shape == (n, q)
    ZtZ = Z.T @ Z  # partially zeroed eye
    y = trait[not_nan, None]
    assert y.shape == (n, 1)
    ZtY = Z.T @ y  # trait with nans as zeros
    # case of no fixed effects
    # if fixed is None:
    #    C = ZtZ + Ainv * alpha
    #    B = ZtY
    # case of fixed effects
    X = fixed[not_nan]
    assert X.shape == (n, p)
    XtX = X.T @ X
    XtZ = X.T @ Z
    ZtX = XtZ.T
    XtY = X.T @ y
    alpha = sig2_e / sig2_a
    C = np.vstack(
        [
            np.hstack([XtX, XtZ]),
            np.hstack([ZtX, ZtZ + Hinv * alpha]),
        ]
    )
    b = np.vstack([XtY, ZtY])

    # solve sytem of matrices
    res = np.linalg.solve(C, b)
    b_hat = res[0:p]
    a_hat = res[p:]
    assert a_hat.shape == (q, 1)
    assert b_hat.shape == (p, 1)

    # calculate heritabilities for next iter
    G = H * sig2_a
    Ginv = Hinv / sig2_a  # double check this
    R = np.eye(5) * sig2_e
    Rinv = np.eye(5) * (1 / sig2_e)
    V = Z @ G @ Z.T + R
    # Vinv = np.linalg.inv(V) # (n, n)
    Vinv = Rinv - Rinv @ Z @ np.linalg.inv(Z.T @ Rinv @ Z + Ginv) @ Z.T @ Rinv
    P = (
        Vinv - Vinv @ X @ np.linalg.inv(X.T @ Vinv @ X) @ X.T @ Vinv
    )  # incorrect in Mrode, see Johnson 1995
    logdetV = np.linalg.slogdet(V)[1]
    logdetXVinvX = np.linalg.slogdet(X.T @ Vinv @ X)[1]
    yPy = y.T @ P @ y
    Cinv22 = np.linalg.inv(C)[p:, p:]  # C is design matrix
    llk = (
        1 / 2 * (-yPy - logdetV - logdetXVinvX)[0, 0]
    )  # is the alpha in Mrode an error?

    Ainf_00 = 0.5 * (
        (y - X @ b_hat - Z @ a_hat).T @ P @ (y - X @ b_hat - Z @ a_hat) / sig2_e**2
    )
    Ainf_11 = 0.5 * (a_hat.T @ Z.T @ P @ Z @ a_hat) / sig2_a**2
    Ainf_01 = (
        0.5 * (a_hat.T @ Z.T @ P @ (y - X @ b_hat - Z @ a_hat)) / (sig2_e * sig2_a)
    )  # mistake? sig4_a in Mrode
    Ainf = np.array([[Ainf_00[0, 0], Ainf_01[0, 0]], [Ainf_01[0, 0], Ainf_11[0, 0]]])
    Ainfinv = np.linalg.inv(Ainf)

    Binf0 = 0.5 * (
        (y - X @ b_hat - Z @ a_hat).T @ (y - X @ b_hat - Z @ a_hat) / sig2_e**2
        - (n - p - q) / sig2_e
        - np.trace(Cinv22 @ Hinv) / sig2_a
    )
    Binf1 = 0.5 * (
        a_hat.T @ Hinv @ a_hat / sig2_a**2
        - q / sig2_a
        + np.trace(Cinv22 @ Hinv) * sig2_e / sig2_a**2
    )
    Binf = np.array([[Binf0[0, 0], Binf1[0, 0]]]).T

    AinfinvBinf = Ainfinv @ Binf
    return b_hat, a_hat, AinfinvBinf[0, 0] + sig2_e, AinfinvBinf[1, 0] + sig2_a, llk
