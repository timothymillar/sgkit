import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

import sgkit as sg
from sgkit.io.plink import read_plink

# This data was generated externally using Hail
# for 10 samples, 100 variants, and genotype calls
# that are missing in ~10% of cases.
# TODO: document and move code to central location
# (cf. https://github.com/sgkit-dev/sgkit-plink/pull/20#discussion_r466907811)
example_dataset_1 = "plink_sim_10s_100v_10pmiss"

# This data was generated by following https://github.com/sgkit-dev/sgkit/issues/947
example_dataset_2 = "example"


@pytest.fixture(params=[dict()])
def ds1(shared_datadir, request):
    path = shared_datadir / example_dataset_1
    return read_plink(path=path, bim_sep="\t", fam_sep="\t", **request.param)


@pytest.fixture(params=[dict()])
def ds2(shared_datadir, request):
    path = shared_datadir / example_dataset_2
    return read_plink(path=path, **request.param)


def test_read_multi_path(shared_datadir, ds1):
    path = shared_datadir / example_dataset_1
    ds2 = read_plink(
        bed_path=path.with_suffix(".bed"),
        bim_path=path.with_suffix(".bim"),
        fam_path=path.with_suffix(".fam"),
        bim_sep="\t",
        fam_sep="\t",
    )
    xr.testing.assert_equal(ds1, ds2)


def test_read_ids(ds1):
    assert ds1["sample_id"].values.tolist() == [
        "000",
        "001",
        "002",
        "003",
        "004",
        "005",
        "006",
        "007",
        "008",
        "009",
    ]
    assert ds1["variant_id"][:10].values.tolist() == [
        "1:1:G:CGCGCG",
        "1:2:ACT:G",
        "1:3:ACT:G",
        "1:4:G:CGCGCG",
        "1:5:G:CGCGCG",
        "1:6:ACT:G",
        "1:7:G:CGCGCG",
        "1:8:T:GTGG",
        "1:9:T:GTGG",
        "1:10:A:C",
    ]


def test_raise_on_both_path_types():
    with pytest.raises(
        ValueError,
        match="Either `path` or all 3 of `{bed,bim,fam}_path` must be specified but not both",
    ):
        read_plink(path="x", bed_path="x")


def test_fixlen_str_variable(ds1):
    assert ds1["sample_id"].dtype == np.dtype("<U3")
    assert ds1["variant_id"].dtype == np.dtype("<U13")
    assert ds1["variant_allele"].dtype == np.dtype("|S6")
    assert ds1["sample_family_id"].dtype == np.dtype("<U1")
    assert ds1["sample_maternal_id"].dtype == np.dtype("<U1")
    assert ds1["sample_paternal_id"].dtype == np.dtype("<U1")


def test_read_slicing(ds1):
    gt = ds1["call_genotype"]
    shape = gt.shape
    assert gt[:3].shape == (3,) + shape[1:]
    assert gt[:, :3].shape == shape[:1] + (3,) + shape[2:]
    assert gt[:3, :5].shape == (3, 5) + shape[2:]
    assert gt[:3, :5, :1].shape == (3, 5, 1)


@pytest.mark.parametrize("ds1", [dict(bim_int_contig=True)], indirect=True)
def test_read_int_contig(ds1):
    # Test contig parse as int (the value is always "1" in .bed for ds1)
    assert np.all(ds1["variant_contig"].values == 1)
    assert_array_equal(ds1["contig_id"], ["1"])


@pytest.mark.parametrize("ds1", [dict(bim_int_contig=False)], indirect=True)
def test_read_str_contig(ds1):
    # Test contig indexing as string (the value is always "1" in .bed for ds1)
    assert np.all(ds1["variant_contig"].values == 0)
    assert_array_equal(ds1["contig_id"], ["1"])


def test_read_call_values(ds1):
    # Validate a few randomly selected individual calls
    # (spanning all possible states for a call)
    idx = np.array(
        [
            [50, 7],
            [81, 8],
            [45, 2],
            [36, 8],
            [24, 2],
            [92, 9],
            [26, 2],
            [81, 0],
            [31, 8],
            [4, 9],
        ]
    )
    expected = np.array(
        [
            [1, 0],
            [1, 0],
            [0, 0],
            [0, 0],
            [-1, -1],
            [1, 1],
            [1, 1],
            [0, 0],
            [1, 1],
            [1, 1],
        ]
    )
    gt = ds1["call_genotype"].values
    actual = gt[tuple(idx.T)]
    np.testing.assert_equal(actual, expected)


def test_read_stat_call_rate(ds1):
    # Validate call rate for each sample
    sample_call_rates = (
        (ds1["call_genotype"] >= 0).max(dim="ploidy").mean(dim="variants").values
    )
    np.testing.assert_equal(
        sample_call_rates, [0.95, 0.9, 0.91, 0.87, 0.86, 0.83, 0.86, 0.87, 0.92, 0.92]
    )


def test_read_stat_alt_alleles(ds1):
    # Validate alt allele sum for each sample
    n_alt_alleles = (
        ds1["call_genotype"].clip(0, 2).sum(dim="ploidy").sum(dim="variants").values
    )
    np.testing.assert_equal(n_alt_alleles, [88, 85, 84, 80, 84, 75, 82, 76, 88, 81])


def test_allele_frequency(ds1):
    test_idx = np.array([0, 2, 5, 6, 7, 8])
    test_expected = np.array(
        ["G", "ACT", "ACT", "CGCGCG", "T", "T"], dtype=np.dtype("|S6")
    )
    ds_sub = sg.call_allele_frequencies(ds1.isel(variants=test_idx))
    minor_allele = ds_sub.call_allele_count.sum(dim="samples").argmin(dim="alleles")
    np.testing.assert_equal(
        ds_sub.variant_allele.values[np.arange(len(minor_allele)), minor_allele.values],
        test_expected,
    )


def test_allele_order(ds2):
    # check allele order: REF=A1, ALT=A2
    np.testing.assert_equal(
        ds2["variant_allele"].values.tolist(),
        [
            [b"A", b"G"],
            [b"T", b"C"],
        ],
    )

    # check allele frequencies across samples
    # should be consistent with `plink --file sgkit/tests/io/plink/data/example --freq`
    ds2 = sg.call_allele_frequencies(ds2)
    mean_af = ds2.call_allele_frequency.mean(dim="samples").values
    np.testing.assert_equal(
        mean_af,
        np.array([[0.3, 0.7], [0.4, 0.6]]),
    )


def test_count_a1_not_implemented(shared_datadir):
    path = shared_datadir / example_dataset_2
    with pytest.raises(NotImplementedError):
        read_plink(path=path, count_a1=True)
