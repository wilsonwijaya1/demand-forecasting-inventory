import numpy as np
import pandas as pd

from scm_forecast.data import N_SKU, PATTERN_COUNTS, demand_profile, generate_dataset


def test_generator_is_deterministic_for_a_fixed_seed():
    a, b = generate_dataset(seed=123), generate_dataset(seed=123)
    pd.testing.assert_frame_equal(a.demand, b.demand)
    pd.testing.assert_frame_equal(a.sku_master, b.sku_master)
    pd.testing.assert_frame_equal(a.lead_times, b.lead_times)


def test_different_seeds_give_different_data():
    a, b = generate_dataset(seed=1), generate_dataset(seed=2)
    assert not a.demand.equals(b.demand)


def test_shape_and_validity():
    ds = generate_dataset(seed=42)
    assert ds.demand.shape == (N_SKU, 72)
    assert ds.history.shape == (N_SKU, 60) and ds.holdout.shape == (N_SKU, 12)
    assert (ds.demand.to_numpy() >= 0).all()
    assert np.issubdtype(ds.demand.to_numpy().dtype, np.integer)
    assert ds.demand.index.is_unique
    assert (ds.lead_times.to_numpy() >= 1).all()
    assert (ds.sku_master["unit_cost"] > 0).all()


def test_pattern_mix_is_diverse_and_fixed():
    ds = generate_dataset(seed=7)
    assert ds.sku_master["pattern"].value_counts().to_dict() == PATTERN_COUNTS
    assert ds.sku_master["category"].nunique() == 5


def test_intermittent_skus_have_many_zero_months_others_few():
    ds = generate_dataset(seed=42)
    zero_share = (ds.history == 0).mean(axis=1)
    inter = ds.sku_master["pattern"] == "intermittent"
    assert zero_share[inter].min() > 0.3
    assert zero_share[~inter].max() < 0.1
    prof = demand_profile(ds.history)
    assert set(prof.loc[inter, "sbc_class"]) <= {"intermittent", "lumpy"}


def test_seasonal_skus_show_an_annual_cycle():
    ds = generate_dataset(seed=42)
    seasonal = ds.sku_master.index[ds.sku_master["pattern"] == "seasonal"]
    for sku in seasonal:
        y = ds.history.loc[sku].to_numpy(float)
        y = y - y.mean()
        acf12 = np.dot(y[12:], y[:-12]) / np.dot(y, y)
        assert acf12 > 0.2
