"""Regression tests for public-data conversion and label/time semantics."""
import numpy as np
import pandas as pd
import pytest

from scripts.data.prepare_financial_data import (
    canonical_binance, contiguous_segments, interval_labels,
    meme_quote_multiplier, rescale_quote, weak_event_labels,
)
from scripts.data.validate_financial_data import validate


def bars(n=8):
    return pd.DataFrame({'timestamps': pd.date_range('2025-01-01', periods=n, freq='5min', tz='UTC'),
                         'open': np.ones(n)*10, 'high': np.ones(n)*11,
                         'low': np.ones(n)*9, 'close': np.ones(n)*10.5,
                         'volume': np.ones(n)*2, 'amount': np.ones(n)*21})


@pytest.mark.parametrize('unit', ['ms', 'us'])
def test_binance_timestamp_epoch_units(unit):
    expected = pd.Timestamp('2026-10-31T00:00:00Z')
    ticks = expected.value // (1000000 if unit == 'ms' else 1000)
    data = bars(1).drop(columns='timestamps')
    data['open_time'] = ticks
    result = canonical_binance(data)
    assert result.timestamps.iloc[0] == expected
    assert result.amount.iloc[0] == 21


def test_source_gaps_are_split_without_inventing_bars():
    data = bars(8).drop(index=4)
    result = contiguous_segments(data, '5min', minimum_rows=3)
    assert [len(x) for x in result] == [4, 3]
    assert sum(map(len,result)) == 7
    assert all(x.timestamps.diff().iloc[1:].eq(pd.Timedelta('5min')).all() for x in result)
    data.loc[2, 'high'] = 0
    result = contiguous_segments(data, '5min', minimum_rows=3)
    assert len(result) == 1 and len(result[0]) == 3


def test_weak_event_labels_use_intersection_and_boundary_conventions():
    times = pd.Series(pd.date_range('2025-01-01 17:50', periods=11, freq='5min', tz='UTC'))
    event = pd.Timestamp('2025-01-01 18:00Z')
    onset, window = weak_event_labels(times, [event])
    assert onset.tolist() == [0,0,1,0,0,0,0,0,0,0,0]
    # 17:55 intersects the uncertainty interval; 18:30 is excluded.
    assert window.tolist() == [0,1,1,1,1,1,1,1,0,0,0]


def test_author_interval_labels_are_explicitly_inclusive():
    times = pd.Series(pd.date_range('2025-01-01', periods=6, freq='1h', tz='UTC'))
    events = pd.DataFrame({'pump_start':[times.iloc[1]],'dump_end':[times.iloc[3]]})
    assert interval_labels(times,events).tolist() == [0,1,1,1,0,0]


def test_quote_unit_conversion_preserves_base_quantity_and_is_reversible():
    data = bars()
    scaled = rescale_quote(data, 1e8)
    np.testing.assert_array_equal(scaled.volume, data.volume)
    for field in ('open','high','low','close','amount'):
        np.testing.assert_allclose(scaled[field] / 1e8, data[field])
    np.testing.assert_allclose(scaled.amount / scaled.close, data.amount / data.close)


def test_meme_quote_scale_is_fixed_from_initial_context_only():
    first = bars(100)
    first[['open','high','low','close']] *= 1e-14
    second = first.copy()
    second.loc[64:, 'open'] *= 1e10
    assert meme_quote_multiplier(first) == meme_quote_multiplier(second)
    scaled = rescale_quote(first, meme_quote_multiplier(first))
    assert 99 <= scaled.open.iloc[:64].median() <= 1001


def test_committed_data_checksums_schema_continuity_and_splits():
    result = validate('data/financial_anomalies')
    assert result['validated_series'] >= 50
    assert result['validated_prepared_rows'] > 90000
