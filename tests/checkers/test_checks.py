"""Unit tests for the reusable checks, on synthetic arrays."""

import numpy as np
import pytest

from embodied_brain_collect.checkers import checks as C


def levels(out):
    return [f.level for f in out.findings]


def messages(out):
    return " | ".join(f.message for f in out.findings)


# =============================================================================
# Numeric helpers
# =============================================================================

def test_runs_encodes_true_stretches():
    assert C.runs(np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], bool)) == [
        (1, 2), (4, 1), (7, 3)]
    assert C.runs(np.zeros(0, bool)) == []
    assert C.runs(np.ones(3, bool)) == [(0, 3)]


def test_mad_outliers_flags_gross_values_against_noise():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 1000)
    a[::100] = 500.0
    assert C.mad_outliers(a)["fraction"] == pytest.approx(0.01, abs=1e-9)


def test_mad_outliers_survives_degenerate_input():
    assert C.mad_outliers(np.empty(0))["n"] == 0
    # A constant array has MAD 0; the std fallback must not divide by zero.
    assert C.mad_outliers(np.ones(50))["fraction"] == 0.0
    assert C.mad_outliers(np.array([1.0, np.nan, np.inf]))["n"] == 1


def test_window_counts_excludes_partial_edge_windows():
    """Regression: a grid anchored on floor(t0) makes the leading fragment
    look like a rate drop on every stream."""
    t = np.arange(10.5, 20.5, 0.1)          # 100 samples, starts mid-second
    starts, counts = C.window_counts(t, 1.0)
    assert starts[0] == pytest.approx(10.5)  # anchored on the data, not 10.0
    # Every window is full; the old floor(t0) grid made the first one a
    # ~5-sample fragment that then read as a 50% rate drop.
    assert counts.min() >= 9
    assert counts[0] >= 9 and counts[-1] >= 9


def test_window_counts_sliding_overlaps():
    t = np.arange(0, 10, 0.1)
    _, plain = C.window_counts(t, 1.0)
    _, slide = C.window_counts(t, 1.0, 0.5)
    assert len(slide) > len(plain)
    assert set(slide.tolist()) == {10}


def test_window_counts_empty_input():
    assert C.window_counts(np.empty(0))[1].size == 0
    assert C.window_counts(np.array([1.0]))[1].size == 0


# =============================================================================
# Timestamp checks
# =============================================================================

def test_clean_series_is_silent(ctx):
    c = ctx(series={"s": np.arange(0, 10, 0.01)})
    for check in C.ts_checks("s"):
        assert check.run(c).findings == [], check.name


def test_sanity_reports_duplicates_and_backsteps(ctx):
    c = ctx(series={"s": np.array([0.0, 1.0, 1.0, 0.5, 2.0])})
    out = C.TimestampSanity(series="s").run(c)
    assert levels(out) == ["WARN", "WARN"]
    assert "回退" in messages(out) and "重复" in messages(out)


def test_sanity_distinguishes_empty_from_out_of_window(ctx):
    """Different faults: a dead sensor versus a clock that disagrees."""
    c = ctx(series={"s": np.empty(0)})
    assert "没有任何样本" in messages(C.TimestampSanity(series="s").run(c))

    c = ctx(series={"s": np.arange(100.0, 110.0, 0.1)},
            window={"t0": 500.0, "t1": 600.0})
    out = C.TimestampSanity(series="s").run(c)
    assert levels(out) == ["ERROR"]
    assert "窗口之外" in messages(out)


def test_sanity_errors_when_the_series_is_absent(ctx):
    """无时间戳的流无法对齐到任何其他流 — 按 recorder 新策略这是错误。"""
    c = ctx(series={}, default="missing")
    out = C.TimestampSanity().run(c)
    assert levels(out) == ["ERROR"] and "缺少" in messages(out)


def test_gap_and_jump_locate_a_hole(ctx):
    t = np.r_[np.arange(0, 3, 0.01), np.arange(5, 8, 0.01)]   # 2 s hole
    c = ctx(series={"s": t})
    gap = C.TimestampGap(series="s").run(c)
    assert levels(gap) == ["ERROR"]        # missing data is unrecoverable
    assert gap.findings[0].observed == pytest.approx(2.0, abs=0.02)
    assert len(gap.findings[0].spans) == 1
    assert C.TimestampJump(series="s").run(c).findings[0].observed > 1.9


def test_gap_threshold_is_a_tenth_of_a_second(ctx):
    """At 30 fps the old 5x-median rule set the bar at 0.167 s, so a hole
    this size slipped through; the 0.1 s floor catches it."""
    t = np.r_[np.arange(0, 1, 1 / 30), np.arange(1.12, 2, 1 / 30)]
    out = C.TimestampGap(series="s").run(ctx(series={"s": t}))
    assert levels(out) == ["ERROR"]
    observed = out.findings[0].observed
    assert 0.1 < observed < 5 / 30      # missed by the old rule, caught now


def test_gap_ignores_normal_cadence(ctx):
    """The floor must stay above one sampling interval, or a slow stream
    reports its own cadence as a gap."""
    assert C.TimestampGap(series="s").run(
        ctx(series={"s": np.arange(0, 10, 1 / 30)})).findings == []


def test_jitter_only_fires_when_unstable(ctx):
    steady = ctx(series={"s": np.arange(0, 10, 0.01)})
    assert C.IntervalJitter(series="s").run(steady).findings == []
    rng = np.random.default_rng(1)
    jumpy = ctx(series={"s": np.cumsum(rng.exponential(0.01, 1000))})
    assert levels(C.IntervalJitter(series="s").run(jumpy)) == ["WARN"]


def test_burst_write_series_does_not_crash(ctx):
    """Every frame of a serial read shares one stamp — median interval 0."""
    c = ctx(series={"s": np.repeat(np.arange(0, 5, 0.1), 7)})
    for check in C.ts_checks("s"):
        check.run(c)          # must not raise
    assert "重复" in messages(C.TimestampSanity(series="s").run(c))


# =============================================================================
# Rate checks
# =============================================================================

def test_sliding_window_rate_finds_the_empty_second(ctx):
    t = np.r_[np.arange(0, 3, 0.01), np.arange(5, 8, 0.01)]
    out = C.SlidingWindowRate(series="s").run(ctx(series={"s": t}))
    assert any("没有任何样本" in f.message for f in out.findings)
    assert out.stats["empty_windows"] >= 1


def test_window_sample_count_needs_a_window(ctx):
    t = np.arange(0, 10, 0.001)
    assert C.WindowSampleCount(series="s").run(ctx(series={"s": t})).findings == []


def test_window_sample_count_catches_a_shortfall(ctx):
    """Half the samples missing, but evenly spread — the median interval
    still looks healthy, so only this comparison catches it."""
    t = np.arange(0.0, 10.0, 0.002)                     # 500 Hz over 10 s
    c = ctx(series={"s": t}, window={"t0": 0.0, "t1": 10.0})
    c.add_series("s", loader=lambda: t, expected_rate=1000.0)
    out = C.WindowSampleCount(series="s").run(c)
    assert levels(out) == ["WARN"]
    assert out.findings[0].observed == pytest.approx(-0.5, abs=0.01)


# =============================================================================
# Data checks
# =============================================================================

def test_dead_channel(ctx):
    a = np.c_[np.random.default_rng(0).normal(size=50), np.ones(50)]
    out = C.DeadChannel("x").run(ctx(arrays={"x": a}))
    assert out.stats["dead_channels"] == [1]


def test_nan_fraction_and_bounds(ctx):
    a = np.ones((100, 3)); a[:20] = np.nan
    assert C.NanFraction("x").run(ctx(arrays={"x": a})).findings[0].observed \
        == pytest.approx(0.2)
    out = C.ValueBounds("x", lo=0, hi=10).run(
        ctx(arrays={"x": np.array([1.0, 5.0, 99.0, -3.0])}))
    assert out.stats["n_out_of_bounds"] == 2


def test_value_jump_per_node(ctx):
    a = np.zeros((10, 4, 3))
    a[5, 2] = [1.0, 0.0, 0.0]        # one node teleports and comes back
    out = C.ValueJump("x", thr=0.2, per_node=True).run(ctx(arrays={"x": a}))
    assert out.stats["n_jumps"] == 2


def test_checks_skip_missing_fields(ctx):
    c = ctx(arrays={})
    for check in (C.MadOutlier("nope"), C.DeadChannel("nope"),
                  C.NanFraction("nope"), C.ValueBounds("nope"),
                  C.ValueJump("nope")):
        assert not check.applies(c), check.name
