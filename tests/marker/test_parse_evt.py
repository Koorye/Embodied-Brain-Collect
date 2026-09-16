"""UDP marker 包解析(发送端时间戳字段)。"""

import pytest

from embodied_brain_collect.recorders.marker.udp_marker_recorder import _parse_evt


def test_parse_packet_with_sender_timestamp():
    evt = _parse_evt(
        b"EVT|trial=1|tag=RUN_START|code=241|t_eprime_ms=1730|t_sent_pc=1787054148.856")
    assert evt["code"] == 241
    assert evt["t_eprime_ms"] == 1730
    assert evt["t_sent_pc"] == pytest.approx(1787054148.856)


def test_parse_legacy_packet_without_sender_timestamp():
    evt = _parse_evt(b"EVT|trial=0|tag=FIX_ON|code=17|t_eprime_ms=1772")
    assert evt["code"] == 17
    assert evt["t_sent_pc"] is None          # 旧包无该字段,由调用方回退


def test_parse_rejects_garbage():
    assert _parse_evt(b"not a packet") is None
    assert _parse_evt(b"EVT|code=abc") is None
    assert _parse_evt(b"") is None
