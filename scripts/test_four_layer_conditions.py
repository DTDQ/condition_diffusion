from datetime import datetime
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from four_layer_conditions import (  # noqa: E402
    Event, FEATURE_NAMES, aggregate_events, bounded_surprise,
    filter_point_in_time,
)


BASE = {
    "event_type": "earnings",
    "layer": "company",
    "direction": 1.0,
    "magnitude": 0.8,
    "exposure": 1.0,
    "horizon_days": 30,
    "novelty": 0.9,
    "source_quality": 1.0,
    "evidence_confidence": 0.9,
    "published_at": "2026-07-10T14:00:00+08:00",
    "source_type": "exchange_announcement",
    "source_ids": ["doc-1"],
}


class FourLayerConditionsTest(unittest.TestCase):
    def test_schema_has_48_unique_features(self):
        self.assertEqual(48, len(FEATURE_NAMES))
        self.assertEqual(48, len(set(FEATURE_NAMES)))

    def test_future_and_duplicate_are_removed(self):
        now = datetime.fromisoformat("2026-07-10T15:00:00+08:00")
        accepted = filter_point_in_time([
            Event.from_mapping(BASE),
            Event.from_mapping({**BASE, "evidence_confidence": 0.8}),
            Event.from_mapping({**BASE, "source_ids": ["doc-2"],
                                "published_at": "2026-07-10T16:00:00+08:00"}),
        ], now)
        self.assertEqual(1, len(accepted))
        self.assertEqual(0.9, accepted[0].evidence_confidence)

    def test_event_is_decayed_and_bounded(self):
        now = datetime.fromisoformat("2026-07-25T14:00:00+08:00")
        result = aggregate_events([Event.from_mapping(BASE)], now)
        self.assertGreater(result["event_earnings"], 0)
        self.assertLess(result["event_earnings"], 1)

    def test_missing_consensus_stays_missing(self):
        self.assertIsNone(bounded_surprise(10.0, None))
        self.assertGreater(bounded_surprise(12.0, 10.0), 0)


if __name__ == "__main__":
    unittest.main()
