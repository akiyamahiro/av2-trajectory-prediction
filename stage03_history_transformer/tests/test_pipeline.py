"""Run from the project root: python -m unittest discover -s stage03_history_transformer/tests -v"""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import encode_dataframe, RejectedSample, scene_metrics, POS_SCALE
from model import HistoryTransformer, position_loss


def synthetic_frame(sid="synthetic", speed=6.0, acceleration=0.2, theta=0.7):
    t = (np.arange(110) - 49) * 0.1
    distance = speed*t + 0.5*acceleration*t*t
    heading = np.full(110, theta)
    direction = np.array([np.cos(theta), np.sin(theta)])
    xy = np.array([1234., -678.]) + distance[:, None]*direction
    vel = (speed + acceleration*t)[:, None]*direction
    return pd.DataFrame(dict(
        scenario_id=sid, focal_track_id="car", track_id="car", object_type="vehicle",
        timestep=np.arange(110), observed=np.arange(110)<50,
        position_x=xy[:, 0], position_y=xy[:, 1], heading=heading,
        velocity_x=vel[:, 0], velocity_y=vel[:, 1],
    ))


class GeometryAndDataTests(unittest.TestCase):
    def test_known_geometry_and_cv(self):
        sample = encode_dataframe(synthetic_frame())
        np.testing.assert_allclose(sample["hist"][-1, :2], 0, atol=1e-7)
        np.testing.assert_allclose(sample["hist"][-1, 2:], [.3, 0, 0, 1], atol=1e-7)
        np.testing.assert_allclose(sample["future_m"][-1], [39.6, 0], atol=1e-5)
        np.testing.assert_allclose(sample["cv_m"][-1], [36, 0], atol=1e-5)
        self.assertAlmostEqual(scene_metrics(sample["cv_m"], sample["future_m"])["FDE"], 3.6, places=4)

    def test_future_changes_do_not_change_model_inputs(self):
        frame = synthetic_frame()
        before = encode_dataframe(frame)
        mask = frame.timestep >= 50
        frame.loc[mask, ["position_x", "position_y"]] += 200
        frame.loc[mask, ["velocity_x", "velocity_y", "heading"]] = 99
        after = encode_dataframe(frame)
        for key in ("hist", "hist_valid", "origin", "theta", "cv_m"):
            np.testing.assert_array_equal(before[key], after[key])
        self.assertFalse(np.array_equal(before["future_m"], after["future_m"]))

    def test_rigid_frame_change_preserves_local_features(self):
        frame = synthetic_frame()
        before = encode_dataframe(frame)
        angle = 1.2
        r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        frame[["position_x", "position_y"]] = frame[["position_x", "position_y"]].to_numpy() @ r.T + [500, 20]
        frame[["velocity_x", "velocity_y"]] = frame[["velocity_x", "velocity_y"]].to_numpy() @ r.T
        frame["heading"] += angle
        after = encode_dataframe(frame)
        for key in ("hist", "future_m", "cv_m"):
            np.testing.assert_allclose(before[key], after[key], atol=1e-5)

    def test_unsorted_rows_and_nonfocal_actor(self):
        focal = synthetic_frame()
        other = focal.iloc[:5].copy()
        other["track_id"] = "other"
        mixed = pd.concat([focal, other]).sample(frac=1, random_state=42)
        np.testing.assert_array_equal(encode_dataframe(focal)["hist"], encode_dataframe(mixed)["hist"])

    def test_reject_invalid_protocol(self):
        frames = [synthetic_frame().iloc[:-1], pd.concat([synthetic_frame(), synthetic_frame().iloc[:1]])]
        nonvehicle = synthetic_frame(); nonvehicle["object_type"] = "bus"
        wrong_mask = synthetic_frame(); wrong_mask.loc[50, "observed"] = True
        frames.extend([nonvehicle, wrong_mask])
        for frame in frames:
            with self.subTest(rows=len(frame)), self.assertRaises(RejectedSample):
                encode_dataframe(frame)

    def test_metrics_and_strict_miss_threshold(self):
        truth = np.zeros((60, 2))
        score = scene_metrics(np.tile([0., 2.], (60, 1)), truth)
        self.assertEqual(score, {"ADE": 2., "FDE": 2., "miss": 0})
        self.assertEqual(scene_metrics(np.tile([3., 4.], (60, 1)), truth), {"ADE": 5., "FDE": 5., "miss": 1})


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shape_mask_and_finite_gradient(self):
        torch.manual_seed(42)
        model = HistoryTransformer(d_model=32, heads=4, layers=1, dropout=0).eval()
        hist = torch.randn(2, 50, 6)
        valid = torch.ones(2, 50, dtype=torch.bool); valid[:, :5] = False
        pred = model(hist, valid)
        self.assertEqual(tuple(pred.shape), (2, 60, 2))
        changed = hist.clone(); changed[:, :5] = 100
        torch.testing.assert_close(pred, model(changed, valid), atol=1e-5, rtol=1e-5)
        loss = position_loss(pred, torch.zeros_like(pred), torch.ones(2, 60, dtype=torch.bool))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_can_learn_small_batch(self):
        torch.manual_seed(42)
        samples = [encode_dataframe(synthetic_frame(speed=v, acceleration=a)) for v, a in [(3, 0), (6, .2), (9, -.1), (12, .4)]]
        hist = torch.tensor(np.stack([s["hist"] for s in samples]))
        future = torch.tensor(np.stack([s["future_m"] for s in samples])) / POS_SCALE
        valid = torch.ones(4, 50, dtype=torch.bool)
        future_valid = torch.ones(4, 60, dtype=torch.bool)
        model = HistoryTransformer(d_model=32, heads=4, layers=1, dropout=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
        initial = position_loss(model(hist, valid), future, future_valid).item()
        for _ in range(120):
            optimizer.zero_grad()
            loss = position_loss(model(hist, valid), future, future_valid)
            loss.backward(); optimizer.step()
        final = position_loss(model(hist, valid), future, future_valid).item()
        self.assertLess(final, initial * .05)


if __name__ == "__main__":
    unittest.main()
