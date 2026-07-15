"""Known-answer tests for the model evaluation metrics."""
import numpy as np
import pytest

from pytRIBS.results.evaluate import Evaluate

OBSERVED = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def test_perfect_simulation_scores():
    simulated = OBSERVED.copy()
    assert Evaluate.nash_sutcliffe(OBSERVED, simulated) == pytest.approx(1.0)
    assert Evaluate.kling_gupta_efficiency(OBSERVED, simulated) == pytest.approx(1.0)
    assert Evaluate.root_mean_squared_error(OBSERVED, simulated) == pytest.approx(0.0)
    assert Evaluate.percent_bias(OBSERVED, simulated) == pytest.approx(0.0)


def test_constant_offset_scores():
    simulated = OBSERVED + 1.0  # uniform overestimation by 1
    # NSE = 1 - 5/10
    assert Evaluate.nash_sutcliffe(OBSERVED, simulated) == pytest.approx(0.5)
    assert Evaluate.root_mean_squared_error(OBSERVED, simulated) == pytest.approx(1.0)
    # Overestimation is negative PBIAS under this sign convention
    assert Evaluate.percent_bias(OBSERVED, simulated) == pytest.approx(-100.0 / 3.0)
    # r = 1, alpha = 1, beta = 4/3 -> KGE = 1 - 1/3
    assert Evaluate.kling_gupta_efficiency(OBSERVED, simulated) == pytest.approx(2.0 / 3.0)
