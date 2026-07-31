import pytest

from scripts.recommend_final_epochs import recommend_final_epochs


def test_recommendation_uses_late_or_wall_clock_epoch_cost():
    result = recommend_final_epochs(
        budget_hours=10,
        reserve_fraction=0.1,
        calibration_wall_seconds=1800,
        late_epoch_seconds=800,
        calibration_epochs=2,
        max_epochs=100,
    )

    assert result["representative_epoch_seconds"] == pytest.approx(900)
    assert result["remaining_final_seconds"] == pytest.approx(30_600)
    assert result["recommended_final_epochs"] == 34


def test_recommendation_caps_the_fixed_horizon():
    result = recommend_final_epochs(
        budget_hours=480,
        reserve_fraction=0.05,
        calibration_wall_seconds=3600,
        late_epoch_seconds=120,
        calibration_epochs=2,
        max_epochs=100,
    )

    assert result["recommended_final_epochs"] == 100


@pytest.mark.parametrize(
    "updates",
    [
        {"budget_hours": 0},
        {"reserve_fraction": 1},
        {"calibration_wall_seconds": 0},
        {"late_epoch_seconds": 0},
        {"calibration_epochs": 0},
    ],
)
def test_recommendation_rejects_invalid_budget_inputs(updates):
    values = {
        "budget_hours": 480,
        "reserve_fraction": 0.05,
        "calibration_wall_seconds": 3600,
        "late_epoch_seconds": 1200,
        "calibration_epochs": 2,
        "max_epochs": 100,
    }
    values.update(updates)

    with pytest.raises(ValueError):
        recommend_final_epochs(**values)
