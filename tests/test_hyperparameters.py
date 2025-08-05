"""
Test hyperparameter configuration functionality.
"""

import pytest

from nupunkt.training import PRESETS, PunktHyperparameters, train_model


def test_hyperparameter_presets():
    """Test that presets work correctly."""
    # Check presets exist
    assert "conservative" in PRESETS
    assert "balanced" in PRESETS
    assert "aggressive" in PRESETS

    # Check threshold ordering
    assert (
        PRESETS["conservative"].sent_starter_threshold > PRESETS["balanced"].sent_starter_threshold
    )
    assert PRESETS["balanced"].sent_starter_threshold > PRESETS["aggressive"].sent_starter_threshold

    # Check specific values
    assert PRESETS["conservative"].sent_starter_threshold == 30.0
    assert PRESETS["balanced"].sent_starter_threshold == 15.0
    assert PRESETS["aggressive"].sent_starter_threshold == 5.0


def test_training_with_presets():
    """Test training with different presets."""
    test_text = "This is a test. However, it is short. Therefore, results may vary." * 50

    # Train with each preset
    results = {}
    for preset in ["conservative", "balanced", "aggressive"]:
        trainer = train_model(
            training_texts=test_text, hyperparameters=preset, use_default_abbreviations=False
        )
        params = trainer.get_params()
        results[preset] = {
            "sent_starters": len(params.sent_starters),
            "collocations": len(params.collocations),
        }

    # Aggressive should learn more or equal patterns
    assert results["aggressive"]["sent_starters"] >= results["balanced"]["sent_starters"]
    assert results["balanced"]["sent_starters"] >= results["conservative"]["sent_starters"]


def test_custom_hyperparameters():
    """Test with custom hyperparameter object."""
    test_text = "Test sentence. Another sentence." * 100

    # Create custom hyperparameters
    custom_hp = PunktHyperparameters(
        sent_starter_threshold=1.0,  # Very low
        sent_starter_min_freq=1,
    )

    trainer = train_model(
        training_texts=test_text, hyperparameters=custom_hp, use_default_abbreviations=False
    )

    params = trainer.get_params()
    # Should learn something with such low thresholds
    assert len(params.sent_starters) > 0


def test_dict_hyperparameters():
    """Test with dictionary hyperparameters."""
    test_text = "Test text. More text." * 100

    trainer = train_model(
        training_texts=test_text,
        hyperparameters={"sent_starter_threshold": 15.0, "collocation_threshold": 4.0},
        use_default_abbreviations=False,
    )

    # Just check it doesn't crash
    assert trainer is not None


def test_invalid_preset():
    """Test error handling for invalid preset."""
    with pytest.raises(ValueError, match="Unknown hyperparameter preset"):
        train_model(training_texts="Test text.", hyperparameters="invalid_preset")


def test_hyperparameter_dataclass():
    """Test PunktHyperparameters dataclass functionality."""
    # Test from_dict
    hp = PunktHyperparameters.from_dict(
        {
            "sent_starter_threshold": 20.0,
            "invalid_field": "ignored",  # Should be filtered out
        }
    )
    assert hp.sent_starter_threshold == 20.0

    # Test to_dict
    hp_dict = hp.to_dict()
    assert "sent_starter_threshold" in hp_dict
    assert "invalid_field" not in hp_dict

    # Test apply_to_trainer
    from nupunkt.trainers.base_trainer import PunktTrainer

    trainer = PunktTrainer()

    hp = PunktHyperparameters(sent_starter_threshold=99.0)
    hp.apply_to_trainer(trainer)

    assert trainer.SENT_STARTER == 99.0
