"""Settings changes must apply to both local inference and server startup."""
from dataclasses import replace

import pytest

from app.config import VisionConfig
from app.settings import AlertSettings, load_settings


def test_default_file_contains_exact_requested_class_ids_and_alert_defaults():
    settings = load_settings()
    assert list(settings.detection.class_names) == [0, 1, 13, 16, 24, 28, 56, 57, 58, 60]
    assert settings.detection.class_names[58] == "potted plant"
    assert settings.alerts == AlertSettings()


def test_custom_settings_reach_environment_server_config(tmp_path, monkeypatch):
    path = tmp_path / "custom.toml"
    path.write_text('[detection.classes]\n"2" = "car"\n[alerts]\nwarning_m = 2.5\nsound_enabled = false\n')
    settings = load_settings(path)
    assert settings.alerts.warning_m == 2.5
    assert not settings.alerts.sound_enabled
    monkeypatch.setenv("OBSTACLE_CONFIG", str(path))
    assert VisionConfig.from_env().target_classes == ((2, "car"),)


@pytest.mark.parametrize("field,value", [
    ("danger_m", 0), ("danger_m", 3), ("warning_m", 4), ("display_max_m", 21),
    ("warning_repeat_s", 0), ("danger_repeat_s", -1), ("track_ttl_s", 0),
    ("result_timeout_s", 0), ("smoothing_window", 0), ("smoothing_window", 5.0),
    ("confirm_frames", True), ("hysteresis_m", -0.1), ("hysteresis_m", 1),
    ("center_overlap_ratio", 0), ("track_iou_threshold", 1.01), ("sound_enabled", "false"),
    ("warning_m", float("nan")), ("danger_repeat_s", float("inf")),
])
def test_invalid_alert_configuration_is_rejected(field, value):
    with pytest.raises(ValueError):
        replace(AlertSettings(), **{field: value})


@pytest.mark.parametrize("content", [
    '[detection.classes]\n"80"="person"\n[alerts]\n',
    '[detection.classes]\n"0"=""\n[alerts]\n',
    '[detection.classes]\n"person"="person"\n[alerts]\n',
    '[detection.classes]\n"0"="person"\n"00"="person"\n[alerts]\n',
    '[detection.classes]\n[alerts]\n',
    '[detection.classes]\n"0"="person"\n[alerts]\nwarining_m=2\n',
    '[unrelated]\na=1\n', 'not a valid TOML [',
])
def test_bad_settings_fail_with_actionable_path(tmp_path, content):
    path = tmp_path / "settings.toml"
    path.write_text(content)
    with pytest.raises(ValueError, match="Cannot load settings"):
        load_settings(path)


def test_missing_settings_file_fails_instead_of_silent_default(tmp_path):
    with pytest.raises(ValueError, match="Cannot load settings"):
        load_settings(tmp_path / "missing.toml")
