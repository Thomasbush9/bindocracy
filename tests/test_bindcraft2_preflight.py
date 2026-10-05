"""Offline settings-key checks at the public configuration-loading boundary."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from conftest import bindcraft2_settings, write_bindcraft2_configs

from bindocracy.config.preflight import ConfigPreflightError
from bindocracy.tools import load_configs

INVALID_SETTINGS = [
    ({"_comment": "not a consumer setting"}, "_comment"),
    ({"comment": "not a consumer setting"}, "comment"),
    ({"binder_lenghts": [65]}, "binder_lenghts"),
    ({"weights_not_a_registered_loss": 0}, "weights_not_a_registered_loss"),
    ({"filters": {"Not_A_Registered_Metric": {}}}, "filters.Not_A_Registered_Metric"),
    ({"losses": {"not_a_registered_loss": {}}}, "losses.not_a_registered_loss"),
    (
        {"filters": {"Hotspot_Contact_Fraction": {"threshhold": 0.5}}},
        "filters.Hotspot_Contact_Fraction.threshhold",
    ),
    (
        {"filters": {"Hotspot_Contact_Fraction": {"params": {"cuttof": 5}}}},
        "filters.Hotspot_Contact_Fraction.params.cuttof",
    ),
    (
        {"losses": {"interface_contacts": {"_comment": "not supported"}}},
        "losses.interface_contacts._comment",
    ),
    (
        {"losses": {"interface_contacts": {"params": {"cuttof": 5}}}},
        "losses.interface_contacts.params.cuttof",
    ),
]

# These exercise preset selectors, derived shorthand names, and nested metric
# parameters, rather than treating the small authored example as an allowlist.
SUPPORTED_SETTINGS = {
    "core": "default",
    "modality": "binder",
    "bigbang": False,
    "weights_interface_contacts": 0,
    "min_plddt_screen": 0,
    "filters": {
        "Hotspot_Contact_Fraction": {
            "threshold": 0.5,
            "higher": True,
            "mandatory": True,
            "prediction_state": "bound",
            "params": {"cutoff": 5},
        },
    },
    "losses": {
        "interface_contacts": {
            "prediction_state": "bound",
            "params": {"cutoff": 5, "contacts_per_residue": 1},
        },
    },
}


@pytest.mark.parametrize("overrides,unknown", INVALID_SETTINGS)
def test_unknown_consumer_keys_are_refused_before_planning(
    tmp_path: Path, overrides: dict, unknown: str
) -> None:
    general, model = write_bindcraft2_configs(
        tmp_path, settings=bindcraft2_settings(**overrides)
    )
    authored = (tmp_path / "dio3_cut.json").read_bytes()

    with pytest.raises(ConfigPreflightError) as refused:
        load_configs(general, model)

    assert unknown in str(refused.value)
    assert (tmp_path / "dio3_cut.json").read_bytes() == authored
    assert not (tmp_path / "run").exists()


def test_supported_presets_and_nested_settings_are_preserved(tmp_path: Path) -> None:
    settings = bindcraft2_settings(**SUPPORTED_SETTINGS)
    general, model = write_bindcraft2_configs(tmp_path, settings=settings)

    loaded = load_configs(general, model)

    assert loaded.preflight.settings == settings
    assert loaded.model.settings.contents == settings
    assert json.loads((tmp_path / "dio3_cut.json").read_text()) == settings


def test_key_validation_agrees_with_the_qualified_upstream_consumer(tmp_path: Path) -> None:
    """Opt-in CPU comparison against the actual upstream rejection function."""
    container = os.environ.get("BINDCRAFT2_CONTAINER")
    if not container:
        pytest.skip("set BINDCRAFT2_CONTAINER to the qualified BC2 1.0.1 image")
    documents = [bindcraft2_settings(**SUPPORTED_SETTINGS)] + [
        bindcraft2_settings(**overrides) for overrides, _ in INVALID_SETTINGS
    ]
    accepted = []
    for index, document in enumerate(documents):
        general, model = write_bindcraft2_configs(tmp_path / str(index), settings=document)
        try:
            load_configs(general, model)
        except ConfigPreflightError:
            accepted.append(False)
        else:
            accepted.append(True)
    parser = """
import json
import sys
from bindcraft.settings import reject_unrecognized_settings
accepted = []
for document in json.load(sys.stdin):
    try:
        reject_unrecognized_settings(document)
    except ValueError:
        accepted.append(False)
    else:
        accepted.append(True)
print(json.dumps(accepted))
"""
    result = subprocess.run(
        [
            "singularity", "exec", "--cleanenv", "--env", "JAX_PLATFORMS=cpu",
            container, "python3", "-c", parser,
        ],
        input=json.dumps(documents), check=True, capture_output=True, text=True, timeout=90,
    )
    assert accepted == json.loads(result.stdout)
    assert accepted == [True] + [False] * len(INVALID_SETTINGS)
