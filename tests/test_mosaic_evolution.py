"""CPU-only behavior checks for the user-authored Mosaic EA operators."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def evolution():
    path = Path(__file__).resolve().parents[1] / "drivers" / "mosaic" / "evolution.py"
    spec = importlib.util.spec_from_file_location("mosaic_evolution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def population():
    tokens = np.array([[0, 1], [1, 2], [2, 0], [0, 2]])
    return 0.85 * np.eye(3)[tokens] + 0.15 / 3


def test_select_replaces_only_worst_members_and_preserves_losses(evolution):
    losses = np.array([7.0, -2.0, 4.0, -1.0])
    original = losses.copy()
    parents = evolution.select(losses, np.random.default_rng(41))

    assert parents.dtype.kind in "iu"
    np.testing.assert_array_equal(parents[[1, 3]], [1, 3])
    assert set(parents[[0, 2]]) <= {1, 3}
    np.testing.assert_array_equal(losses, original)


def test_select_uses_exact_resampled_population_references(evolution):
    losses = np.array([9.0, 2.0, 8.0, 1.0, 7.0, 0.0])
    reference_rng = np.random.default_rng(29)
    survivors, culled = evolution.cull(losses, 0.5, reference_rng)
    expected = np.arange(len(losses))
    expected[culled] = evolution.resample(survivors, len(culled), reference_rng)

    parents = evolution.select(losses, np.random.default_rng(29))
    np.testing.assert_array_equal(parents, expected)
    assert set(parents[culled]) <= {1, 3, 5}


def test_cull_ties_use_seeded_permutation_and_floor_count(evolution):
    losses = np.ones(7)
    expected_order = np.random.default_rng(52).permutation(len(losses))
    survivors, culled = evolution.cull(losses, 0.5, np.random.default_rng(52))
    np.testing.assert_array_equal(survivors, expected_order[:4])
    np.testing.assert_array_equal(culled, expected_order[4:])
    first = evolution.select(losses, np.random.default_rng(52))
    second = evolution.select(losses, np.random.default_rng(52))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[survivors], survivors)
    assert set(first[culled]) <= set(survivors)


@pytest.mark.parametrize("fraction", [0.0, 0.1])
def test_select_with_no_cull_is_identity(evolution, fraction):
    parents = evolution.select([4, 3, 2, 1], np.random.default_rng(3), fraction=fraction)
    np.testing.assert_array_equal(parents, np.arange(4))


@pytest.mark.parametrize("losses", [
    [], [[1, 2]], [1, np.nan], [1, np.inf], [1, 2j], ["1", "2"],
])
def test_select_rejects_malformed_losses(evolution, losses):
    with pytest.raises(ValueError, match="losses"):
        evolution.select(losses, np.random.default_rng(1))


@pytest.mark.parametrize("fraction", [-0.1, 1.0, np.inf, np.nan])
def test_select_rejects_invalid_cull_fraction(evolution, fraction):
    with pytest.raises(ValueError, match="fraction"):
        evolution.select([1, 2, 3, 4], np.random.default_rng(1), fraction=fraction)


def test_resample_returns_original_indices_and_preserves_survivors(evolution):
    survivors = np.array([3, 17, 42])
    original = survivors.copy()
    sampled = evolution.resample(survivors, 50, np.random.default_rng(81))
    expected = np.random.default_rng(81).choice(survivors, size=50, replace=True)
    np.testing.assert_array_equal(sampled, expected)
    assert set(sampled) == {3, 17, 42}
    np.testing.assert_array_equal(survivors, original)


def test_resample_empty_result_is_integer(evolution):
    result = evolution.resample([], 0, np.random.default_rng(1))
    assert result.shape == (0,)
    assert result.dtype.kind in "iu"


@pytest.mark.parametrize("survivors,n", [
    ([0, 0], 1), ([0, -1], 1), ([0.0, 1.0], 1), ([True, False], 1),
    ([[0, 1]], 1), (["0", "1"], 1), ([], 1), ([0, 1], -1),
    ([0, 1], 1.0), ([0, 1], True), ([0, 1], np.bool_(False)),
    ([1, 1], 0), ([-1], 0),
])
def test_resample_rejects_malformed_indices_and_counts(evolution, survivors, n):
    with pytest.raises(ValueError):
        evolution.resample(survivors, n, np.random.default_rng(1))


def test_mutate_uses_already_gathered_pssms_exactly_once(evolution, population):
    parents = np.array([2, 0, 2, 1])
    gathered = population[parents]
    result = evolution.mutate(
        gathered, parents, [4, 3, 2, 1], np.random.default_rng(8), low=0, high=0,
    )
    expected = np.eye(3)[gathered.argmax(axis=-1)]
    np.testing.assert_array_equal(result, expected)


def test_mutate_restarts_survivors_and_independent_children_without_mutation(
    evolution, population,
):
    parents = np.array([0, 1, 0, 1])
    losses = np.array([1.0, 2.0, 3.0, 4.0])
    gathered = population[parents]
    originals = [array.copy() for array in (population, gathered, parents, losses)]
    result = evolution.mutate(gathered, parents, losses, np.random.default_rng(66))
    expected = evolution.perturb(gathered, 0.10, 0.20, np.random.default_rng(66))

    np.testing.assert_array_equal(result, expected)
    assert result.shape == gathered.shape
    assert np.isfinite(result).all()
    assert (result >= 0).all()
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, rtol=0, atol=1e-14)
    assert not np.array_equal(result[0], result[2])
    assert not np.array_equal(result[1], result[3])
    for member in range(4):
        assert not np.array_equal(result[member], gathered[member])
    for array, original in zip((population, gathered, parents, losses), originals):
        np.testing.assert_array_equal(array, original)
    assert not np.shares_memory(result, gathered)


@pytest.mark.parametrize("parents", [
    [0, 1, 2], [[0, 1, 2, 3]], [0.0, 1.0, 2.0, 3.0],
    [True, False, True, False], [-1, 1, 2, 3], [0, 1, 2, 4],
    [0, 1, 2, np.nan], ["0", "1", "2", "3"], [],
])
def test_mutate_rejects_malformed_parents(evolution, population, parents):
    with pytest.raises(ValueError, match="parents"):
        evolution.mutate(population, parents, [1, 2, 3, 4], np.random.default_rng(1))


@pytest.mark.parametrize("losses", [
    [], [1, 2, 3], [[1, 2, 3, 4]], [1, 2, 3, np.inf],
    [1, 2, 3, np.nan], [1j, 2, 3, 4], ["1", "2", "3", "4"],
])
def test_mutate_rejects_malformed_losses(evolution, population, losses):
    with pytest.raises(ValueError, match="losses"):
        evolution.mutate(population, np.arange(4), losses, np.random.default_rng(1))


@pytest.mark.parametrize("pssm", [
    np.zeros((4, 2)), np.zeros((4, 0, 3)), np.zeros((4, 2, 3)),
    np.full((4, 2, 3), -1.0), np.full((4, 2, 3), np.nan),
    np.full((4, 2, 3), 1j),
])
def test_mutate_rejects_malformed_probabilities(evolution, pssm):
    with pytest.raises(ValueError, match="pssm"):
        evolution.mutate(pssm, np.arange(4), [1, 2, 3, 4], np.random.default_rng(1))


def test_make_hooks_binds_nondefault_settings_to_driver_signatures(evolution, population):
    select, mutate = evolution.make_hooks(0.25, 0.35, 0.35)
    losses = np.array([1.0, 4.0, 2.0, 3.0])
    rng = np.random.default_rng(101)
    parents = select(losses, rng)
    result = mutate(population[parents], parents, losses, rng)

    reference_rng = np.random.default_rng(101)
    expected_parents = evolution.select(losses, reference_rng, fraction=0.25)
    expected = evolution.perturb(population[expected_parents], 0.35, 0.35, reference_rng)
    np.testing.assert_array_equal(parents, expected_parents)
    np.testing.assert_array_equal(parents[[0, 2, 3]], [0, 2, 3])
    assert parents[1] in {0, 2, 3}
    np.testing.assert_array_equal(result, expected)
