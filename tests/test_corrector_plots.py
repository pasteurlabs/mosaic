"""Bounded corrector figures preserve all arms without ranking own-solver targets."""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.problems.navier_stokes_grid import corrector_plots as plots


def test_three_figures_keep_all_arms_and_unclipped_fields(tmp_path, monkeypatch):
    for filename in ("result.json", "corrector_fields.npz"):
        (tmp_path / filename).touch()
    arrays = {"solver_names": np.array(["ins"]), "evaluation_times": np.arange(3)}
    for index, arm in enumerate(
        ("uncorrected", "supervised", "stop_gradient", "corrected")
    ):
        arrays[f"error_{arm}_0"] = np.array([0, index + 1, index + 2])
        arrays[f"rollout_{arm}_0"] = np.full((3, 4, 4, 1, 2), index + 1.0)
    arrays["reference_rollout_0"] = np.full((3, 4, 4, 1, 2), -5.0)
    for key in ("loss", "loss_supervised", "loss_stop_gradient"):
        arrays[f"{key}_0"] = np.array([2.0, 1.0])
    metrics = {
        "training_wall_time_s": 3.0,
        "supervised_training_wall_time_s": 1.0,
        "stop_gradient_training_wall_time_s": 2.0,
        "end_to_end_fd_rel_error": 0.001,
    }
    monkeypatch.setattr(plots, "experiment_dir", lambda *args: tmp_path)
    monkeypatch.setattr(
        plots, "load_json", lambda path: {"by_solver": {"ins": metrics}}
    )
    monkeypatch.setattr(plots, "try_load_npz", lambda path: arrays)
    figures = plots.plot_solver_in_loop(SimpleNamespace(name="ns-grid"), save=False)
    try:
        assert len(figures) == 3
        assert len(figures[0].axes[0].lines) == 4
        field_axes = figures[1].axes[:5]
        assert all(len(ax.images) == 1 for ax in field_axes)
        assert all(ax.images[0].get_clim() == (-5.0, 5.0) for ax in field_axes)
        assert len(figures[2].axes[0].lines) == 3
        assert [bar.get_height() for bar in figures[2].axes[1].patches] == [
            1.0,
            2.0,
            3.0,
        ]
        for index, figure in enumerate(figures):
            figure.savefig(tmp_path / f"figure-{index}.png")
    finally:
        for figure in figures:
            plt.close(figure)


def test_corrector_own_reference_results_are_not_ranked(tmp_path, monkeypatch):
    from docs import generate_results

    directory = tmp_path / "ns-grid" / "optimization" / "solver_in_loop"
    directory.mkdir(parents=True)
    monkeypatch.setattr(generate_results, "RESULTS_DIR", tmp_path)
    # Even an accidentally shared metric name must not rank incompatible targets.
    monkeypatch.setattr(
        generate_results,
        "_load_suite_results",
        lambda *args: [
            {"solver": "a", "metrics": {"final_error": 0.1}},
            {"solver": "b", "metrics": {"final_error": 0.2}},
        ],
    )
    assert generate_results._rank_optimization("ns-grid") is None
