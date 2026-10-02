# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Development selection must never favor incomplete tasks or unregistered settings."""

from __future__ import annotations

import copy

import pytest

from experiments.flow_control.direct_orchestrate import select_settings

SEEDS = list(range(3000, 3008))
SETTINGS = [
    {"setting_id": "adam-first", "method": "adam", "optimizer_settings": {"lr": 0.01}},
    {"setting_id": "adam-second", "method": "adam", "optimizer_settings": {"lr": 0.05}},
    {"setting_id": "spsa", "method": "spsa", "optimizer_settings": {"lr": 0.01}},
    {"setting_id": "powell", "method": "powell", "optimizer_settings": {}},
]


@pytest.fixture
def results():
    return [
        {
            "task_seed": seed,
            "completed": True,
            "admitted": True,
            "common_admitted": True,
            "settings": [
                {
                    **copy.deepcopy(setting),
                    "completed": True,
                    "failure": None,
                    "snapshots": [
                        {
                            "budget_kind": "wall_time_s",
                            "budget": 120,
                            "available": True,
                            "admitted": True,
                            "fine_objective": 0.1 + index / 100,
                        },
                    ],
                }
                for index, setting in enumerate(SETTINGS)
            ],
        }
        for seed in SEEDS
    ]


def test_complete_eight_task_selection_and_exact_tie_manifest_order(results):
    for result in results:
        result["settings"][1]["snapshots"][0]["fine_objective"] = 0.1
        result["settings"].reverse()  # Execution order cannot resolve ties.
    chosen = select_settings(list(reversed(results)), SETTINGS, SEEDS)
    assert chosen["selected"]["adam"]["setting_id"] == "adam-first"
    assert set(chosen["selected"]) == {"adam", "spsa", "powell"}
    assert all(len(row["task_values"]) == 8 for row in chosen["development_scores"])


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "extra", "test_task"])
def test_refuses_wrong_task_identity(results, invalid):
    if invalid == "missing":
        results.pop()
    elif invalid == "duplicate":
        results[-1]["task_seed"] = results[0]["task_seed"]
    elif invalid == "extra":
        results.append(copy.deepcopy(results[0]))
    else:
        results[-1]["task_seed"] = 5000
    with pytest.raises(ValueError, match="identit"):
        select_settings(results, SETTINGS, SEEDS)


@pytest.mark.parametrize(
    "invalid",
    [
        "failure",
        "missing_snapshot",
        "query_snapshot",
        "duplicate_snapshot",
        "unavailable",
        "audit",
        "nan",
        "infinite",
        "missing_run",
        "duplicate_run",
        "method",
        "learning_rate",
    ],
)
def test_one_invalid_setting_does_not_poison_others(results, invalid):
    result = results[0]
    result["admitted"] = False  # Whole-job aggregate is deliberately stricter.
    run = result["settings"][0]
    snap = run["snapshots"][0]
    if invalid == "failure":
        run["failure"] = "numerical failure"
    elif invalid == "missing_snapshot":
        run["snapshots"] = []
    elif invalid == "query_snapshot":
        snap["budget_kind"] = "queries"
    elif invalid == "duplicate_snapshot":
        run["snapshots"].append(copy.deepcopy(snap))
    elif invalid == "unavailable":
        snap["available"] = False
    elif invalid == "audit":
        snap["admitted"] = False
    elif invalid == "nan":
        snap["fine_objective"] = float("nan")
    elif invalid == "infinite":
        snap["fine_objective"] = float("inf")
    elif invalid == "missing_run":
        result["settings"].pop(0)
    elif invalid == "duplicate_run":
        result["settings"].append(copy.deepcopy(run))
    elif invalid == "method":
        run["method"] = "spsa"
    elif invalid == "learning_rate":
        run["optimizer_settings"]["lr"] = 999
    chosen = select_settings(results, SETTINGS, SEEDS)
    assert chosen["selected"]["adam"]["setting_id"] == "adam-second"
    assert set(chosen["selected"]) == {"adam", "spsa", "powell"}
    bad = chosen["development_scores"][0]
    assert not bad["eligible"]
    assert bad["mean_fine_objective_120s"] is None
    assert bad["failed_task_seeds"] == [3000]


def test_common_admission_failure_blocks_all_methods(results):
    results[0]["common_admitted"] = False
    chosen = select_settings(results, SETTINGS, SEEDS)
    assert not chosen["selected"]
    assert all(not row["eligible"] for row in chosen["development_scores"])


def test_primary120_selection_ignores_better_early_checkpoint(results):
    for result in results:
        result["settings"][1]["snapshots"].append(
            {
                "budget_kind": "wall_time_s",
                "budget": 30,
                "available": True,
                "admitted": True,
                "fine_objective": 0.0,
            }
        )
    assert (
        select_settings(results, SETTINGS, SEEDS)["selected"]["adam"]["setting_id"]
        == "adam-first"
    )


def test_failed_method_is_not_replaced_by_unregistered_setting(results):
    for result in results:
        result["settings"][2]["failure"] = "failed"
        unknown = copy.deepcopy(result["settings"][2])
        unknown.update(setting_id="unregistered", failure=None)
        unknown["snapshots"][0]["fine_objective"] = 0.0
        result["settings"].append(unknown)
    chosen = select_settings(results, SETTINGS, SEEDS)
    assert set(chosen["selected"]) == {"adam", "powell"}
