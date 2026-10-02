"""One-shot, stdlib-only completion publisher for the frozen PR 116 campaigns.

Numerical analysis and plots are produced by existing Slurm report jobs. This
login-node process only waits, copies artifacts and publishes the authorized PR.
"""

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/personal/andrinr/runner/results/mosaic")
WORKTREE = Path("/home/andrinr/mosaic-pr116-artifacts")
GH = "/home/andrinr/.local/bin/gh"
CAMPAIGNS = ["pr116-warp-repaired-20261001", "pr116-forced-training-20261002"]
GATES = [2858787, 2858730, 2858917, 2858883]
CAMPAIGN_DIR = ROOT / CAMPAIGNS[0]
ACTIVE = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "REQUEUED",
    "RESIZING",
}


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def state(status, **details):
    value = {
        "status": status,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **details,
    }
    (CAMPAIGN_DIR / "completion-status.json").write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)


def row_for(report, phase, unroll, decay):
    rows = [
        r
        for r in report["comparisons"]
        if r["phase"] == phase
        and r["unroll"] == unroll
        and r["updates"] == 3000
        and (len({s["lr"] for s in r["curriculum"]}) > 1) == decay
    ]
    assert len(rows) <= 1, "Ambiguous protocol"
    return rows[0] if rows else None


def pct(value):
    return f"{100 * value:.3f}%" if value is not None else "nonfinite"


def effect(row, key):
    value = row[key]
    if value["ratio"] is None:
        return "nonfinite"
    point = 100 * (1 - 1 / value["ratio"])
    ci = value["ci95"]
    interval = (
        f" [{100 * (1 - 1 / ci[0]):+.2f}, {100 * (1 - 1 / ci[1]):+.2f}]"
        if ci
        else " (no interval)"
    )
    return f"{point:+.2f}%{interval}"


def build_body(warp, forced, revision):
    fresh = row_for(warp, "confirm", 8, True)
    complete = fresh is not None and fresh["n_model_seeds"] == 8
    validated = complete and fresh["admitted"] and fresh["gradient_checks_passed"]
    resolved = validated and fresh["resolved_joint_benefit"]
    lead = (
        "**The selected Warp setting reproduces a solver-gradient benefit on fresh test ICs across eight model seeds.**"
        if resolved
        else "**An admitted eight-seed fresh-IC confirmation is unavailable; the exploratory gain remains unconfirmed.**"
    )
    if validated and not resolved:
        lead = "**The fresh-IC result does not resolve an advantage over every baseline; the exploratory gain is not fully confirmed.**"
    body = (
        """Test whether a neural corrector benefits from solver-in-the-loop training, and whether differentiating through the solver helps beyond stopped gradients.

Use each solver at **64² coarse and 192² reference resolution**. Restrict that same solver's fine fields to 64²; audit temporal accuracy by halving the fine timestep. This does not establish spatial convergence. No 256² runs.

Compare fixed-pair supervision, recurrent training with stopped solver gradients, and full solver gradients. Match initialization, sampled windows and optimizer updates; evaluate held-out free-running trajectories. Budgets match updates and target exposure, not wall time.

"""
        + lead
        + "\n\nMean held-out relative L2 error, lower is better:\n\n"
    )
    body += "| Warp protocol | Seeds / test ICs | Solver only | Supervised | Stopped | Full | Checks |\n|---|---:|---:|---:|---:|---:|---|\n"
    specifications = [
        ("Fixed LR, original ICs", "explore", 8, False),
        ("Fixed LR, fresh ICs 1000–1007", "confirm", 8, False),
        ("LR decay, original ICs", "explore", 8, True),
        ("LR decay, fresh ICs 2000–2007", "confirm", 8, True),
        ("Horizon 16, original ICs", "explore", 16, False),
    ]
    for label, phase, horizon, decay in specifications:
        row = row_for(warp, phase, horizon, decay)
        if row is None:
            body += f"| {label} | — | — | — | — | — | No completed trained result |\n"
            continue
        checks = (
            "pass"
            if row["admitted"] and row["gradient_checks_passed"]
            else "reference/gradient failure"
        )
        values = [
            pct(row["arm_summaries"][a]["mean_rollout_error"])
            for a in ["uncorrected", "supervised", "stop_gradient", "corrected"]
        ]
        body += (
            f"| {label} | {row['n_model_seeds']} / {row['n_test_ics']} | "
            + " | ".join(values)
            + f" | {checks} |\n"
        )
    body += "\nCompare methods within each row; rows differ in data and seed counts. All use 3,000 updates. LR decay uses 1,500 updates at 1e-4 then 1,500 at 1e-5, identically for all arms with Adam state retained.\n\n"
    if fresh:
        body += f"**Fresh-IC decay result:** full-gradient error reduction versus supervision **{effect(fresh, 'vs_supervised')}**, and versus stopped gradients **{effect(fresh, 'vs_stopped')}**. Brackets give paired model-seed/IC bootstrap 95% intervals; positive means lower error. "
        body += f"Full training costs {fresh['full_training_seconds'] / fresh['supervised_total_seconds']:.2f}× supervision and {fresh['full_training_seconds'] / fresh['stopped_training_seconds']:.2f}× stopped gradients. "
        body += "This is evidence for this setting and update budget, not superiority over every tuned supervised baseline.\n\n"
    body += """The decay schedule and eight-seed sample size were frozen before inspecting outcomes on ICs 2000–2007. Confirmation uses training ICs 0–15, disjoint from test ICs. The separate fixed-LR expansion to eight seeds was chosen after its first three results; all eight are retained. Exploration selected the decay candidate, so the new test-IC check is the relevant confirmation.

**Other outcomes:** JAX-CFD and INS decaying-flow experiments with longer training, supervised pretraining and curricula favor supervision. PICT favors supervision at 300 updates; its longer run timed out. PhiFlow and XLB fail reference admission. Warp's original failure led to a GPU-validated non-power-of-two FFT normalization fix. JAX-CFD's failed warm-start gradient check remains unresolved after a same-model finite-difference study. Horizon-16 Warp seed 2 fails its original perturbation check; separate smaller-step probes converge, and both records are retained.

**Forced INS pilot:** """
    for row in forced["comparisons"]:
        values = ", ".join(
            f"{label} {pct(row['arm_summaries'][arm]['mean_rollout_error'])}"
            for arm, label in [
                ("supervised", "supervised"),
                ("stop_gradient", "stopped"),
                ("corrected", "full"),
            ]
        )
        checks = (
            "pass"
            if row["admitted"] and row["gradient_checks_passed"]
            else "do not all pass"
        )
        body += f"{row['n_model_seeds']} completed seeds, 1,000 updates, horizon 8: {values}; reference/gradient checks {checks}. "
    if not forced["comparisons"]:
        body += "No completed trained comparison. "
    for name, report in [("Warp", warp), ("forced INS", forced)]:
        for failure in report["failures"]:
            body += f"{name} cell `{failure['cell']}`: {failure['reason']}. "
        if report["pending"]:
            body += f"{name}: {len(report['pending'])} missing/pending outcomes. "
    body += """The pilot uses forcing k=6, amplitude 1, viscosity 0.001 and burn-in 75. Its cheaper reference preflight passed (0.0010% temporal discrepancy); full runs repeat admission. Burn-in does not certify stationarity. JAX-CFD's forced reference remains ineligible at 0.69% discrepancy.

[List et al. (2025)](https://doi.org/10.1016/j.cma.2024.117441) motivated forcing, curricula and learning-rate decay. These are adaptations, not replications.

"""
    selected = fresh if fresh else row_for(warp, "explore", 8, True)
    if selected:
        configurations = sorted((CAMPAIGN_DIR / "configs").glob("*.json"))
        matching = []
        for path in configurations:
            payload = json.loads(path.read_text())
            run = payload["run"]
            run["training"].pop("model_seeds", None)
            digest = hashlib.sha256(
                json.dumps(run, sort_keys=True).encode()
            ).hexdigest()
            if (
                digest == selected["protocol_sha256"]
                and (CAMPAIGN_DIR / "plots" / path.stem / "fields.png").exists()
            ):
                matching.append(path.stem)
        assert matching, "No deterministic full-field plot for selected protocol"
        cell = matching[0]
        base = f"https://raw.githubusercontent.com/pasteurlabs/mosaic/{revision}/pr-116/warp-repaired-20261001/plots"
        filename = (
            f"outcome-{selected['phase']}-warp-ns-{selected['protocol_sha256'][:8]}.png"
        )
        assert (CAMPAIGN_DIR / "plots" / filename).exists(), filename
        body += f"Results for the {'fresh-IC confirmation' if fresh else 'exploratory candidate'}; dots show model-seed effects and bars give paired 95% intervals.\n\n![Outcome]({base}/{filename})\n\n"
        body += "Full fields show the first available model seed and first held-out IC, selected by configuration order. Vorticity uses a shared reference-p99 colour scale; velocity errors share an unclipped scale. The rollout curve averages held-out ICs for that model seed.\n\n"
        for filename, label in [
            ("fields.png", "Full fields"),
            ("field_errors.png", "Spatial velocity errors"),
            ("rollout.png", "Rollout errors"),
        ]:
            body += f"![{label}]({base}/{cell}/{filename})\n\n"
    body += f"""[All reports, configurations, plots and failures](https://github.com/pasteurlabs/mosaic/tree/{revision}/pr-116). Raw reference and gradient failures remain visible. [Frozen protocols](https://github.com/pasteurlabs/mosaic/blob/cd1b459/experiments/solver_in_loop/WARP_REPLICATION.md), [gradient diagnostics](https://github.com/pasteurlabs/mosaic/blob/ebf6ad6/experiments/solver_in_loop/GRADIENT_CHECKS.md).

Cluster validation: **693 tests passed, 3 skipped**; separate Warp GPU FFT/Poisson/VJP checks passed. Numerical work and rendering use `slurm-runner` on the cluster. Earlier shared-reference and 32²→128² results are superseded. Merged `main`; remains stacked on #121.
"""
    return body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    expected = json.loads((CAMPAIGN_DIR / "completion-expected-body.json").read_text())[
        "body"
    ]
    if not args.preview:
        deadline = time.monotonic() + 6 * 3600
        while True:
            raw = command(
                [
                    "sacct",
                    "-X",
                    "-n",
                    "-P",
                    "-j",
                    ",".join(map(str, GATES)),
                    "--format=JobIDRaw,State%30",
                ]
            )
            states = {
                line.split("|")[0]: line.split("|")[1].split()[0].rstrip("+")
                for line in raw.splitlines()
                if line
            }
            if all(
                str(job) in states and states[str(job)] not in ACTIVE for job in GATES
            ):
                break
            state("waiting_for_reports", jobs=states)
            if time.monotonic() > deadline:
                raise RuntimeError("Six-hour completion deadline exceeded")
            time.sleep(55)
        if any(states[str(job)] != "COMPLETED" for job in GATES):
            raise RuntimeError(
                f"Report job failed; manual inspection required: {states}"
            )
    warp, forced = [
        (
            json.loads((ROOT / c / "report.json").read_text())
            if (ROOT / c / "report.json").exists()
            else {"comparisons": [], "pending": ["report not produced"], "failures": []}
        )
        for c in CAMPAIGNS
    ]
    preview = build_body(warp, forced, "pr-116-local-results")
    (CAMPAIGN_DIR / "completion-preview.md").write_text(preview)
    if args.preview:
        print(preview)
        return
    assert not warp["pending"] and not forced["pending"], (
        "Reports still contain pending outcomes"
    )
    current = command(
        [GH, "api", "repos/pasteurlabs/mosaic/pulls/116", "--jq", ".body"]
    )
    assert current.rstrip() == expected.rstrip(), (
        "PR body changed since arming; refusing to overwrite"
    )
    assert not command(["git", "-C", str(WORKTREE), "status", "--porcelain"]), (
        "Artifact worktree is dirty"
    )
    for campaign in CAMPAIGNS:
        source = ROOT / campaign
        destination = WORKTREE / "pr-116" / campaign.replace("pr116-", "", 1)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "report.json", destination / "report.json")
        shutil.copytree(source / "plots", destination / "plots", dirs_exist_ok=True)
        shutil.copytree(source / "configs", destination / "configs", dirs_exist_ok=True)
    shutil.copy2(__file__, WORKTREE / "pr-116" / "completion-publisher-20261002.py")
    git = ["git", "-C", str(WORKTREE)]
    command(
        git
        + [
            "add",
            "pr-116/warp-repaired-20261001",
            "pr-116/forced-training-20261002",
            "pr-116/completion-publisher-20261002.py",
        ]
    )
    if command(git + ["diff", "--cached", "--name-only"]):
        command(
            git
            + [
                "-c",
                "user.name=andrinr",
                "-c",
                "user.email=andrinrehmann@gmail.com",
                "commit",
                "-m",
                "results: publish all frozen eight-seed confirmations and forced pilot outcomes",
            ]
        )
    command(git + ["push", "origin", "HEAD:pr-116-local-results"])
    revision = command(git + ["rev-parse", "HEAD"])
    body = build_body(warp, forced, revision)
    update = CAMPAIGN_DIR / "completion-pr-update.json"
    update.write_text(json.dumps({"body": body}))
    # Check again after artifact publication to avoid overwriting a concurrent edit.
    current = command(
        [GH, "api", "repos/pasteurlabs/mosaic/pulls/116", "--jq", ".body"]
    )
    assert current.rstrip() == expected.rstrip(), (
        "PR body changed during artifact publication"
    )
    url = command(
        [
            GH,
            "api",
            "--method",
            "PATCH",
            "repos/pasteurlabs/mosaic/pulls/116",
            "--input",
            str(update),
            "--jq",
            ".html_url",
        ]
    )
    state("published", artifact_commit=revision, pr=url)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        state("failed", error=str(exc))
        raise
