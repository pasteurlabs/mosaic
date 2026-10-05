"""Publish solver-specific transfer artifacts without replacing the INS section."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

REPO = "pasteurlabs/mosaic"


def merge_transfer(body: str, section: str, solver: str) -> str:
    """Replace one explicitly named transfer section and preserve all other text."""
    start, end = (
        f"<!-- correction-transfer-{solver}-{suffix} -->" for suffix in ("start", "end")
    )
    block = f"{start}\n{section.strip()}\n{end}"
    if start in body or end in body:
        if (
            body.count(start) != 1
            or body.count(end) != 1
            or body.index(end) < body.index(start)
        ):
            raise ValueError("ambiguous transfer section markers")
        return body[: body.index(start)] + block + body[body.index(end) + len(end) :]
    return body.rstrip() + "\n\n" + block + "\n"


def main() -> None:
    """Publish artifacts; stage the PR edit unless the caller explicitly requests it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--gh", default="/home/andrinr/.local/bin/gh")
    parser.add_argument("--update-pr", action="store_true")
    parser.add_argument("--status-section", type=Path)
    args = parser.parse_args()
    campaign = args.campaign
    summary = json.loads((campaign / "report/report.json").read_text())
    audit = json.loads((campaign / "independent-audit.json").read_text())
    if (
        not summary.get("report_complete")
        or summary.get("failures")
        or not audit.get("passed")
    ):
        raise ValueError("completed independently audited transfer required")
    folder = (
        Path("pr-116") / campaign.parent.name.removeprefix("pr116-") / campaign.name
    )
    checkout = campaign / "publication-checkout"
    if not checkout.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--single-branch",
                "--filter=blob:none",
                "--depth=1",
                "--no-checkout",
                "--branch",
                "pr-116-local-results",
                f"git@github.com:{REPO}.git",
                str(checkout),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "sparse-checkout", "set", str(folder)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "pr-116-local-results"], check=True
        )
    git = ["git", "-C", str(checkout)]
    if subprocess.check_output([*git, "status", "--porcelain"], text=True).strip():
        raise RuntimeError("artifact checkout has uncommitted changes")
    subprocess.run([*git, "pull", "--ff-only"], check=True)
    folder = (
        Path("pr-116") / campaign.parent.name.removeprefix("pr116-") / campaign.name
    )
    destination = checkout / folder
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(campaign / "report", destination / "report", dirs_exist_ok=True)
    for name in (
        "plan.json",
        "report-input.json",
        "source.tar",
        "source.sha256",
        "controller-state.json",
        "reference-infrastructure-retries.json",
        "retry-controller-provenance.json",
        "independent-audit.json",
        "TRANSFER_PROTOCOL.md",
    ):
        if (campaign / name).exists():
            shutil.copy2(campaign / name, destination / name)
    shutil.copytree(campaign / "configs", destination / "configs", dirs_exist_ok=True)
    for path in sorted((campaign / "results").glob("*/outcome.json")):
        target = destination / "outcomes" / path.parent.name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target / path.name)
    subprocess.run([*git, "add", str(folder)], check=True)
    if subprocess.check_output(
        [*git, "diff", "--cached", "--name-only"], text=True
    ).strip():
        subprocess.run(
            [
                *git,
                "-c",
                "user.name=andrinr",
                "-c",
                "user.email=andrinrehmann@gmail.com",
                "commit",
                "-m",
                f"results: publish frozen {summary['solver']} corrector transfer",
            ],
            check=True,
        )
    subprocess.run([*git, "push", "origin", "HEAD:pr-116-local-results"], check=True)
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    url = f"https://raw.githubusercontent.com/{REPO}/{revision}/{folder}"
    section = (
        (campaign / "report/PRsection.md").read_text().replace("ARTIFACT_URL", url)
    )
    fitting = summary["costs"]["training_wall_s_all_eight_models"]
    ratio = fitting["full"] / fitting["supervised"]
    section += (
        f"\nFull-gradient model fitting took {ratio:.1f}× the supervision fitting time on the same B200 hardware. "
        "This ratio excludes shared reference preparation and evaluation; "
        "the report records shared preparation costs separately.\n"
    )
    if summary["solver"] == "jax-cfd":
        section += "\n‘Tuned supervision’ on the plot means the INS-selected recipe; no JAX-CFD retuning.\n"
    section += f"\n[Independent source/data/checkpoint and complete-matrix audit]({url}/independent-audit.json).\n"
    if args.status_section:
        section += "\n" + args.status_section.read_text().strip() + "\n"
    api = [args.gh, "api", f"repos/{REPO}/pulls/116"]
    current = json.loads(subprocess.check_output(api, text=True))
    current_body = current["body"]
    if summary["solver"] == "jax-cfd":
        current_body = current_body.replace(
            "Transfer to JAX-CFD, PhiFlow, PICT, Warp and XLB is being checked using the frozen recipe; "
            "no cross-solver success is claimed.",
            "The frozen recipe also improves JAX-CFD rollouts, as reported below. "
            "PhiFlow, PICT, Warp and XLB remain under admission checks; "
            "no result is claimed for those solvers.",
        )
    body = merge_transfer(current_body, section, summary["solver"])
    request = campaign / "transfer-publication-request.json"
    request.write_text(json.dumps({"body": body}))
    (campaign / "transfer-publication-section.md").write_text(section)
    if args.update_pr:
        latest = json.loads(subprocess.check_output(api, text=True))
        if latest["body"] != current["body"]:
            raise RuntimeError("PR changed during publication; refusing to overwrite")
        subprocess.run(
            [*api, "--method", "PATCH", "--input", str(request)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    receipt = {
        "artifact_commit": revision,
        "artifact_url": url,
        "pr_updated": args.update_pr,
    }
    (campaign / "transfer-publication.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
