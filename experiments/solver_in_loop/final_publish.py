"""Publish the bounded correction study after its cluster report completes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

START = "<!-- correction-final-start -->"
END = "<!-- correction-final-end -->"
REPO = "pasteurlabs/mosaic"


def merge_section(body: str, section: str) -> str:
    """Replace only this campaign's marked section, preserving other PR text."""
    block = f"{START}\n{section.strip()}\n{END}"
    if START in body or END in body:
        if body.count(START) != 1 or body.count(END) != 1:
            raise ValueError("ambiguous correction section markers")
        start, end = body.index(START), body.index(END)
        if end < start:
            raise ValueError("reversed correction section markers")
        return body[:start] + block + body[end + len(END) :]
    return body.rstrip() + "\n\n" + block + "\n"


def main() -> None:
    """Run on the login node: copy finished artifacts and update the PR body."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--gh", default="/home/andrinr/.local/bin/gh")
    args = parser.parse_args()
    campaign = args.campaign
    report = campaign / "report"
    summary = json.loads((report / "report.json").read_text())
    if not summary.get("report_complete"):
        raise ValueError("report is not finalized")
    checkout = campaign / "publication-checkout"
    if not checkout.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--single-branch",
                "--branch",
                "pr-116-local-results",
                "git@github.com:pasteurlabs/mosaic.git",
                str(checkout),
            ],
            check=True,
        )
    git = ["git", "-C", str(checkout)]
    if subprocess.check_output([*git, "status", "--porcelain"], text=True).strip():
        raise RuntimeError("publication checkout has uncommitted changes")
    subprocess.run([*git, "pull", "--ff-only"], check=True)
    folder = Path("pr-116") / campaign.name.removeprefix("pr116-")
    destination = checkout / folder
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(report, destination / "report", dirs_exist_ok=True)
    for name in (
        "plan.json",
        "report-input.json",
        "extensions.json",
        "selection.json",
        "FINAL_PROTOCOL.md",
        "source.tar",
        "source.sha256",
        "images.json",
        "status.json",
        "submissions.jsonl",
    ):
        if (campaign / name).exists():
            shutil.copy2(campaign / name, destination / name)
    if (campaign / "configs").exists():
        shutil.copytree(
            campaign / "configs", destination / "configs", dirs_exist_ok=True
        )
    # Preserve every outcome, including failed candidates; large field arrays and
    # checkpoints remain in the campaign, while the report contains fixed fields.
    for outcome in sorted((campaign / "results").glob("*/outcome.json")):
        target = destination / "outcomes" / outcome.parent.name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(outcome, target / "outcome.json")
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
                "results: close bounded correction comparison",
            ],
            check=True,
        )
    subprocess.run([*git, "push", "origin", "HEAD:pr-116-local-results"], check=True)
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    url = f"https://raw.githubusercontent.com/{REPO}/{revision}/{folder}"
    section = (report / "PRsection.md").read_text().replace("ARTIFACT_URL", url)
    section = section.replace(
        "ARTIFACT_TREE", f"https://github.com/{REPO}/tree/{revision}/{folder}"
    )
    api = [args.gh, "api", f"repos/{REPO}/pulls/116"]
    current = json.loads(subprocess.check_output(api, text=True))
    body = merge_section(current["body"], section)
    request = campaign / "publication-request.json"
    request.write_text(json.dumps({"body": body}))
    latest = json.loads(subprocess.check_output(api, text=True))
    if latest["body"] != current["body"]:
        raise RuntimeError("PR changed during publication; preserving intervening edit")
    subprocess.run(
        [*api, "--method", "PATCH", "--input", str(request), "--jq", ".html_url"],
        check=True,
    )
    (campaign / "publication.json").write_text(
        json.dumps({"published": True, "artifact_commit": revision}, indent=2)
    )


if __name__ == "__main__":
    main()
