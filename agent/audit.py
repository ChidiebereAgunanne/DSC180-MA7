"""Batch leakage auditor built on the Claude Agent SDK.

Why this exists: the `/leakage-auditor` subagent inside Claude Code is the
right tool for reviewing *your* diff, once, while you watch. This script is
the right tool for reviewing *forty* repos, unattended, and getting back a
row per repo you can put in a dataframe. Same reviewer, different harness.

    pip install claude-agent-sdk
    python agent/audit.py ../student-repos/*/ --out audit.jsonl

Requires ANTHROPIC_API_KEY, or a logged-in Claude Code install.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ResultMessage,
    create_sdk_mcp_server,
    query,
    tool,
)

# --------------------------------------------------------------------------
# A custom tool. The model is bad at deciding whether three float columns
# sum to a fourth; numpy is perfect at it. Give the deterministic part to
# code and leave the judgement to the model.
# --------------------------------------------------------------------------


@tool(
    "check_identity",
    "Test whether any subset of candidate columns is an exact linear "
    "restatement of the target column. Returns the offending subsets.",
    {"csv_path": str, "target": str, "candidates": list},
)
async def check_identity(args: dict[str, Any]) -> dict[str, Any]:
    df = pd.read_csv(args["csv_path"])
    target = df[args["target"]].to_numpy(dtype=float)
    cols = [c for c in args["candidates"] if c in df.columns and c != args["target"]]

    # Least squares with an intercept column, so `cnt = a*x + b` is caught
    # as well as `cnt = casual + registered`. Rows with any NaN are dropped
    # first; lstsq on NaN returns garbage rather than raising.
    hits = []
    for r in (1, 2, 3):
        for subset in _combinations(cols, r):
            block = df[list(subset) + [args["target"]]].dropna()
            if len(block) < r + 2:
                continue
            X = np.column_stack([block[list(subset)].to_numpy(dtype=float),
                                 np.ones(len(block))])
            y = block[args["target"]].to_numpy(dtype=float)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = float(np.abs(X @ coef - y).max())
            if resid < 1e-6:
                hits.append({"columns": list(subset), "max_residual": resid})
    verdict = "LEAKAGE" if hits else "clean"
    return {
        "content": [
            {"type": "text", "text": json.dumps({"verdict": verdict, "identities": hits})}
        ]
    }


def _combinations(items, r):
    from itertools import combinations

    return combinations(items, r)


auditor_tools = create_sdk_mcp_server(
    name="audit", version="1.0.0", tools=[check_identity]
)

SYSTEM = """\
You review student data-science repos for methodology errors. You did not
write this code and you have no stake in it passing.

Check exactly four things:
  1. Target leakage: any feature that contains, restates, or derives from
     the target. Use the check_identity tool on the processed feature
     table rather than judging by column names.
  2. Temporal validity: splits by date not at random; every rolling or
     lag feature shifts before it aggregates; no transformer is fit on
     validation or test rows.
  3. Baseline: a mean or naive-persistence baseline reported next to
     every model score.
  4. Reproducibility: the pipeline runs from a documented command and
     random seeds are fixed.

Verify by reading files and running code. Do not accept a docstring or a
README claim as evidence.

Your final answer is a single JSON object matching the schema you were
given. No prose around it.
"""

# Enforced by the SDK: the final turn must validate against this schema and
# arrives parsed in ResultMessage.structured_output. Asking nicely in the
# prompt is not a contract; a schema is.
REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "repo": {"type": "string"},
        "leakage": {"type": "string", "enum": ["none", "suspected", "confirmed"]},
        "temporal_ok": {"type": "boolean"},
        "baseline_reported": {"type": "boolean"},
        "reproducible": {"type": "boolean"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "issue": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                },
                "required": ["file", "line", "issue", "why_it_matters"],
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["repo", "leakage", "temporal_ok", "baseline_reported",
                 "reproducible", "findings", "confidence"],
}


async def audit_repo(repo: Path) -> dict[str, Any]:
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM,
        cwd=str(repo),
        mcp_servers={"audit": auditor_tools},
        allowed_tools=[
            "Read",
            "Grep",
            "Glob",
            "Bash",
            "mcp__audit__check_identity",
        ],
        # allowed_tools is the boundary. Anything not listed is denied
        # automatically in a headless run, so the auditor can read and run
        # but has no Edit or Write to grant.
        permission_mode="default",
        max_turns=40,
        model="claude-sonnet-4-6",
        output_format={"type": "json_schema", "schema": REPORT_SCHEMA},
    )

    # Only the ResultMessage carries the final answer. Concatenating every
    # intermediate TextBlock ("Let me look at config.py first...") and
    # json.loads-ing the lot fails on the first repo that thinks out loud.
    final: ResultMessage | None = None
    async for message in query(prompt=f"Audit the repo at {repo}.", options=options):
        if isinstance(message, ResultMessage):
            final = message

    if final is None or final.is_error:
        return {"repo": repo.name, "error": "run_failed",
                "raw": (final.result or "")[:500] if final else ""}
    if final.structured_output is not None:
        return {"repo": repo.name, **final.structured_output}
    try:
        # Belt and braces: schema enforcement can be unavailable on some
        # models; fall back to parsing the final text.
        return {"repo": repo.name, **json.loads((final.result or "").strip())}
    except json.JSONDecodeError:
        # Never let one malformed reply kill a 40-repo run.
        return {"repo": repo.name, "error": "unparseable", "raw": (final.result or "")[:500]}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("audit.jsonl"))
    ap.add_argument("--concurrency", type=int, default=4)
    ns = ap.parse_args()

    sem = asyncio.Semaphore(ns.concurrency)

    async def bounded(repo: Path) -> dict[str, Any]:
        async with sem:
            return await audit_repo(repo)

    results = await asyncio.gather(*(bounded(r) for r in ns.repos))
    with ns.out.open("w") as fh:
        for row in results:
            fh.write(json.dumps(row) + "\n")

    flagged = sum(r.get("leakage") in {"suspected", "confirmed"} for r in results)
    print(f"audited {len(results)} repos; {flagged} flagged -> {ns.out}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
