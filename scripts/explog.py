"""Experiment log: every attempt at the moon landing goes through MLflow.

The tracking store is the repo-local mlflow.db (sqlite) with artifacts
in ./mlruns, both gitignored. Each
run carries, besides params/metrics, three free-text tags that make the log
readable as a lab notebook:

    hypothesis   what this attempt expects to improve, and why
    outcome      what happened (filled at the end)
    diagnosis    what seemed to be the problem (filled at the end)

EXPERIMENTS.md mirrors every finished run as a short entry so the history
survives without the MLflow UI.

    python scripts/explog.py              # table of all attempts
    python scripts/explog.py show <run>   # params, metrics, notes of one run
    python scripts/explog.py diagnosis <run> "what went wrong"   # annotate
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # the full dashboard
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKING_URI = "sqlite:///" + os.path.join(REPO, "mlflow.db")  # artifacts: ./mlruns
EXPERIMENT = "moon-landing"
MD_LOG = os.path.join(REPO, "EXPERIMENTS.md")


def setup(experiment: str = EXPERIMENT) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment,
                                 artifact_location="file:" + os.path.join(REPO, "mlruns"))
    mlflow.set_experiment(experiment)


def start(name: str, params: dict, hypothesis: str, tags: dict | None = None):
    setup()
    run = mlflow.start_run(run_name=name)
    flat = {k: (v if isinstance(v, (int, float, str, bool)) else repr(v))
            for k, v in params.items()}
    items = list(flat.items())
    for i in range(0, len(items), 90):  # MLflow caps a param batch at 100
        mlflow.log_params(dict(items[i:i + 90]))
    mlflow.set_tags({"hypothesis": hypothesis, **(tags or {})})
    return run


def note(text: str) -> None:
    """Append a timestamped line to the run's notes (shown in the UI)."""
    run = mlflow.active_run()
    if run is None:
        return
    client = mlflow.tracking.MlflowClient()
    cur = client.get_run(run.info.run_id).data.tags.get("mlflow.note.content", "")
    stamp = time.strftime("%H:%M:%S")
    mlflow.set_tag("mlflow.note.content", f"{cur}\n[{stamp}] {text}".strip())


def finish(outcome: str, diagnosis: str, status: str = "FINISHED",
           summary_metrics: dict | None = None) -> None:
    """Close the run and mirror it into EXPERIMENTS.md."""
    run = mlflow.active_run()
    if run is None:
        return
    mlflow.set_tags({"outcome": outcome, "diagnosis": diagnosis})
    if summary_metrics:
        mlflow.log_metrics({k: float(v) for k, v in summary_metrics.items()})
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run.info.run_id).data
    name = data.tags.get("mlflow.runName", run.info.run_id)
    hyp = data.tags.get("hypothesis", "")
    metrics = summary_metrics or {}
    lines = [
        f"\n## {time.strftime('%Y-%m-%d %H:%M')} - {name}  (`{run.info.run_id[:8]}`, {status})\n",
        f"**Hypothesis:** {hyp}\n",
        f"**Outcome:** {outcome}\n",
        f"**Diagnosis:** {diagnosis}\n",
    ]
    if metrics:
        lines.append("**Key metrics:** " + ", ".join(
            f"{k}={v:.3g}" for k, v in metrics.items()) + "\n")
    with open(MD_LOG, "a") as f:
        f.write("\n".join(lines))
    mlflow.end_run(status=status)


# --- CLI ----------------------------------------------------------------------

def _table() -> None:
    setup()
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        print("no runs yet")
        return
    runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time ASC"])
    if runs.empty:
        print("no runs yet")
        return
    mcols = [c for c in runs.columns if c.startswith("metrics.")]
    for _, r in runs.iterrows():
        rid = r["run_id"][:8]
        name = r.get("tags.mlflow.runName", "")
        print(f"{str(r['start_time'])[:16]}  {rid}  {r['status']:<9} {name}")
        for key in ("hypothesis", "outcome", "diagnosis"):
            v = r.get(f"tags.{key}")
            if isinstance(v, str) and v:
                print(f"    {key:<10} {v}")
        keep = {c[8:]: r[c] for c in mcols
                if r[c] == r[c] and c[8:].startswith(("eval/", "final/"))}
        if keep:
            print("    metrics    " + ", ".join(f"{k}={v:.3g}" for k, v in sorted(keep.items())))


def _show(prefix: str) -> None:
    setup()
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    runs = mlflow.search_runs([exp.experiment_id])
    hit = runs[runs["run_id"].str.startswith(prefix)]
    if hit.empty:
        print("no such run")
        return
    r = hit.iloc[0]
    for c in sorted(hit.columns):
        v = r[c]
        if v == v and v not in ("", None):
            print(f"{c}: {v}")


def _set_tag(prefix: str, key: str, text: str) -> None:
    setup()
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    runs = mlflow.search_runs([exp.experiment_id])
    hit = runs[runs["run_id"].str.startswith(prefix)]
    if hit.empty:
        raise SystemExit("no such run")
    rid = hit.iloc[0]["run_id"]
    mlflow.tracking.MlflowClient().set_tag(rid, key, text)
    with open(MD_LOG, "a") as f:
        f.write(f"\n_{key} update for `{rid[:8]}` ({time.strftime('%Y-%m-%d %H:%M')}):_ {text}\n")
    print(f"{key} set on {rid[:8]}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "show":
        _show(sys.argv[2])
    elif len(sys.argv) > 3 and sys.argv[1] in ("diagnosis", "outcome", "hypothesis"):
        _set_tag(sys.argv[2], sys.argv[1], " ".join(sys.argv[3:]))
    else:
        _table()
