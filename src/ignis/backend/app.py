"""Minimal admin console (FastAPI, server-rendered).

Endpoints:
- ``GET /``                  model list + gate summary
- ``GET /model/{name}``      train report + HA-vs-NILM table
- ``GET /api/models``        JSON registry (for scripts / future React UI)
- ``GET /api/health``        liveness

No template engine (f-strings keep it dependency-light). UI text in French
(single-language internal console).
"""

from __future__ import annotations

import html
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from ignis.nilm.config import settings

from .registry import ModelEntry, discover, get

app = FastAPI(title="Ignis admin", docs_url="/docs")


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang=fr><head><meta charset=utf-8>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:60rem}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0}"
        "th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left}"
        "th{background:#f4f4f4}.ok{color:#137333}.ko{color:#c5221f}"
        "a{color:#1a73e8;text-decoration:none}code{background:#f4f4f4;padding:.1rem .3rem}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1>{body}</body></html>"
    )


def _gate_cell(entry: ModelEntry) -> str:
    if entry.comparison is None:
        return "<span class=ko>non evalue</span>"
    n_pass = len(entry.gate_passed)
    total = n_pass + len(entry.gate_failed)
    cls = "ok" if total and n_pass == total else "ko"
    return f"<span class={cls}>{n_pass}/{total}</span>"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    entries = discover()
    if not entries:
        return _page("Ignis - modeles", "<p>Aucun modele entraine. Lancez <code>make train</code>.</p>")
    rows = "".join(
        f"<tr><td><a href='/model/{html.escape(e.name)}'>{html.escape(e.name)}</a></td>"
        f"<td>{html.escape(e.trained_at or '-')}</td>"
        f"<td>{'oui' if e.predictable else 'non'}</td>"
        f"<td>{_gate_cell(e)}</td></tr>"
        for e in entries
    )
    table = (
        "<table><tr><th>Modele</th><th>Entraine (fin periode)</th>"
        "<th>Predictible</th><th>Gate (F1>=0.8, err<=15%)</th></tr>"
        f"{rows}</table>"
    )
    return _page("Ignis - modeles", table)


def _report_table(report: dict) -> str:
    apps = report.get("appliances", {})
    rows = "".join(
        f"<tr><td>{html.escape(app)}</td>"
        f"<td>{a.get('labeled_hours', '-')}</td>"
        f"<td>{a.get('active_hours', '-')}</td>"
        f"<td>{a.get('active_ticks', '-')}</td></tr>"
        for app, a in apps.items()
    )
    head = (
        f"<p>Periode: <code>{html.escape(str(report.get('period_start')))}</code> &rarr; "
        f"<code>{html.escape(str(report.get('period_end')))}</code><br>"
        f"Grille: {report.get('grid_seconds')}s | fenetres: {report.get('n_windows')} | "
        f"epochs: {report.get('epochs_trained', '-')} | heures labellisees: {report.get('labeled_hours')}</p>"
    )
    return head + (
        "<h2>Couverture par appareil (signatures HA)</h2>"
        "<table><tr><th>Appareil</th><th>Heures labellisees</th>"
        f"<th>Heures actives</th><th>Ticks actifs</th></tr>{rows}</table>"
    )


def _comparison_table(comparison: dict) -> str:
    apps = comparison.get("appliances", {})
    rows = ""
    for app, m in apps.items():
        cls = "ok" if m.get("passes_gate") else "ko"
        err = m.get("energy_error")
        err_s = "inf" if err is None else f"{err:.3f}"
        rows += (
            f"<tr><td>{html.escape(app)}</td>"
            f"<td>{m.get('state_f1', '-'):.3f}</td>"
            f"<td>{err_s}</td>"
            f"<td class={cls}>{'PASS' if m.get('passes_gate') else 'FAIL'}</td></tr>"
        )
    return (
        "<h2>Comparaison HA vs NILM</h2>"
        "<table><tr><th>Appareil</th><th>State F1</th>"
        f"<th>Erreur energie</th><th>Gate</th></tr>{rows}</table>"
    )


@app.get("/model/{name}", response_class=HTMLResponse)
def model_detail(name: str) -> str:
    entry = get(name)
    if entry is None:
        return _page(
            "Introuvable", f"<p>Modele <code>{html.escape(name)}</code> introuvable. <a href='/'>Retour</a></p>"
        )
    body = "<p><a href='/'>&larr; Retour</a></p>"
    if entry.report:
        body += _report_table(entry.report)
    else:
        body += "<p>Pas de rapport d'entrainement.</p>"
    if entry.comparison:
        body += _comparison_table(entry.comparison)
    else:
        body += "<p>Pas encore evalue (<code>python -m eval --model ...</code>).</p>"
    return _page(f"Modele {name}", body)


@app.get("/api/models")
def api_models() -> JSONResponse:
    entries = discover()
    return JSONResponse(
        [
            {
                "name": e.name,
                "trained_at": e.trained_at,
                "predictable": e.predictable,
                "gate_passed": e.gate_passed,
                "gate_failed": e.gate_failed,
                "has_report": e.report is not None,
                "has_comparison": e.comparison is not None,
            }
            for e in entries
        ]
    )


@app.get("/api/models/history")
def api_models_history(limit: int = 500) -> JSONResponse:
    """Evolution curve: one point per logged retrain (portfolio surface C)."""
    from .dbapi import models_history

    return JSONResponse(models_history(limit=limit))


@app.get("/api/truth/recent")
def api_truth_recent(window: str = "15m") -> JSONResponse:
    """Latest per-appliance HA truth for the live NILM-vs-real diff (surface A)."""
    from .dbapi import truth_recent

    return JSONResponse(truth_recent(_parse_window(window)))


def _parse_window(window: str) -> int:
    """Parse '15m' / '30s' / '2h' -> seconds. Defaults to 900s on bad input."""
    units = {"s": 1, "m": 60, "h": 3600}
    try:
        return int(window[:-1]) * units[window[-1]]
    except (ValueError, KeyError, IndexError):
        return 900


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "models_dir": str(Path(settings.nilm_model_path)), "models": len(discover())}
