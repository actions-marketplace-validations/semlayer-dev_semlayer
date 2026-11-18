# SPDX-License-Identifier: FSL-1.1-ALv2
"""The engine's query path must run where `mcp` and threads do not exist.

A browser runtime (Pyodide) has neither the `mcp` package nor the ability to
start a thread. Three things are pinned here: the query path imports without
`mcp`; the query watchdog degrades, and the limits it reports say so; and the
additive progress events (`item`, `found`, enrich phases) are tolerated by the
existing consumer.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import duckdb
import pytest

from semlayer import session as sess
from semlayer.pipeline import infer_with_report
from semlayer.source import DuckDBSource

MESSY = Path(__file__).resolve().parent.parent / "fixtures" / "build" / "messy_mart.duckdb"


# --- A1: no mcp ------------------------------------------------------------

def test_query_path_imports_without_mcp():
    """Run in a fresh interpreter with `mcp` made unimportable."""
    code = textwrap.dedent("""
        import sys
        sys.modules["mcp"] = None          # `import mcp...` raises ImportError
        from semlayer.cq import answer, generate
        from semlayer import agent_query, mcp_server
        doc = {"semantic_layer": {"tables": [{"name": "t", "description": "orders",
               "columns": [{"name": "id", "sql_type": "INTEGER"}]}]}}
        out = agent_query.context_for_question(doc, "orders")
        assert isinstance(out, dict), out
        try:
            mcp_server.build_server(doc)
        except ImportError:
            print("OK")
        else:
            raise SystemExit("build_server did not fail without mcp")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("OK")


# --- A2: watchdog degrades -------------------------------------------------

def _no_threads(monkeypatch):
    def boom(self):
        raise RuntimeError("can't start new thread")
    monkeypatch.setattr(threading.Timer, "start", boom)


def test_query_capped_runs_without_a_thread(monkeypatch):
    _no_threads(monkeypatch)
    src = DuckDBSource(duckdb.connect(":memory:"))
    cols, rows, trunc = src.query_capped("SELECT 1 AS x", timeout_s=5)
    assert (cols, rows, trunc) == (["x"], [(1,)], False)
    assert src.last_timeout_armed is False


def test_run_says_no_time_limit_when_the_watchdog_could_not_start(monkeypatch):
    from semlayer import agent_query
    _no_threads(monkeypatch)
    src = DuckDBSource(duckdb.connect(":memory:"))
    doc = {"semantic_layer": {"tables": []}}
    out = agent_query.run(doc, src, "SELECT 1 AS x")
    assert out["ran"] is True and out["rows"] == [[1]]
    assert out["limits"]["timeout_s"] is None
    assert "no time limit" in out["limits"]["in_words"]


def test_run_still_states_the_limit_when_the_watchdog_started():
    from semlayer import agent_query
    src = DuckDBSource(duckdb.connect(":memory:"))
    out = agent_query.run({"semantic_layer": {"tables": []}}, src, "SELECT 1 AS x")
    assert out["limits"]["timeout_s"] == agent_query.RUN_SQL_TIMEOUT_S
    assert "no time limit" not in out["limits"]["in_words"]


def test_cq_execute_runs_without_a_thread(monkeypatch):
    from semlayer.cq.answer import _execute
    _no_threads(monkeypatch)
    con = duckdb.connect(":memory:")
    assert _execute(con, "SELECT 2") == [(2,)]


# --- A3: additive progress events -----------------------------------------

@pytest.fixture(scope="module")
def events():
    ev: list[dict] = []
    src = DuckDBSource(duckdb.connect(str(MESSY), read_only=True))
    infer_with_report(src, None, progress=ev.append)
    return ev


def test_profile_units_name_their_table(events):
    units = [e for e in events if e["stage"] == "profile" and e["event"] == "unit"]
    assert units and all(isinstance(e.get("item"), str) and e["item"] for e in units)


def test_link_found_precedes_link_done(events):
    link = [e["event"] for e in events if e["stage"] == "link"]
    assert link.count("found") == 1 and link.index("found") < link.index("done")
    found = next(e for e in events if e["event"] == "found")
    assert len(found["relationships"]) == 37
    assert set(found["relationships"][0]) == {
        "from_table", "from_columns", "to_table", "to_columns", "confidence"}


def test_enrich_emits_a_few_labelled_units(events):
    units = [e for e in events if e["stage"] == "enrich" and e["event"] == "unit"]
    assert 3 <= len(units) <= 6
    assert all(e["label"] and 0 <= e["within"] <= 1 for e in units)


def test_inferjob_tolerates_found_and_item():
    job = sess.InferJob("duckdb:w.duckdb", None, None, with_llm=False)
    job.progress({"stage": "link", "event": "start"})
    job.progress({"stage": "link", "event": "unit", "done": 1, "total": 2,
                  "within": 0.5, "label": "checking inclusion: 1 of 2"})
    before = (job.stage, job.label, job.fraction)
    job.progress({"stage": "link", "event": "found", "relationships": []})
    assert (job.stage, job.label, job.fraction) == before     # not a boundary
    job.progress({"stage": "profile", "event": "unit", "done": 0, "total": 3,
                  "within": 0.0, "label": "profiling table 1 of 3", "item": "orders"})
    assert job.tables_total == 3 and job.label == "profiling table 1 of 3"
    job.progress({"stage": "weird", "event": "future_kind", "extra": 1})   # no raise


def test_ticker_item_is_optional():
    from semlayer.progress import ticker
    got: list[dict] = []
    tick = ticker(got.append, "profile", 2, "profiling table")
    tick(0)
    tick(1, "orders")
    assert "item" not in got[0] and got[1]["item"] == "orders"
