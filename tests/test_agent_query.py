"""Answering a question: what may run, what may not, and what is disclosed.

The three claims being tested are the ones a user has to be able to trust:
nothing here can change their warehouse, a query that the linter rejects is
not executed at all, and an answer says what it is resting on.

Real DuckDB throughout -- a mocked executor would have agreed with a
`query_capped` that never applied its cap.
"""

from __future__ import annotations

import pytest

from semlayer import agent_query as aq


def _root():
    from pathlib import Path
    return Path(__file__).resolve().parents[1]


GOLD = _root() / "fixtures" / "golds" / "messy_mart.yaml"


@pytest.fixture(scope="module")
def doc():
    from semlayer.session import load_document
    return load_document(GOLD)


@pytest.fixture(scope="module")
def source():
    """The real warehouse behind the gold layer, or a skip if unbuilt."""
    import duckdb

    from semlayer.source import DuckDBSource
    warehouse = _root() / "fixtures" / "build" / "messy_mart.duckdb"
    if not warehouse.is_file():
        pytest.skip("fixtures not built: python fixtures/build.py messy_mart")
    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        yield DuckDBSource(con)
    finally:
        con.close()


# --- what may run ---------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "select 1",
    "select * from ord_hdr",
    "with a as (select 1 as x) select * from a",
    "select * from ord_hdr union all select * from ord_hdr",
    "select count(*) from ord_hdr where sts_cd = 'C'",
    "(select 1)",
])
def test_a_question_is_allowed(sql):
    assert aq.select_only(sql) is None, sql


@pytest.mark.parametrize("sql", [
    "delete from ord_hdr",
    "update ord_hdr set sts_cd = 'C'",
    "insert into ord_hdr values (1)",
    "drop table ord_hdr",
    "create table x as select 1",
    "alter table ord_hdr add column c int",
    "truncate ord_hdr",
    "grant select on ord_hdr to bob",
    "merge into ord_hdr using s on s.id = ord_hdr.ord_id when matched then delete",
    "copy ord_hdr to 'out.csv'",
    "select * into other from ord_hdr",
])
def test_a_change_is_refused(sql):
    """Not a blocklist of words -- a parse. The statement type decides."""
    refused = aq.select_only(sql)
    assert refused is not None, sql
    assert "SELECT" in refused or "not allowed" in refused


def test_two_statements_are_refused_even_if_the_first_is_innocent():
    """The injection shape: a SELECT that a string check would wave through."""
    refused = aq.select_only("select 1; delete from ord_hdr")
    assert refused is not None
    assert "one" in refused


def test_a_literal_mentioning_a_write_is_still_a_question():
    """`delete` appears in plenty of harmless places. A parse knows that."""
    assert aq.select_only(
        "select * from ord_hdr where sts_cd like '%delete from%'") is None


def test_sql_that_does_not_parse_says_so_rather_than_running():
    refused = aq.select_only("sel ect ((")
    assert refused is not None
    assert "parse" in refused


def test_the_dialect_is_the_connector_family():
    """Snowflake's QUALIFY is a parse error under the default dialect, and an
    agent told its valid SQL is invalid will rewrite working SQL."""
    sql = "select a, row_number() over (order by a) rn from t qualify rn = 1"
    assert aq.select_only(sql, aq.dialect_for("snowflake")) is None
    assert aq.dialect_for("postgres") == "postgres"
    assert aq.dialect_for(None) == "duckdb"


# --- lint first, then execute --------------------------------------------

def test_a_lint_error_is_not_executed(doc, source):
    out = aq.run(doc, source, "select cust_seg from cust_mstr")
    assert out["ran"] is False
    assert "unknown_column" in out["findings"]
    assert "repair" in out
    assert "rows" not in out


def test_a_warning_does_not_block(doc, source):
    """A fan-out warning on a query deliberately counting rows is noise, and
    an agent that cannot proceed past one rewrites correct SQL to silence it."""
    out = aq.run(doc, source, "select count(*) as n from cust_mstr")
    assert out["ran"] is True
    assert out["warnings"]
    assert "scd2" in out["warnings"]


def test_a_refusal_never_reaches_the_warehouse(doc, source):
    out = aq.run(doc, source, "delete from cust_mstr")
    assert out["ran"] is False
    assert "refused" in out
    # and the table is still there
    assert aq.run(doc, source, "select count(*) as n from cust_mstr")["ran"]


def test_a_good_query_returns_named_columns_and_rows(doc, source):
    out = aq.run(doc, source,
                 "select seg_cd, count(*) as n from cust_mstr "
                 "where is_curr_flg = 1 group by 1 order by 2 desc")
    assert out["ran"] is True
    assert out["columns"] == ["seg_cd", "n"]
    assert len(out["rows"]) == 3
    assert out["truncated"] is False
    assert out["limits"]["rows"] == 200
    assert out["limits"]["timeout_s"] == 600
    # A sentence the agent repeats, so it does not have to assemble one from
    # two numbers and guess at the units.
    assert out["limits"]["in_words"] == "at most 200 rows, and up to 600s on the warehouse"


def test_the_limits_read_correctly_with_the_clock_off(doc, source):
    out = aq.run(doc, source, "select ord_id from ord_hdr", timeout_s=None)
    assert "no time limit" in out["limits"]["in_words"]


def test_the_row_cap_truncates_and_says_so(doc, source):
    out = aq.run(doc, source, "select ord_id from ord_hdr", limit=5)
    assert out["row_count"] == 5
    assert out["truncated"] is True
    assert "truncated" in out["next"]


def test_values_come_back_as_json_not_as_python_repr(doc, source):
    """A Decimal through `default=str` becomes the STRING "13006346.06", and
    the agent is left doing arithmetic on text."""
    import json
    out = aq.run(doc, source, "select sum(tot_amt) as amt from ord_hdr")
    assert isinstance(out["rows"][0][0], float)
    json.dumps(out)              # no default= rescue needed


def test_a_warehouse_error_comes_back_with_repair_context(doc, source):
    """The linter does not know every function, so execution still fails --
    and an agent shown the error fixes its own SQL, which is what the
    benchmark's repair loop measured."""
    out = aq.run(doc, source, "select no_such_function(ord_id) from ord_hdr")
    assert out["ran"] is False
    assert "error" in out
    assert out["repair"]["sql"]


def test_the_sql_is_never_rewritten(doc, source):
    sql = "select ord_id from ord_hdr"
    out = aq.run(doc, source, sql, limit=3)
    assert out["sql"] == sql, "no LIMIT appended, no filter injected"


# --- what the answer is resting on ---------------------------------------

def test_context_for_a_question_selects_tables_and_renders_them(doc):
    out = aq.context_for_question(doc, "revenue by region last quarter")
    assert "ord_hdr" in out["tables"]
    assert len(out["context"]) > 1000
    assert "check_sql" in out["next"]


def test_open_questions_are_scoped_to_the_tables(doc):
    everywhere = aq.open_questions(doc)
    scoped = aq.open_questions(doc, ["cust_mstr"])
    assert everywhere
    assert len(scoped) < len(everywhere)
    assert all(q["decided_by"] == "nobody" for q in scoped)


def test_a_gating_kind_is_marked_as_one(doc):
    """A decode guess a metric's filter depends on is a different kind of
    uncertainty from a label nobody has blessed, and the agent has to be able
    to tell the user which it is."""
    from semlayer import review
    kinds = {q["kind"]: q["gates_behaviour"] for q in aq.open_questions(doc)}
    assert kinds, kinds
    for kind, gates in kinds.items():
        assert gates == (kind in review.GATING_KINDS), kind


def test_an_answer_carries_the_assumptions_it_rests_on(doc, source):
    out = aq.run(doc, source,
                 "select seg_cd, count(*) as n from cust_mstr "
                 "where is_curr_flg = 1 group by 1")
    assert out["assumptions"]
    assert {a["decided_by"] for a in out["assumptions"]} == {"nobody"}
    assert all(a["claim"] for a in out["assumptions"])


def test_tables_an_answer_touched_are_read_from_the_sql(doc):
    assert aq.tables_in("select * from ord_hdr join cust_mstr using (cust_id)") == \
        ["ord_hdr", "cust_mstr"]
    assert aq.tables_in("not sql at all ((") == []


@pytest.mark.parametrize(("sql", "label"), [
    ("truncate ord_hdr", "TRUNCATE TABLE"),
    ("grant select on ord_hdr to bob", "GRANT"),
    ("create table x as select 1", "CREATE"),
    ("vacuum", "VACUUM"),
    ("call some_proc()", "CALL"),
])
def test_the_refusal_names_the_statement_in_sql_words(sql, label):
    """"TRUNCATETABLE is not allowed" reads like a typo rather than a rule,
    and an unparseable statement arrives as sqlglot's generic Command, where
    the keyword the user wrote is the first token."""
    refused = aq.select_only(sql)
    assert refused is not None
    assert refused.startswith(label), refused


# --- why the warehouse said no -------------------------------------------
#
# A statement the warehouse PARSED and then refused for want of a grant is
# not a broken statement, and the difference decides what the agent should
# do with it: run it unchanged somewhere that has the grant, or rewrite it.
# That used to be the model's call, made by reading a vendor error string.
# These are the strings, taken from what each warehouse actually says.

class _Boom(Exception):
    """A driver exception, with whatever attributes that driver sets."""

    def __init__(self, message, **attrs):
        super().__init__(message)
        for k, v in attrs.items():
            setattr(self, k, v)


@pytest.mark.parametrize("kind,exc", [
    # Postgres and Redshift: SQLSTATE 42501 is insufficient_privilege.
    ("permission", _Boom("permission denied for table ord_hdr", sqlstate="42501")),
    # psycopg2 spells the same field `pgcode`.
    ("permission", _Boom("permission denied for table ord_hdr", pgcode="42501")),
    # Snowflake 003001.
    ("permission", _Boom("003001: Insufficient privileges to operate on table "
                         "'ORD_HDR'", errno=3001)),
    # BigQuery, through google.api_core, which carries the status as `code`.
    ("permission", _Boom("403 Access Denied: Table p:d.t: User does not have "
                         "permission to query table p:d.t.", code=403)),
    # No code at all: the message is the only signal left.
    ("permission", _Boom("User is not authorized to perform this action")),
    # Snowflake 002003, which is ONE error for two causes on purpose, so that
    # a role cannot discover objects it may not see.
    ("permission_or_missing",
     _Boom("002003: SQL compilation error: Object 'ORD_HDR' does not exist or "
           "not authorized.", errno=2003)),
    ("permission_or_missing",
     _Boom("Object 'ORD_HDR' does not exist or not authorized.")),
    # And the negatives, which matter more: a missing table is NOT a grant
    # problem, and routing it to another tool would waste a round trip and
    # then report the same failure.
    ("statement", _Boom("Catalog Error: Table with name foo does not exist!")),
    ("statement", _Boom('Parser Error: syntax error at or near "slect"')),
    ("statement", _Boom("Binder Error: No function matches the given name and "
                        "argument types 'no_such_function(INTEGER)'")),
    ("statement", _Boom("Out of Memory Error: could not allocate block")),
])
def test_a_refusal_is_classified_here_not_by_the_model(kind, exc):
    why = aq.classify_warehouse_error(exc)
    assert why["kind"] == kind, why
    assert why["retryable_elsewhere"] is (kind != "statement")


def test_the_classification_says_what_decided_it():
    """So a wrong call can be read off the result rather than reconstructed
    from the warehouse's logs."""
    assert aq.classify_warehouse_error(
        _Boom("nope", sqlstate="42501"))["signal"] == "SQLSTATE 42501"
    assert aq.classify_warehouse_error(
        _Boom("nope", errno=3001))["signal"] == "warehouse error 3001"
    assert aq.classify_warehouse_error(
        _Boom("Access Denied: no"))["signal"] == "access denied"
    assert aq.classify_warehouse_error(_Boom("Parser Error"))["signal"] is None


class _Refusing:
    """A source whose connection is open and whose warehouse says no."""

    def __init__(self, exc):
        self.exc = exc

    def query_capped(self, sql, limit=None, timeout_s=None):
        raise self.exc


def test_a_grant_problem_is_not_offered_a_repair(doc):
    """`repair` is how correct SQL gets rewritten until the message changes.
    The statement passed check_sql and the warehouse parsed it."""
    out = aq.run(doc, _Refusing(_Boom("permission denied for table ord_hdr",
                                      sqlstate="42501")),
                 "select ord_id from ord_hdr")
    assert out["ran"] is False
    assert out["error_kind"]["kind"] == "permission"
    assert "repair" not in out, "nothing here is broken enough to repair"
    assert "VERBATIM" in out["next"]
    assert "Do NOT" in out["next"] and "re-draft" in out["next"]


def test_an_ambiguous_refusal_sends_the_agent_both_ways(doc):
    """Snowflake does not say which it was, so neither do we: try elsewhere,
    and if it fails there too the name is wrong."""
    out = aq.run(doc, _Refusing(_Boom("002003: Object 'ORD_HDR' does not exist "
                                      "or not authorized.", errno=2003)),
                 "select ord_id from ord_hdr")
    assert out["error_kind"]["kind"] == "permission_or_missing"
    assert "VERBATIM" in out["next"], "the other tool may have the role"
    assert "get_tables" in out["repair"]["rule"], "and the name may be wrong"


def test_a_broken_statement_still_gets_the_repair_loop(doc, source):
    """The path the benchmark's repair loop is built on must not have moved."""
    out = aq.run(doc, source, "select no_such_function(ord_id) from ord_hdr")
    assert out["ran"] is False
    assert out["error_kind"]["kind"] == "statement"
    assert out["error_kind"]["retryable_elsewhere"] is False
    assert out["repair"]["sql"] == "select no_such_function(ord_id) from ord_hdr"
    assert "VERBATIM" not in out["next"], "another tool will refuse it too"
