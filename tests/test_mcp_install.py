"""Writing semlayer into an MCP client's config.

Every assertion here is a mistake someone actually made on a real machine,
mostly in one evening:

  * a `"semlayer": {...}` snippet pasted at the TOP LEVEL of ~/.claude.json,
    because the instruction did not say it belonged inside `mcpServers`;
  * a bare `semlayer` as the command, which resolves in a terminal and not
    in a client launched from a dock;
  * an argv joined into one string, naming an executable that cannot exist.

The point of the command these test is that none of those are reachable.
"""

from __future__ import annotations

import json

import pytest

from semlayer import mcp_install as mi
from semlayer.errors import SemlayerError


def _client(tmp_path, name="claude-code", key="mcpServers"):
    return mi.Client(name, name.title(), tmp_path / "cfg.json", key)


# --- the entry itself -------------------------------------------------------

def test_the_command_is_absolute(tmp_path, monkeypatch):
    """A bare name resolves in your shell and not in the client's process."""
    fake = tmp_path / "bin" / "semlayer"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake.parent))

    e = mi.entry()
    assert e["command"] == str(fake)
    assert e["args"] == ["mcp"]


def test_arguments_stay_arguments(tmp_path, monkeypatch):
    """Never one joined string: that names an executable that cannot exist."""
    monkeypatch.setenv("PATH", str(tmp_path))      # nothing on PATH
    e = mi.entry()
    assert " " not in e["command"], "a command with a space in it is not a command"
    assert e["args"][-1] == "mcp"
    # With no launcher found, the fallback must still be runnable.
    assert e["command"].endswith("python") or "python" in e["command"]


def test_a_layer_is_recorded_absolutely(tmp_path, monkeypatch):
    """A relative path means something different in the client's cwd."""
    monkeypatch.setenv("PATH", str(tmp_path))
    layer = tmp_path / "messy_mart.yaml"
    layer.write_text("semantic_layer: {}\n")
    monkeypatch.chdir(tmp_path)

    e = mi.entry("messy_mart.yaml")
    assert e["args"][-1] == str(layer)


# --- merging into a file that is not ours -----------------------------------

def test_other_servers_and_unrelated_keys_survive(tmp_path):
    c = _client(tmp_path)
    c.path.write_text(json.dumps({
        "mcpServers": {"github": {"command": "gh-mcp"}},
        "numStartups": 41,
        "projects": {"/home/u/x": {"allowedTools": []}},
    }))

    mi.apply(c)
    after = json.loads(c.path.read_text())

    assert after["mcpServers"]["github"] == {"command": "gh-mcp"}
    assert after["numStartups"] == 41
    assert after["projects"] == {"/home/u/x": {"allowedTools": []}}
    assert "semlayer" in after["mcpServers"]


def test_the_entry_goes_INSIDE_mcpservers(tmp_path):
    """The mistake that started this: a top-level key is silently inert."""
    c = _client(tmp_path)
    mi.apply(c)
    after = json.loads(c.path.read_text())
    assert "semlayer" in after["mcpServers"]
    assert "semlayer" not in {k for k in after if k != "mcpServers"}


def test_a_client_that_calls_the_key_something_else(tmp_path):
    """VS Code says `servers`. Writing the wrong key parses and does nothing."""
    c = _client(tmp_path, "vscode", key="servers")
    mi.apply(c)
    after = json.loads(c.path.read_text())
    assert "semlayer" in after["servers"]
    assert "mcpServers" not in after


def test_a_missing_config_is_created_with_its_directory(tmp_path):
    c = mi.Client("cline", "Cline", tmp_path / "deep" / "nested" / "cfg.json")
    assert not c.exists
    mi.apply(c)
    assert json.loads(c.path.read_text())["mcpServers"]["semlayer"]


def test_invalid_json_is_refused_not_overwritten(tmp_path):
    """It is the user's file and probably holds other servers."""
    c = _client(tmp_path)
    c.path.write_text('{"mcpServers": {"github": {}},,,')
    with pytest.raises(SemlayerError, match="not valid JSON"):
        mi.apply(c)
    assert c.path.read_text().endswith(",,,"), "the file must be left alone"


# --- the plan, and saying what will happen before it happens ----------------

def test_dry_run_writes_nothing(tmp_path):
    c = _client(tmp_path)
    before = json.dumps({"mcpServers": {}})
    c.path.write_text(before)

    result = mi.plan(c)
    assert result["after"]["args"][-1] == "mcp"
    assert c.path.read_text() == before, "plan must not touch the file"


def test_rewriting_the_same_entry_is_reported_as_unchanged(tmp_path):
    c = _client(tmp_path)
    mi.apply(c)
    again = mi.plan(c)
    assert again["unchanged"] is True
    assert again["before"] == again["after"]


def test_the_plan_names_what_it_is_not_touching(tmp_path):
    c = _client(tmp_path)
    c.path.write_text(json.dumps({"mcpServers": {"a": {}, "b": {}}}))
    assert mi.plan(c)["other_servers"] == ["a", "b"]


# --- knowing what it does not know ------------------------------------------

def test_an_unknown_client_lists_the_known_ones(tmp_path):
    with pytest.raises(SemlayerError) as e:
        mi.find("windsurf")
    assert "claude-code" in str(e.value.hint)


def test_every_known_client_has_a_distinct_id_and_path():
    cs = mi.clients()
    assert len({c.id for c in cs}) == len(cs)
    assert len({str(c.path) for c in cs}) == len(cs)
    assert {c.id for c in cs} == {"claude-code", "claude-desktop", "cursor",
                                  "cline", "vscode", "codex"}


def test_vscode_is_the_portable_file_not_the_deprecated_workspace_one(monkeypatch):
    """`.vscode/mcp.json` is editor-managed, workspace-scoped, and VS Code's
    own docs mark it for removal in February 2027. The portable Copilot file
    is user-level and is what new configurations are pointed at."""
    monkeypatch.delenv("COPILOT_HOME", raising=False)
    c = mi.find("vscode")
    assert c.path.name == "mcp-config.json"
    assert ".copilot" in str(c.path)
    assert ".vscode" not in str(c.path)
    assert c.key == "mcpServers", "the portable format, same as every other client"


def test_copilot_home_moves_the_vscode_config(monkeypatch, tmp_path):
    monkeypatch.setenv("COPILOT_HOME", str(tmp_path / "elsewhere"))
    assert mi.find("vscode").path == tmp_path / "elsewhere" / "mcp-config.json"


def test_claude_code_warns_about_the_file_it_shares_with_itself():
    """It keeps session state in the same JSON and can overwrite an edit."""
    warn = mi.warning_for(mi.find("claude-code"))
    assert warn and "overwrite" in warn
    assert mi.warning_for(mi.find("cursor")) is None


def test_the_restart_note_says_what_restart_means():
    assert "NEW session" in mi.RESTART_NOTE
    assert "startup" in mi.RESTART_NOTE


# --- Codex, which keeps TOML ------------------------------------------------
#
# A different FORMAT, not just a different path, and the reason it is worth
# the code: OpenAI's MCP docs say Codex, the Codex IDE extension and the
# ChatGPT DESKTOP app all read this one file. It is the only local-config
# route to a ChatGPT user. A `config.toml` is also hand-written, so these
# tests are mostly about what must survive being edited.

CODEX_CONFIG = '''# my codex config -- hand written
model = "gpt-5"

[mcp_servers.context7]
command = "npx"
args = ["-y", "@upstash/context7-mcp"]  # docs lookup

[mcp_servers.context7.env]
MY_ENV_VAR = "MY_ENV_VALUE"

[sandbox_workspace_write]
network_access = true
'''


def _codex(tmp_path, body=CODEX_CONFIG):
    path = tmp_path / "config.toml"
    if body is not None:
        path.write_text(body)
    return mi.Client("codex", "Codex", path, key="mcp_servers", fmt="toml")


def test_codex_is_one_file_for_three_clients(monkeypatch):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    c = mi.find("codex")
    assert c.path.name == "config.toml"
    assert ".codex" in str(c.path)
    assert c.key == "mcp_servers", "snake_case: mcpServers here parses and does nothing"
    assert c.fmt == "toml"


def test_codex_home_moves_the_config(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "elsewhere"))
    assert mi.find("codex").path == tmp_path / "elsewhere" / "config.toml"


def test_a_hand_written_toml_keeps_its_comments_its_order_and_its_servers(tmp_path):
    """The reason this is a text edit and not a round-trip.

    A TOML serialiser would return a valid file with the comments gone, the
    keys in some other order, and no way for the user to tell us that was
    not what they wanted.
    """
    tomllib = pytest.importorskip("tomllib")
    c = _codex(tmp_path)
    mi.apply(c)
    after = c.path.read_text()

    assert "# my codex config -- hand written" in after
    assert "# docs lookup" in after, "an inline comment on another server"
    assert after.index("[mcp_servers.context7]") < after.index("[sandbox_workspace_write]")
    doc = tomllib.loads(after)
    assert sorted(doc["mcp_servers"]) == ["context7", "semlayer"]
    assert doc["mcp_servers"]["context7"]["env"] == {"MY_ENV_VAR": "MY_ENV_VALUE"}
    assert doc["model"] == "gpt-5"
    assert doc["sandbox_workspace_write"]["network_access"] is True


def test_the_codex_entry_is_replaced_in_place_not_appended_again(tmp_path):
    """Twice must not leave two tables: a duplicate key stops TOML parsing,
    which would cost the user every server in the file and not just ours."""
    tomllib = pytest.importorskip("tomllib")
    c = _codex(tmp_path, CODEX_CONFIG + '''
[mcp_servers.semlayer]
command = "semlayer"
args = ["mcp"]

[tui]
notifications = true
''')
    mi.apply(c)
    after = c.path.read_text()

    assert after.count("[mcp_servers.semlayer]") == 1
    assert "command = \"semlayer\"\n" not in after, "the stale bare command is gone"
    assert "[tui]" in after, "what followed our table is still there"
    assert tomllib.loads(after)["tui"]["notifications"] is True


def test_a_sub_table_the_user_added_under_our_entry_survives(tmp_path):
    """`[mcp_servers.semlayer.env]` is their setting, in a table of its own.
    We rewrite ours and leave theirs attached."""
    tomllib = pytest.importorskip("tomllib")
    c = _codex(tmp_path, '''[mcp_servers.semlayer]
command = "old"
args = ["mcp"]

[mcp_servers.semlayer.env]
ANTHROPIC_API_KEY = "sk-theirs"
''')
    mi.apply(c)
    doc = tomllib.loads(c.path.read_text())
    assert doc["mcp_servers"]["semlayer"]["env"] == {"ANTHROPIC_API_KEY": "sk-theirs"}
    assert doc["mcp_servers"]["semlayer"]["command"] != "old"


def test_a_windows_command_is_escaped_not_pasted(tmp_path, monkeypatch):
    r"""`command = "C:\Users\..."` is a TOML parse error, because `\U` is not
    an escape. One unescaped backslash takes out the whole file."""
    tomllib = pytest.importorskip("tomllib")
    bin_dir = tmp_path / "Users\\Raj"
    bin_dir.mkdir()
    fake = bin_dir / "semlayer"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))

    c = _codex(tmp_path, "")
    mi.apply(c)
    text = c.path.read_text()
    assert "Users\\\\Raj" in text, "the backslash is doubled in the file"
    assert tomllib.loads(text)["mcp_servers"]["semlayer"]["command"] == str(fake)


def test_an_entry_we_cannot_edit_is_refused_not_duplicated(tmp_path):
    """An inline `mcp_servers = {...}` is legal TOML that the table scan
    cannot see. Appending our table would be a duplicate key."""
    pytest.importorskip("tomllib")
    c = _codex(tmp_path, 'mcp_servers = { semlayer = { command = "x", args = ["mcp"] } }\n')
    with pytest.raises(SemlayerError, match="cannot edit"):
        mi.apply(c)
    assert "command = \"x\"" in c.path.read_text(), "the file must be left alone"


def test_invalid_toml_is_refused_not_overwritten(tmp_path):
    """Same stance as a broken JSON config: it is theirs, and it probably
    holds other servers."""
    pytest.importorskip("tomllib")
    c = _codex(tmp_path, "[mcp_servers.context7\ncommand = ")
    with pytest.raises(SemlayerError, match="not valid TOML"):
        mi.apply(c)
    assert c.path.read_text() == "[mcp_servers.context7\ncommand = "


def test_a_missing_codex_config_is_created_with_its_directory(tmp_path):
    tomllib = pytest.importorskip("tomllib")
    c = mi.Client("codex", "Codex", tmp_path / "deep" / "config.toml",
                  key="mcp_servers", fmt="toml")
    assert not c.exists
    mi.apply(c)
    assert tomllib.loads(c.path.read_text())["mcp_servers"]["semlayer"]["args"] == ["mcp"]


def test_the_codex_plan_is_a_dry_run_too(tmp_path):
    c = _codex(tmp_path)
    before = c.path.read_text()
    result = mi.plan(c)
    assert c.path.read_text() == before, "plan must not touch the file"
    assert result["other_servers"] == ["context7"]
    assert result["after_text"].startswith("[mcp_servers.semlayer]")


def test_rewriting_the_same_codex_entry_is_reported_as_unchanged(tmp_path):
    c = _codex(tmp_path)
    mi.apply(c)
    assert mi.plan(c)["unchanged"] is True


# --- what the caller is allowed to print ------------------------------------

def test_the_entry_is_shown_in_the_format_it_is_written_in(tmp_path):
    """`after_text` is what the user is told. It has to match what they can
    go and read in the file, or the next person debugs the wrong shape."""
    assert mi.plan(_client(tmp_path))["after_text"].startswith('"semlayer": {')
    assert "command = " in mi.plan(_codex(tmp_path))["after_text"]


def test_the_rendered_file_is_available_but_is_not_the_report(tmp_path):
    """`rendered` is the whole config, including everything else the user has
    set up. `semlayer client ... --json` strips it for that reason."""
    r = mi.plan(_codex(tmp_path))
    assert "network_access" in r["rendered"], "it really is the whole file"
    assert "network_access" not in json.dumps(
        {k: v for k, v in r.items() if k not in ("merged", "rendered")})


def test_configured_reads_both_formats_and_never_raises(tmp_path):
    """`client list` has to describe a machine with a broken config on it."""
    pytest.importorskip("tomllib")
    jsonc = _client(tmp_path)
    assert mi.configured(jsonc) is False, "no file at all"
    mi.apply(jsonc)
    assert mi.configured(jsonc) is True

    codex = _codex(tmp_path)
    assert mi.configured(codex) is False
    mi.apply(codex)
    assert mi.configured(codex) is True

    codex.path.write_text("[mcp_servers.semlayer\n")
    assert mi.configured(codex) is False, "broken file: False, not a traceback"


def test_the_codex_warning_says_what_the_web_app_cannot_do():
    warn = mi.warning_for(mi.find("codex"))
    assert warn and "desktop" in warn
    assert "web" in warn, "ChatGPT on the web reads no local config at all"
