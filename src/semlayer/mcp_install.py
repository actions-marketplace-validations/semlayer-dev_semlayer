# SPDX-License-Identifier: FSL-1.1-ALv2
"""Writing semlayer into an MCP client's config, so nobody hand-edits JSON.

Hand-editing is the step that loses people, and it fails in three ways that
have all been observed:

  * The snippet lands in the wrong place. A bare ``"semlayer": {...}`` does
    not say that it belongs inside ``mcpServers``, and at the top level of
    the file it is silently inert.
  * The command is a bare name. A client launched from a dock or a desktop
    icon does not inherit a shell's PATH, so ``semlayer`` resolves in your
    terminal and not in the client, which reports ``executable not found``.
  * An editor's own process rewrites the file. Claude Code keeps session
    state in the same JSON, so an edit made while it is running can be
    overwritten by its next write.

The VS Code extension solved this in TypeScript for the people who install
it. This is the same thing for everyone else -- a terminal user, a container,
and above all an agent that was asked to "install the semlayer MCP server"
and would otherwise be editing JSON by hand on someone's machine.

Two config FORMATS, not one. Five clients keep a JSON object, which is
re-serialised whole because nothing else is in it that a user wrote. Codex
keeps `config.toml`, which is hand-written and holds comments, so its entry
is edited in place: the `[mcp_servers.semlayer]` table is replaced or
appended and every other byte of the file is left exactly where it was.

What it will not do: invent a client. The six here are the ones whose config
shape is known and exercised. An unknown name is an error that lists what is
known, rather than a guess at a path.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from semlayer.errors import SemlayerError


@dataclass(frozen=True)
class Client:
    """An MCP client this machine might have, and where its config lives."""

    id: str
    label: str
    path: Path
    #: The key servers live under. VS Code calls it `servers`; everyone else
    #: `mcpServers`. Recorded per client rather than assumed, because writing
    #: the wrong key produces a config that parses and does nothing.
    key: str = "mcpServers"
    #: `json`, or `toml` for Codex. This decides how the entry is WRITTEN,
    #: which is a different question from what it is called: a JSON config
    #: is re-serialised, a TOML one is edited in place.
    fmt: str = "json"

    @property
    def exists(self) -> bool:
        """Whether this client has a config file here at all."""
        return self.path.exists()


def _home(*parts: str) -> Path:
    return Path.home().joinpath(*parts)


def _desktop_config() -> Path:
    """Claude Desktop keeps its config somewhere different on every platform."""
    system = platform.system()
    if system == "Darwin":
        return _home("Library", "Application Support", "Claude",
                     "claude_desktop_config.json")
    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else _home("AppData", "Roaming")
        return base / "Claude" / "claude_desktop_config.json"
    return _home(".config", "Claude", "claude_desktop_config.json")


def _vscode_config() -> Path:
    """VS Code's own agent, via Copilot's portable config.

    NOT `.vscode/mcp.json`. That is the editor-managed format, it is scoped
    to one workspace, and VS Code's own documentation marks it deprecated
    with removal in February 2027 -- writing it would be shipping a path
    with an expiry date on it. The portable file is user-level, is what the
    docs point new configurations at, and uses the same `mcpServers` shape
    as every other client here.

    `COPILOT_HOME` moves it, the same way `SEMLAYER_HOME` moves ours.
    """
    base = os.environ.get("COPILOT_HOME")
    return (Path(base) if base else _home(".copilot")) / "mcp-config.json"


def _codex_config() -> Path:
    """One file for Codex, the Codex IDE extension, and ChatGPT desktop.

    OpenAI's MCP documentation says all three read the same
    `$CODEX_HOME/config.toml`, so this single entry serves the ChatGPT
    desktop population as well -- which is the reason it is here. ChatGPT on
    the WEB reads no local file and cannot be configured this way at all.

    Not the project-scoped `.codex/config.toml`: it applies to one directory
    and only in a project Codex has been told to trust.
    """
    base = os.environ.get("CODEX_HOME")
    return (Path(base) if base else _home(".codex")) / "config.toml"


def clients() -> list[Client]:
    """The clients we know how to write, in the order worth offering them."""
    return [
        Client("claude-code", "Claude Code", _home(".claude.json")),
        Client("claude-desktop", "Claude Desktop", _desktop_config()),
        Client("cursor", "Cursor", _home(".cursor", "mcp.json")),
        Client("cline", "Cline",
               _home(".cline", "data", "settings", "cline_mcp_settings.json")),
        Client("vscode", "VS Code (Copilot)", _vscode_config()),
        Client("codex", "Codex / ChatGPT desktop", _codex_config(),
               # snake_case, and TOML. Writing `mcpServers` here produces a
               # config that parses and does nothing, exactly as the wrong
               # key does in a JSON client.
               key="mcp_servers", fmt="toml"),
    ]


def find(name: str) -> Client:
    """One client by id, or an error naming the ones that exist."""
    for c in clients():
        if c.id == name:
            return c
    known = ", ".join(c.id for c in clients())
    raise SemlayerError(f"{name!r} is not a client semlayer knows how to configure",
                        hint=f"one of: {known}")


def command_for(cli: str | None = None) -> list[str]:
    """How a DIFFERENT process should invoke this engine.

    An absolute path, because the client is another process with another
    environment: `~/.local/bin` is exactly the directory a desktop app
    launched from a dock cannot see, and a bare `semlayer` that works in your
    terminal fails there with `executable not found`.

    Falls back to `<this python> -m semlayer.cli` when no launcher is on
    PATH, which is the normal state inside a venv that was never activated
    for the client.
    """
    found = shutil.which(cli or "semlayer")
    if found:
        # Absolute, but NOT symlink-resolved. `uv tool install` leaves a shim
        # at `~/.local/bin/semlayer` pointing into its own tool directory;
        # the shim is the stable, documented path (`which semlayer` prints
        # it) and the target is uv's private layout, which it may rearrange.
        return [os.path.abspath(found)]
    return [sys.executable, "-m", "semlayer.cli"]


def entry(layer: str | Path | None = None, cli: str | None = None) -> dict:
    """The `mcpServers.semlayer` value: command and arguments kept separate.

    Separate because they are separate fields in every client's schema. A
    whole argv joined into `command` names an executable that cannot exist --
    the mistake is invisible for a plain binary and fatal for anything with
    arguments, such as a `uvx` invocation.
    """
    argv = command_for(cli)
    args = [*argv[1:], "mcp"]
    if layer is not None:
        args.append(str(Path(layer).expanduser().resolve()))
    return {"command": argv[0], "args": args}


# --- TOML, for Codex --------------------------------------------------------
#
# Edited as TEXT, not round-tripped. A `config.toml` is a file someone wrote
# by hand: it has comments, an order they chose, and settings this module
# knows nothing about. Re-serialising it would silently delete all three, so
# the only thing written here is our own table.

#: A table header on its own line: `[mcp_servers.semlayer]`. Deliberately
#: does not match `[[array.of.tables]]`, whose inner `]` the character class
#: excludes.
_TABLE = re.compile(r"^\s*\[\s*([^\[\]]+?)\s*\]\s*$")


def _toml_string(value: str) -> str:
    r"""A TOML basic string, encoded by `json.dumps`.

    Not laziness: TOML basic strings use JSON's escapes (`\"`, `\\`, `\n`,
    `\uXXXX`), and the one that matters is the backslash. A Windows command
    written raw gives `command = "C:\Users\..."`, where `\U` is an invalid
    escape -- the file stops parsing and the client loses every server in it,
    not just ours.
    """
    return json.dumps(value)


def _toml_block(client: Client, value: dict) -> str:
    """Our table, rendered.

    Command and args only: anything else under the entry is the user's, and
    lives in a sub-table this does not touch.
    """
    args = ", ".join(_toml_string(a) for a in value.get("args") or [])
    return (f"[{client.key}.semlayer]\n"
            f"command = {_toml_string(value['command'])}\n"
            f"args = [{args}]\n")


def _toml_tables(text: str) -> list[tuple[int, str]]:
    """(line index, dotted name) for every table header, in file order."""
    return [(i, m.group(1).strip())
            for i, line in enumerate(text.splitlines())
            if (m := _TABLE.match(line))]


def _parse_toml(path: Path, text: str) -> dict | None:
    """The file as data, or None where this Python has no TOML reader.

    `tomllib` is 3.11; the floor is 3.10. The parse is a CHECK, not the edit
    path -- it catches the forms the text scan cannot see (an inline
    `mcp_servers = {...}`, a quoted `["mcp_servers"."semlayer"]`) before we
    append a duplicate key and break the user's file. Without it the scan
    still works for the shape Codex itself writes.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - 3.10 only
        return None
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise SemlayerError(f"{path} is not valid TOML ({e})",
                            hint="fix or move the file; refusing to rewrite a "
                                 "config that may hold other servers") from e


def _plan_toml(client: Client, layer: str | Path | None = None,
               cli: str | None = None) -> dict:
    """Replace our table, or append it. Everything else is bytes we copy."""
    text = (client.path.read_text(encoding="utf-8")
            if client.path.exists() else "")
    doc = _parse_toml(client.path, text) if text.strip() else {}
    after = entry(layer, cli)
    block = _toml_block(client, after)
    header = f"{client.key}.semlayer"
    tables = _toml_tables(text)
    at = next((i for i, name in tables if name == header), None)

    servers = (doc or {}).get(client.key)
    declared = isinstance(servers, dict) and "semlayer" in servers
    if at is None and declared:
        # The parse can see it and the scan cannot: an inline table, or a
        # quoted key. Appending our table now would be a duplicate key and
        # would stop the whole file parsing.
        raise SemlayerError(
            f"{client.path} already configures semlayer in a form this cannot edit",
            hint=f"it is not a plain `[{header}]` table. Edit it by hand, or "
                 "`codex mcp remove semlayer` first")

    lines = text.splitlines(keepends=True)
    if at is None:
        before = None
        head = text if not text or text.endswith("\n") else text + "\n"
        rendered = head + ("\n" if head.strip() else "") + block
    else:
        # To the next header, so a sub-table the user added under our entry
        # (`[mcp_servers.semlayer.env]`) survives: it is a table in its own
        # right, and it still belongs to us after ours is rewritten.
        end = next((i for i, _ in tables if i > at), len(lines))
        before = "".join(lines[at:end]).strip() or None
        rendered = "".join(lines[:at]) + block + "".join(lines[end:])

    if isinstance(servers, dict):
        others = sorted(k for k in servers if k != "semlayer")
    else:
        others = sorted({name[len(client.key) + 1:].split(".")[0]
                         for _, name in tables
                         if name.startswith(f"{client.key}.")} - {"semlayer"})
    return {
        "client": client.id,
        "path": str(client.path),
        "existed": client.path.exists(),
        "before": before,
        "after": after,
        "unchanged": before == block.strip(),
        "other_servers": others,
        "merged": None,  # there is no merged document: this is a text edit
        "after_text": block,
        "rendered": rendered,
    }


def configured(client: Client) -> bool:
    """Whether this client already has a semlayer entry.

    Never raises. The caller is `semlayer client list`, which has to be able
    to describe a machine with one broken config on it rather than die on it.
    """
    try:
        if not client.path.exists():
            return False
        if client.fmt == "toml":
            text = client.path.read_text(encoding="utf-8", errors="replace")
            servers = (_parse_toml(client.path, text) or {}).get(client.key)
            if isinstance(servers, dict) and "semlayer" in servers:
                return True
            return any(n == f"{client.key}.semlayer"
                       or n.startswith(f"{client.key}.semlayer.")
                       for _, n in _toml_tables(text))
        return "semlayer" in (read_config(client.path).get(client.key) or {})
    except (SemlayerError, OSError):
        return False


def read_config(path: Path) -> dict:
    """Parse a client's config, or start an empty one.

    A file that is not JSON is an ERROR, not an empty dict: it is the user's
    file and probably holds other servers, and overwriting it would be the
    worst outcome this module can produce.
    """
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise SemlayerError(f"{path} is not valid JSON ({e})",
                            hint="fix or move the file; refusing to overwrite "
                                 "a config that may hold other servers") from e
    if not isinstance(value, dict):
        raise SemlayerError(f"{path} does not hold a JSON object")
    return value


def plan(client: Client, layer: str | Path | None = None,
         cli: str | None = None) -> dict:
    """What `apply` would do, without doing it.

    Returns the merged config, the exact text that would be written, and a
    verdict, so a caller can show the user the change before it happens and
    an agent can report it.
    """
    if client.fmt == "toml":
        return _plan_toml(client, layer, cli)
    config = read_config(client.path)
    servers = config.get(client.key)
    servers = dict(servers) if isinstance(servers, dict) else {}
    before = servers.get("semlayer")
    after = entry(layer, cli)
    merged = dict(config)
    merged[client.key] = {**servers, "semlayer": after}
    return {
        "client": client.id,
        "path": str(client.path),
        "existed": client.path.exists(),
        "before": before,
        "after": after,
        "unchanged": before == after,
        # Everything else in the file is carried through untouched. Named in
        # the plan so a reviewer can see what is NOT being rewritten.
        "other_servers": sorted(k for k in servers if k != "semlayer"),
        "merged": merged,
        # The entry as the file will state it. Shown to the user instead of
        # a repr, so that what they are told matches what they can go and
        # read -- a JSON fragment for a JSON client, a TOML table for Codex.
        "after_text": f'"semlayer": {json.dumps(after)}',
        # What `apply` writes, byte for byte. Rendered here so that a dry run
        # shows the file that would land, not an approximation of it.
        "rendered": json.dumps(merged, indent=2) + "\n",
    }


def apply(client: Client, layer: str | Path | None = None,
          cli: str | None = None) -> dict:
    """Write the entry, preserving every other key in the file.

    One write for both formats: `plan` has already decided what the file
    should say, so there is no second code path here to disagree with it.
    """
    result = plan(client, layer, cli)
    client.path.parent.mkdir(parents=True, exist_ok=True)
    client.path.write_text(result["rendered"], encoding="utf-8")
    return result


def warning_for(client: Client) -> str | None:
    """What is true about this client that a user has to know afterwards.

    Both of these have cost a person an evening, so they are returned rather
    than left to the reader of a doc.
    """
    if client.id == "codex":
        return ("Codex, the Codex IDE extension and the ChatGPT desktop app "
                "all read this one file, so this entry serves all three. "
                "ChatGPT on the web reads no local config and cannot be "
                "pointed at a local server at all. `codex mcp add semlayer "
                "-- <command> mcp` writes the same table through the client "
                "itself, if you would rather it did.")
    if client.id == "claude-code":
        return ("Claude Code keeps session state in this same file and can "
                "overwrite an edit made while it is running. Quit it first, "
                "or use `claude mcp add semlayer -- <command> mcp`, which "
                "writes it through the client itself.")
    return None


RESTART_NOTE = ("Start a NEW session of the client before asking it anything: "
                "an MCP client reads its server list once, at startup, so one "
                "that was already running has never seen this entry.")
