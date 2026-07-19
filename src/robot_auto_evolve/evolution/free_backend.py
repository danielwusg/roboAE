"""Freer coding-agent backend (Revision 8).

This reverts the coding-agent mechanism to the earlier ``multimodel`` form: the
revision agent runs the plain ``claude`` CLI as a subprocess with a shell and the
ability to read the raw episode traces, instead of the OS-sandboxed Read/Edit-only
agent behind a network relay (``backends.ClaudeRevisionBackend``).

Drop-in for the driver's ``revise(prompt, candidate_dir, log_dir, index)`` call.
The agent edits ``scaffold.py`` in ``candidate_dir`` in place; anything else it
writes into ``candidate_dir`` is removed afterwards so the driver's
``validate_revision`` (which requires only ``scaffold.py`` to change) still holds.

Fairness fence (kept from the multimodel mechanism):
  * a static grep-guard rejects any revised ``.py`` that references privileged
    simulator state (object/goal poses, ``_check_success``, ``sim.data`` ...);
  * ``CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`` keeps the dev project's auto-memory out.

Filesystem access is intentionally NOT fenced (operator decision, 2026-07-19):
the agent may read any file it needs -- the scaffold, the frozen policy/sim source
under ``external/``, the on-disk public evidence. Fairness does not rely on hiding
files: it is enforced at ROLLOUT time (the scaffold only ever receives a
privilege-stripped observation) plus the grep-guard on the committed scaffold below.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

# Privileged-state accessors that a fair scaffold may never read. Ported verbatim
# from multimodel/scripts/openvla_libero_mm.sh:265 (GUARD_RE).
_GUARD_RE = re.compile(
    r"body_xpos|body_xquat|body_xmat|site_xpos|site_xmat|geom_xpos|geom_xmat|xipos"
    r"|get_body_xpos|get_site_xpos|get_geom_xpos|get_xpos|_check_success|obj_of_interest"
    r"|object_states|objects_dict|object-state|_to_robot0|env_state|\.sim\.data"
    r"|\.data\.(qpos|qvel|xpos|xquat|xmat|body|geom|site)|named\.data|\.physics\b"
    r"|_get_observations|get_observable|get_sim_state|regenerate_obs_from_state"
    r"|\"[a-z_0-9]*_[0-9]_(pos|quat)"
)

# Strip full-line comments before scanning, so prose that merely names a banned
# token does not trip the guard (matches the multimodel `sed 's/#.*$//'` step).
_COMMENT_RE = re.compile(r"#.*$", re.MULTILINE)

_EDITABLE = "scaffold.py"


class FairnessViolation(RuntimeError):
    """Raised when a revised file reads privileged simulator state."""


def _grep_guard(scaffold_dir: Path) -> None:
    for path in sorted(scaffold_dir.rglob("*.py")):
        text = _COMMENT_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
        hit = _GUARD_RE.search(text)
        if hit is not None:
            raise FairnessViolation(
                f"revision {path.name} reads privileged simulator state: {hit.group(0)!r}"
            )


class ClaudeFreeRevisionBackend:
    """Runs ``claude -p`` unsandboxed with a shell, editing ``scaffold.py`` in place."""

    def __init__(
        self,
        executable: Path,
        coding_model: str,
        *,
        timeout_s: float = 3600.0,
        max_turns: int = 200,
        effort: str | None = None,
        evidence_root: Path | None = None,
    ) -> None:
        self.executable = Path(executable)
        self.coding_model = str(coding_model)
        self.timeout_s = float(timeout_s)
        self.max_turns = int(max_turns)
        self.effort = effort
        # A read-only path (e.g. the incumbent evidence / raw traces) the prompt can
        # point the agent at; recorded for provenance only, not enforced here.
        self.evidence_root = None if evidence_root is None else Path(evidence_root)
        if (
            not self.executable.is_file()
            or self.timeout_s <= 0
            or self.max_turns < 1
            or re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", self.coding_model) is None
        ):
            raise ValueError("invalid free-agent backend configuration")

    def _command(self, prompt: str) -> list[str]:
        # Prompt goes on STDIN (not argv): at long horizons the prompt can exceed
        # the 128 KiB single-arg cap and exec() fails "Argument list too long".
        command = [
            str(self.executable),
            "-p",
            "--model",
            self.coding_model,
            "--max-turns",
            str(self.max_turns),
            "--permission-mode",
            "acceptEdits",
            # Tool grant matches the prior roboAutoEvol/multimodel mechanism
            # (meta_loop_cc.py:490 = Read,Write,Bash,Grep,Glob) + Edit, since our flow
            # edits the copied scaffold.py in place (the roboAE equivalent of the prior's
            # write-complete-file-then-rsync). Bash provides the prior's "do anything"
            # (python, curl, arbitrary shell — this node has internet). No --tools cap, no
            # WebFetch/WebSearch (Bash covers web, exactly as the prior did), no Task
            # (prior didn't grant it; it multiplies API cost without adding capability).
            # This does NOT compromise fairness: fairness is enforced at ROLLOUT time (the
            # scaffold gets a privilege-stripped observation, so it cannot act on ground
            # truth regardless of what the agent read) + the grep-guard on the committed
            # scaffold below. For airtight network egress, wrap in `unshare --net` + an
            # Anthropic-only relay (not done — matches the prior's network-open behavior).
            "--allowedTools",
            "Read,Edit,Write,Bash,Grep,Glob",
            "--output-format",
            "stream-json",
            "--verbose",
            # Prior-isolation for a clean experiment run (paired with
            # CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 in revise()): no inherited user/project
            # settings or CLAUDE.md, no MCP servers, no slash commands, no cross-session
            # persistence. This is the multimodel isolation intent, kept while the agent
            # is otherwise free (shell + raw traces).
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--disable-slash-commands",
            "--no-session-persistence",
        ]
        if self.effort:
            command += ["--effort", self.effort]
        return command

    def revise(self, prompt: str, candidate_dir: Path, log_dir: Path, index: int) -> None:
        candidate_dir = Path(candidate_dir).resolve()
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        scaffold = candidate_dir / _EDITABLE
        if not scaffold.is_file():
            raise ValueError(f"free-agent revision requires {_EDITABLE} in the candidate dir")
        before = scaffold.read_text(encoding="utf-8")

        env = dict(os.environ)
        env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"

        transcript = log_dir / "claude_transcript.jsonl"
        started = time.time()
        with open(transcript, "w") as handle:
            try:
                completed = subprocess.run(
                    self._command(prompt),
                    cwd=str(candidate_dir),
                    input=prompt,
                    stdout=handle,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout_s,
                    env=env,
                )
                returncode = completed.returncode
                stderr = completed.stderr or ""
            except subprocess.TimeoutExpired as exc:
                (log_dir / "claude_stderr.txt").write_text(
                    f"TIMEOUT after {time.time() - started:.0f}s\n{exc}", encoding="utf-8"
                )
                raise RuntimeError(
                    f"free-agent revision timed out after {self.timeout_s:.0f}s"
                ) from exc
        if stderr:
            (log_dir / "claude_stderr.txt").write_text(stderr, encoding="utf-8")

        # Remove anything the shell left behind except scaffold.py, so the driver's
        # validate_revision (only scaffold.py may change) still passes.
        for path in candidate_dir.iterdir():
            if path.name == _EDITABLE:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

        transcript_bytes = transcript.stat().st_size if transcript.is_file() else 0
        after = scaffold.read_text(encoding="utf-8")
        if returncode != 0 or transcript_bytes == 0:
            head = stderr.strip().splitlines()
            raise RuntimeError(
                f"free-agent claude failed: rc={returncode}, transcript={transcript_bytes}B, "
                f"stderr={(head[0] if head else '(none)')[:300]}"
            )
        if after == before:
            raise RuntimeError("free-agent revision did not change scaffold.py")

        # Fairness: reject a revision that reads privileged simulator state.
        _grep_guard(candidate_dir)

        # scaffold.py must still compile (mirrors backends validate_revision's compile).
        compile(after, str(scaffold), "exec")
