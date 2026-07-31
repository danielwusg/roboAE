
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

_GUARD_RE = re.compile(
    r"body_xpos|body_xquat|body_xmat|site_xpos|site_xmat|geom_xpos|geom_xmat|xipos"
    r"|get_body_xpos|get_site_xpos|get_geom_xpos|get_xpos|_check_success|obj_of_interest"
    r"|object_states|objects_dict|object-state|_to_robot0|env_state|\.sim\.data"
    r"|\.data\.(qpos|qvel|xpos|xquat|xmat|body|geom|site)|named\.data|\.physics\b"
    r"|_get_observations|get_observable|get_sim_state|regenerate_obs_from_state"
    r"|\"[a-z_0-9]*_[0-9]_(pos|quat)"
)

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_MOVE_TARGET_RE = re.compile(
    r"\.move_to\s*\(\s*[A-Za-z_][\w.\[\]'\"]*\s*,\s*"
    r"(?:np\.(?:array|asarray|float32|float64)\s*\(\s*)?(?:[\(\[]\s*)?"
    rf"{_NUMBER}\s*,\s*{_NUMBER}\s*,\s*{_NUMBER}"
)

_COMMENT_RE = re.compile(r"#.*$", re.MULTILINE)

_EDITABLE = "scaffold.py"


class FairnessViolation(RuntimeError):
    pass


def _grep_guard(scaffold_dir: Path) -> None:
    for path in sorted(scaffold_dir.rglob("*.py")):
        text = _COMMENT_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
        hit = _GUARD_RE.search(text)
        if hit is not None:
            raise FairnessViolation(
                f"revision {path.name} reads privileged simulator state: {hit.group(0)!r}"
            )
        target = _MOVE_TARGET_RE.search(text)
        if target is not None:
            raise FairnessViolation(
                f"revision {path.name} moves the robot to a hardcoded position: {target.group(0)!r}. "
                "A movement destination must be computed from this episode's own observation."
            )


class ClaudeFreeRevisionBackend:

    def __init__(
        self,
        executable: Path,
        coding_model: str,
        *,
        timeout_s: float = 3600.0,
        max_turns: int = 200,
        effort: str | None = None,
        evidence_root: Path | None = None,
        fairness_guard: bool = False,
    ) -> None:
        self.executable = Path(executable)
        self.coding_model = str(coding_model)
        self.timeout_s = float(timeout_s)
        self.max_turns = int(max_turns)
        self.effort = effort
        self.fairness_guard = bool(fairness_guard)
        self.evidence_root = None if evidence_root is None else Path(evidence_root)
        if (
            not self.executable.is_file()
            or self.timeout_s <= 0
            or self.max_turns < 1
            or re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", self.coding_model) is None
        ):
            raise ValueError("invalid free-agent backend configuration")

    def _command(self, prompt: str) -> list[str]:
        command = [
            str(self.executable),
            "-p",
            "--model",
            self.coding_model,
            "--max-turns",
            str(self.max_turns),
            "--permission-mode",
            "acceptEdits",
            "--allowedTools",
            "Read,Edit,Write,Bash,Grep,Glob",
            "--output-format",
            "stream-json",
            "--verbose",
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

        if self.fairness_guard:
            _grep_guard(candidate_dir)

        compile(after, str(scaffold), "exec")
