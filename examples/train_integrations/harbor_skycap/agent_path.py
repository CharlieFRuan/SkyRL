"""Keep directories that are not meant for the agent off its shell's PATH (opt-in, ``SKYCAP_AGENT_PATH_DROP``).

    SKYCAP_AGENT_PATH_DROP=/opt/verifier/bin      (several, colon-separated)

Some task images put the grader's own tools first on PATH. CalibForge's ``tb2-verifier-base`` image (41% of its
tasks) has ``/opt/verifier/bin`` there: the agent's ``python3`` is then the grader's bare interpreter (Python 3.13,
pytest only, no pip), not the system Python the task's Dockerfile installed numpy, pandas, Pillow, ... for. The model
spends turns finding out why "installed" packages are missing, can ``pip install`` into the grader's environment, and
can break it. The grader calls ``/opt/verifier/bin/python`` by its full path, so it does not need PATH.

With this, Terminus-2's tmux session starts with those directories removed from PATH. Nothing else changes.
"""

import os
import shlex

from harbor.agents.terminus_2.tmux_session import TmuxSession

DROP = [path for path in os.environ.get("SKYCAP_AGENT_PATH_DROP", "").split(":") if path]
#: ``SKYCAP_AGENT_PATH_FIRST``: directories moved to the front of the agent's PATH (after the drop), e.g. /usr/bin
#: where the task's packages were installed for the system interpreter but another one comes first.
FIRST = [path for path in os.environ.get("SKYCAP_AGENT_PATH_FIRST", "").split(":") if path]


def path_export(drop, first=()) -> str:
    """Shell that re-exports PATH without the directories in ``drop`` and with those in ``first`` in front."""
    filters = " ".join(f"| grep -vxF {shlex.quote(path)}" for path in [*drop, *first])
    front = "".join(f"{path}:" for path in first)
    return f"export PATH=\"{front}$(printf %s \"$PATH\" | tr ':' '\\n' {filters} | paste -sd: -)\""


def apply(drop=None, first=None) -> None:
    drop = DROP if drop is None else drop
    first = FIRST if first is None else first
    if not (drop or first) or getattr(TmuxSession, "_skycap_agent_path", False):
        return
    original = TmuxSession._tmux_start_session

    @property
    def _tmux_start_session(self) -> str:
        # The tmux server, and through it the pane's shell, inherits the environment of this command.
        return f"{path_export(drop, first)} && {original.fget(self)}"

    TmuxSession._tmux_start_session = _tmux_start_session
    TmuxSession._skycap_agent_path = True


apply()
