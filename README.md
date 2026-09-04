# cecilystuff

Data processing and simulation code. Each tool lives in its own subdirectory under
[processing/](processing/) and comes with a README that explains what it does and how it was
validated.

## Setup

Everything runs through [uv](https://docs.astral.sh/uv/). Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the repo root:

```bash
uv sync                     # creates .venv and installs everything
uv run pre-commit install   # runs the checks automatically on every commit
```

You never need to activate the virtualenv. Prefix commands with `uv run`:

```bash
uv run python processing/my_tool/run.py
uv run pytest
```

## Adding a dependency

```bash
uv add numpy          # a package the code needs to run
uv add --dev pytest   # a package only used for checks and tests
```

This updates `pyproject.toml` and `uv.lock`. Commit both, otherwise CI installs something
different from what you ran locally.

## Checks

The same three checks run in pre-commit and in CI, so the repo cannot drift:

| Command | What it catches |
| --- | --- |
| `uv run ruff check .` | unused imports, undefined names, likely bugs, import order |
| `uv run ruff format .` | formatting (rewrites files; CI uses `--check` instead) |
| `uv run pyright` | type errors, typos in attribute names, wrong argument counts |
| `uv run pytest` | your validation tests |

To run them all at once the way CI does:

```bash
uv run pre-commit run --all-files && uv run pytest
```

If pre-commit blocks a commit, it has usually already fixed the problem. Look at the diff,
`git add` it, and commit again.

## Layout

```
processing/<tool_name>/
    README.md      # specs first, then how the code was validated
    <code>.py
    test_<tool_name>_*.py
```

Test files need unique names across the repo (hence the `test_<tool_name>_` prefix), because
pytest imports them by filename.
