# processing

One subdirectory per tool. Each one is self-contained and starts with a `README.md`:

1. A `## Specs` section restating what the tool is supposed to do, written before the code.
2. After the code exists, a section on how it was validated: what was checked, against what
   reference, what the known limits are, and links to any papers or articles used.

Tests live next to the code as `test_<tool_name>_*.py` and run with `uv run pytest` from the
repo root.
