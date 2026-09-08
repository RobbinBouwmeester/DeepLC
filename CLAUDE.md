# Instructions for AI assistants

## Documentation style

Documentation must be self-contained, not conversation-dependent. Comments, docstrings, and
PR descriptions must be understandable to a reader with no access to the conversation or
reasoning that produced the code.

- Comments and docstrings describe the current state and behavior of the code: what it does
  and how it is used. They must not read as a log of the development process.
- Historical context (why a previous approach was replaced, what motivated a design choice)
  belongs in commit messages and PR descriptions, not in comments or docstrings, unless it is
  strictly necessary to understand the current implementation, in which case keep it brief.
- Benchmark results belong in the PR description and the changelog, not in docstrings.
- Use inline comments only where the code is not self-explanatory.
- PR descriptions are the place for process: reasoning behind the change, alternatives
  considered, and measurements.

## Project conventions

- Follow [Keep a Changelog](https://keepachangelog.com/) in `CHANGELOG.md` and semantic
  versioning in `pyproject.toml`.
- Run the test suite with `pytest tests` before pushing.
