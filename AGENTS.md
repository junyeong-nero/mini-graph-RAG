# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `tiny_graph_rag/`, organized by pipeline stage:
- `chunking/`, `extraction/`, `graph/`, `retrieval/`, `llm/`, `evaluation/`, `visualization/`
- CLI entrypoint: `main.py`
- Streamlit UI: `streamlit_app.py`
- Utility scripts: `scripts/` (for example `scripts/apply_er.py`)

Tests are under `tests/`. Documentation is in `docs/` (`docs/README.md`, `docs/evaluation.md`). Sample corpora and graph/eval artifacts are in `data/novels/`.

## Build, Test, and Development Commands
- `uv sync`: install dependencies from `pyproject.toml`/`uv.lock`.
- `export OPENAI_API_KEY="..."`: required before CLI/UI features using the LLM.
- `uv run python main.py process <doc.txt> -o <name-KG.json>`: build a graph from text.
- `uv run python main.py query "<question>" -g <name-KG.json>`: run graph-based QA.
- `uv run python main.py eval --dataset <name-eval.jsonl> -g <name-KG.json> -o <results.json>`: run retrieval evaluation.
- `uv run streamlit run streamlit_app.py`: launch local UI.
- `uv run pytest`: run the full test suite.

## Coding Style & Naming Conventions
Target Python is 3.13+. Follow PEP 8 with 4-space indentation and clear type hints (matching current dataclass-heavy style).
- Modules/functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

Data and output naming in `data/novels/` should stay consistent:
`<작품명>.txt`, `<작품명>-KG.json`, `<작품명>-eval.jsonl`, `<작품명>-hardset.jsonl`, and `*-results.json`.

## Testing Guidelines
Use `pytest` with files named `test_*.py` and focused test methods describing behavior. Prefer deterministic tests with mocks for LLM calls (see `tests/test_integration.py` and `tests/test_extraction.py`).

For logic changes, add or update:
- unit tests for pure functions
- integration-style tests for graph flow
- evaluation tests when metric behavior changes

## Commit & Pull Request Guidelines
Recent history uses Conventional Commits, often with Korean summaries:
- `feat: ...`, `fix: ...`, `test: ...`, `docs: ...`, `chore: ...`

Keep commits logically scoped. In PRs, include:
- what changed and why
- commands used to validate (`uv run pytest`, relevant eval command)
- metric deltas or sample outputs for retrieval/eval changes
- screenshots for `streamlit_app.py` UI changes

## Security & Configuration Tips
Do not commit secrets. Keep API keys in environment variables (`OPENAI_API_KEY`; optional `OPENAI_BASE_URL`, `OPENAI_MODEL`) and treat `config.yaml` as non-secret defaults.
