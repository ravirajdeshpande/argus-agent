# ARGUS — Autonomous Risk & Global Unified Surveillance Agent

ARGUS is an agentic AI system that watches news and other sources for risks, checks which claims are real, relates them to a company's internal exposure, and produces a qualified recommendation. It is built as a multi-agent graph (LangGraph) on top of a RAG pipeline (FAISS), tool servers (MCP), a 3-tier model cascade, and guardrails.

This is the **master README**: a high-level map of the repository. Each folder under `src/argus/` has its own detailed README.

Built as a training assignment from the coach's brief `ARGUS_Agentic_AI_Use_Case.docx` (structure §27, model cascade §13, metrics §24, Definition of Done §28).

---

## 1A. Quick start (Windows / PowerShell)

```powershell
git clone https://github.com/ravirajdeshpande/argus-agent.git
cd argus-agent
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .          # makes `import argus` work from anywhere
copy .env.example .env    # then fill in the values below
pytest -q                 # run all unit tests
```

Requires Python 3.13. Key `.env` settings:

```
OPENAI_API_KEY=sk-...
OPENAI_MID_MODEL=gpt-4.1-mini
```
## 1B. Quick start (Mac / Linux)

```powershell
git clone https://github.com/ravirajdeshpande/argus-agent.git
cd argus-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .          # makes `import argus` work from anywhere
cp .env.example .env    # then fill in the values below
pytest -q                 # run all unit tests
```

Requires Python 3.13. Key `.env` settings:

```
OPENAI_API_KEY=sk-...
OPENAI_MID_MODEL=gpt-4.1-mini
```
## 2. Run ARGUS

Build the demo index, then ask a question:

```powershell
python -m argus.rag.build_index --demo
python -m argus.app "Is our supplier's inventory at risk?"
```

| Flag | Effect |
|---|---|
| `--thread-id ID` | Name the checkpointed run |
| `--pause` | Stop at the human gates (`human_approval`, `analyst_review`) |
| `--resume` | Continue a paused run (use with `--thread-id`) |
| `--no-demo` | Use your own indexed data instead of the demo data |

## 3. Repository layout

```
argus-agent/
├── pyproject.toml          # package definition (src layout) + pytest settings
├── requirements.txt        # runtime dependencies
├── requirements-dev.txt    # test / dev tools (pytest, ...)
├── .env.example            # template of settings; copy to .env (never commit .env)
├── .gitignore
├── README.md               # this file
├── data/                   # raw and processed data; generated files are git-ignored
│   ├── raw/
│   ├── processed/          # FAISS index, checkpoint DB (generated)
│   └── evaluation/         # baseline.json for regression checks
├── docs/decisions/         # architecture decision notes
├── scripts/                # helper scripts (e.g. corpus ingestion)
├── tests/                  # unit/ and integration/ tests, mirror src/argus
└── src/argus/              # the application package
    ├── app.py              # entry point: runs ARGUS end to end
    ├── config/             # settings and model-tier configuration
    ├── graph/              # LangGraph workflow: state, nodes wiring, routing, checkpoints
    ├── agents/             # the specialist agents
    ├── rag/                # ingestion, FAISS store, retrieval, reranking
    ├── mcp/                # clients for external tool servers (news, government, market, company)
    ├── models/             # model cascade logic and shared data schemas
    ├── guardrails/         # evidence, prompt-injection, access-control and budget protections
    └── evaluation/         # metrics, regression checks, evaluation datasets
```

## 4. How the pieces connect

```
app.py  →  graph  →  agents  →  rag · mcp · models · guardrails  →  config
```

A layer may call the layers to its right, never the reverse. This keeps the code easy to test and change: for example `config` knows nothing about agents.

## 5. How a question flows through the graph

```
question → input guard → supervisor → investigators (news / government / market / RAG)
        → verifier (claims, conflicts, confidence) → loop for more evidence if confidence is low
        → risk analyst → human approval gate → final report
```

- **Orchestration:** LangGraph `StateGraph` with checkpointing (in-memory or SQLite) and `interrupt_before` human gates, so a run can be paused and resumed.
- **Verifier scoring (deterministic):** confidence = 0.4 × authority + 0.4 × corroboration + 0.2 × recency, multiplied by 0.7 if the claim is contradicted.
  - Authority by source: government 0.95, internal 0.85, market 0.7, news 0.6.
  - Corroboration = 1 − 0.5^(distinct origins).
  - Recency uses a 90-day half-life; unknown dates score 0.5.
- **Conflict handling:** for exclusive claim types (`effective_date`, `scope`), claims are grouped by fuzzy subject. A primary source wins; otherwise the conflict stays unresolved and is reported.
- **RAG:** FAISS index with local sentence-transformers embeddings (all-MiniLM-L6-v2). Access control fails closed to `restricted`.
- **Tools:** MCP-style `ToolClient`s for news, government and market data, running on in-memory demo transports.

## 6. What each folder is for

| Folder | Why it exists | Main requirement it serves | Status |
|---|---|---|---|
| `config/` | One place for settings and for "which model serves which tier". No other file reads `.env` directly. | Configurable model cascade (Sec. 13), 12-factor style settings | Built and tested (settings, model specs, prices) |
| `graph/` | Defines the workflow: shared state, the graph of steps, routing decisions, checkpoints so runs can be resumed. | Orchestration, loops, retries, auditability | Built and tested (graph, checkpoints, resume) |
| `agents/` | The specialists: supervisor, news investigator, internal analyst, verifier, risk analyst, recommender, evaluator. | Multi-agent reasoning pipeline | Built and tested |
| `rag/` | Turns internal documents into a searchable FAISS index and retrieves the best passages. | Internal exposure analysis over company documents | Built and tested (index builder, retrieval, access control) |
| `mcp/` | Thin clients that call external tool servers for news, government, market and company data. | Live external evidence gathering | Clients built; demo transports only, no real servers yet |
| `models/` | `cascade.py`: sends work to a cheap model first and escalates to stronger ones when needed. `schemas.py`: typed data objects (claims, evidence, results). | Model Cascading Strategy (Sec. 13), structured outputs | Built and tested |
| `guardrails/` | Safety rules: evidence must support claims, block prompt injection, enforce access control, cap spending (`ControlBudget`). | Safety, cost control, graceful degradation | Built and tested |
| `evaluation/` | Measures quality and catches regressions against fixed datasets. | Definition of Done (Sec. 28) | Built: metrics, 8 cases, regression runner |

## 7. The model cascade in one picture

```
~500 articles → Tier 1 (small/fast): relevance, language/topic, entities, duplicates
   → ~35 items → Tier 2 (mid): summaries, claims, preliminary risk, contradictions
   → ~8 critical items → Tier 3 (strong): cross-source reasoning, scenarios, recommendation
```

Models in use: Tier 1 `gpt-5.4-nano`, Tier 2 `gpt-4.1-mini`, Tier 3 `gpt-5.1`.

Low-confidence results escalate to the next tier. Every attempt records tier, model, latency, tokens, cost and escalation reason. A shared `ControlBudget` caps calls, tokens and cost; when it runs out ARGUS degrades gracefully and exits with a qualified report.

## 8. Evaluation

Eight cases: a positive RAG answer, direct injection, role-override injection, a confidential memo, an expired contract, a date conflict, an unsupported entity, and the approval gate.

```powershell
python -m argus.evaluation.regression                  # run and compare to baseline
python -m argus.evaluation.regression --save-baseline  # store a new baseline
```

The baseline lives in `data/evaluation/baseline.json`. A full run takes about 8 minutes because it calls the real models. LLM nondeterminism can move individual results, so comparisons use a tolerance.

## 9. Working method

1. **Test first:** write unit tests from the requirement, watch them fail, write the code, watch them pass.
2. **Small steps:** one module at a time, committed on the `refactor/architecture` branch.
3. **Secrets:** keys live only in `.env` (ignored by git). `.env.example` shows the variable names.

## 10. Known limitations

- The confidence threshold (0.75) is often unreachable with a single internal document, so the evidence loop usually runs to its limit.
- Some supplier statements are not always fetched by the news step.
- Thresholds are constants, not yet in `Settings`.
- MCP clients use in-memory demo transports, not real servers.
- The in-memory checkpointer logs a harmless LangGraph deserialisation warning for `Claim`; the SQLite checkpointer (the app default) does not.
