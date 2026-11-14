# Agentic AI Pipeline for Systematic Reviews

An AI system that runs a **systematic review** (PRISMA 2020) from start to finish, on any topic:
it writes the protocol, searches the literature, screens articles, extracts data, assesses risk of bias,
and drafts the manuscript — with a human checking the key decisions.

Everything runs **locally**: the language model is **Qwen3.5 9B** through **Ollama** (no paid API).

---

## How it works

```
Topic ─► Protocol ─► Search ─► Screening ─► Full texts ─► Eligibility ─► Extraction ─► Report ─► Manuscript ─► Quality check
          (human        (4          (2 AI         (open         (2 AI          + risk        (PRISMA     (Word +
          approval)     databases)  reviewers     access)       reviewers)     of bias       diagram,    Markdown)
                                    + arbiter)                                                tables)
```

- **Search:** PubMed, Semantic Scholar, arXiv, OpenAlex, with automatic duplicate removal.
- **Screening and eligibility:** two simulated reviewers + an arbiter; uncertain cases go to a human.
- **Extraction:** every value must come with a quote that is checked against the article text.
- **Synthesis:** by direction of effect (SWiM); comparisons are verified and written by code.
- **Manuscript:** abstract, introduction (with references found and checked automatically), methods,
  results, PRISMA flow diagram (PNG), tables, discussion and references.
- **Quality check:** a critic agent verifies the counts, the protocol and what still needs human review.
- **Orchestration:** LangGraph, with pauses for human decisions.

---

## Installation (Windows)

1. Install **Ollama**: https://ollama.com/download
2. Download the models:
   ```bash
   ollama pull qwen3.5:9b
   ollama pull nomic-embed-text
   ```
3. Set up Python (3.11):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   copy .env.example .env
   ```
4. (Optional) Add your email / API keys in `.env` for faster access to PubMed and OpenAlex.

---

## Usage

**Run a full review on a new topic** (the program will ask you to approve the protocol):
```bash
python main.py --graph --topic "Effectiveness of AI for breast cancer detection in screening mammography"
```

**Continue an existing run** (only the missing steps are done):
```bash
python main.py --graph --resume outputs\run_20261109_125531
```

**Review the AI's uncertain decisions yourself:**
```bash
python main.py --resume outputs\run_20261109_125531 --review screening
python main.py --resume outputs\run_20261109_125531 --review eligibility
```

**Other useful commands:**
```bash
python main.py --resume <RUN> --manuscript-only   # rebuild the manuscript
python main.py --resume <RUN> --qa-only           # quality check
python main.py --resume <RUN> --eval              # evaluation report
python main.py --max-results 50 --topic "..."     # smaller test run
pytest tests/                                     # run the tests
```

---

## Results (in `outputs/run_<date>/`)

| File | What it is |
|---|---|
| `report/manuscript.docx` | Draft manuscript (Word) |
| `report/prisma_flow.png` | PRISMA 2020 flow diagram |
| `report/tables.md` | Study characteristics, results, risk of bias |
| `report/references.bib` | References of the included studies |
| `report/qa_report.md` | Quality check and what still needs human review |
| `evaluation/evaluation_report.md` | Accuracy, consistency and time savings |
| `trace.jsonl` | Every prompt and answer of the AI (full traceability) |

**Human corrections** are written in simple CSV files (`human_overrides.csv` in `screening/`, `fulltext/`
and `evidence/`). The AI's original decisions are never deleted, so every change can be audited.

---

## Project structure

```
main.py        command-line entry point
agents/        the agents (protocol, search, screening, extraction, synthesis, critic, ...)
graph/         LangGraph orchestration
rag/           hybrid search (embeddings + BM25) over articles
tools/         database clients, PDF parsing, quote checking, BibTeX and Word export
config/        protocol template (sources, risk-of-bias tool, extraction fields)
tests/         unit tests
```

---

## Limits

- The output is a **draft**: a human must review it before any use.
- Only **open-access** full texts can be read automatically.
- A small local model (9B) makes mistakes; this is why every result is checked by code and flagged for human review.

---

## Team contributions

_To be completed by each team member._
