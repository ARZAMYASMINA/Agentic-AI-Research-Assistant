"""Point d'entrée du pipeline.

Étapes 2-4 : sujet -> protocole PRISMA -> stratégie de recherche -> récupération + déduplication
-> index RAG -> screening titre/résumé (double relecture + arbitre).

Exemples :
    python main.py
    python main.py --topic "Effect of intermittent fasting on type 2 diabetes"
    python main.py --topic "..." --yes                 # sans validation humaine
    python main.py --no-retrieve                       # s'arrête après la stratégie de recherche
    python main.py --resume outputs\\run_XXXX          # reprend un run : récupération + index
    python main.py --resume outputs\\run_XXXX --index-only --query "second reader mammography"
    python main.py --resume outputs\\run_XXXX --screen-only --screen-limit 10   # test du screening
    python main.py --resume outputs\\run_XXXX --screen-only                     # screening complet
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

from agents.eligibility import print_eligibility, run_eligibility
from agents.evidence import print_evidence, refresh_evidence_tables, run_extraction, run_reappraisal
from agents.fulltext import print_fulltext, run_fulltext_retrieval
from agents.llm import check_model, set_trace_file
from agents.planner import generate_protocol
from agents.report import print_report, write_report_data
from agents.retrieval import run_retrieval
from agents.schemas import Protocol, SearchStrategy
from agents.screening import print_screening, run_screening
from agents.search_strategy import build_search_strategy, make_queries

DEFAULT_TOPIC = "Effectiveness of artificial intelligence for breast cancer detection in screening mammography"
TEMPLATE = Path("config/protocol_template.yaml")


def load_config() -> dict:
    if TEMPLATE.exists():
        return yaml.safe_load(TEMPLATE.read_text(encoding="utf-8")) or {}
    return {}


def print_protocol(p: Protocol) -> None:
    print("\n=== PROTOCOLE ===")
    print(f"Titre    : {p.title}")
    print(f"Question : {p.research_question}")
    print(f"P : {p.pico.population}\nI : {p.pico.intervention}\nC : {p.pico.comparator}\nO : {p.pico.outcome}")
    print("Inclusion :")
    for c in p.inclusion_criteria:
        print(f"  + {c}")
    print("Exclusion :")
    for c in p.exclusion_criteria:
        print(f"  - {c}")
    print(f"Langues : {p.languages} | Période : {p.date_from or '...'} -> {p.date_to or '...'}")


def print_prisma(prisma: dict) -> None:
    ident = prisma["identification"]
    print("\n=== PRISMA : IDENTIFICATION ===")
    for src, s in ident["per_source"].items():
        print(f"  {src:<18} {s['retrieved']:>5} récupérés / {s['hits_total']} disponibles")
    print(f"  Total récupéré      : {ident['total_retrieved']}")
    print(f"  Doublons retirés    : {ident['duplicates_removed']} "
          f"(DOI {ident['duplicates_by_doi']}, PMID {ident['duplicates_by_pmid']}, titre {ident['duplicates_by_title']})")
    print(f"  Articles uniques    : {ident['records_unique']}")
    print(f"  Sans résumé         : {ident['without_abstract']}")
    for src, err in prisma["errors"].items():
        print(f"  ERREUR {src} : {err}")


def print_hits(query: str, hits) -> None:
    print(f"\n=== RECHERCHE HYBRIDE : « {query} » ===")
    for h in hits:
        p = h.paper
        print(f"  {p.id} | {p.year or '----'} | dense #{h.dense_rank or '-'} sparse #{h.sparse_rank or '-'} | {p.title[:90]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline de revue systématique agentique")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--yes", action="store_true", help="Ne pas demander la validation humaine")
    parser.add_argument("--no-retrieve", action="store_true", help="Ne pas lancer la récupération des articles")
    parser.add_argument("--resume", metavar="RUN_DIR", help="Reprendre un run existant (protocole + stratégie déjà générés)")
    parser.add_argument("--index-only", action="store_true", help="Avec --resume : ne pas relancer la récupération, reconstruire l'index")
    parser.add_argument("--query", help="Tester la recherche hybride sur l'index après sa construction")
    parser.add_argument("--screen", action="store_true", help="Lancer le screening titre/résumé après l'index")
    parser.add_argument("--screen-only", action="store_true", help="Avec --resume : seulement le screening (ni récupération ni index)")
    parser.add_argument("--screen-limit", type=int, help="Ne screener que les N premiers articles (test)")
    parser.add_argument("--fulltext", action="store_true", help="Récupérer les textes intégraux open access après le screening")
    parser.add_argument("--fulltext-only", action="store_true", help="Avec --resume : seulement la récupération des textes intégraux")
    parser.add_argument("--eligibility", action="store_true", help="Évaluer l'éligibilité sur texte intégral après la récupération")
    parser.add_argument("--eligibility-only", action="store_true", help="Avec --resume : seulement l'éligibilité sur texte intégral")
    parser.add_argument("--extract", action="store_true", help="Extraire les données et évaluer le risque de biais après l'éligibilité")
    parser.add_argument("--extract-only", action="store_true", help="Avec --resume : seulement l'extraction et le risque de biais")
    parser.add_argument("--reappraise-only", action="store_true", help="Avec --resume : refaire seulement le risque de biais et re-vérifier les champs (sans ré-extraire)")
    parser.add_argument("--extract-limit", type=int, help="Ne traiter que les N premiers articles (test)")
    parser.add_argument("--report", action="store_true", help="Générer diagramme PRISMA, références BibTeX et tableaux après l'extraction")
    parser.add_argument("--report-only", action="store_true", help="Avec --resume : seulement les livrables du rapport (sans LLM)")
    parser.add_argument("--manuscript", action="store_true", help="Rédiger le brouillon de manuscrit après le rapport")
    parser.add_argument("--manuscript-only", action="store_true", help="Avec --resume : seulement le brouillon de manuscrit (Markdown + DOCX)")
    parser.add_argument("--review", choices=["screening", "eligibility"], help="Avec --resume : relecture humaine interactive des décisions à revoir, puis application")
    parser.add_argument("--review-limit", type=int, help="Nombre maximal de décisions à relire dans cette session")
    parser.add_argument("--qa", action="store_true", help="Lancer l'agent critique (contrôle qualité) en fin de run")
    parser.add_argument("--qa-only", action="store_true", help="Avec --resume : seulement le contrôle qualité (sans LLM)")
    parser.add_argument("--graph", action="store_true", help="Exécuter tout le pipeline avec le graphe LangGraph (supervision humaine par interruptions ; avec --resume : reprend le run)")
    parser.add_argument("--eval-prepare", action="store_true", help="Avec --resume : créer les gabarits de vérité terrain (evaluation/)")
    parser.add_argument("--eval-repro", type=int, metavar="N", help="Avec --resume : refaire le screening de N articles pour mesurer la reproductibilité (LLM)")
    parser.add_argument("--eval", action="store_true", help="Avec --resume : calculer le rapport d'évaluation (sans LLM)")
    parser.add_argument("--max-results", type=int, help="Maximum d'articles par source (défaut : config YAML)")
    args = parser.parse_args()

    if args.graph:
        return run_with_graph(args)

    if (args.index_only or args.screen_only or args.fulltext_only or args.eligibility_only or args.extract_only or args.reappraise_only or args.report_only or args.manuscript_only or args.qa_only or bool(args.review) or args.eval_prepare or args.eval or bool(args.eval_repro)) and not args.resume:
        parser.error("--index-only, --screen-only, --fulltext-only, --eligibility-only, --extract-only, --reappraise-only, --report-only, --manuscript-only, --qa-only et --review nécessitent --resume RUN_DIR")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    max_results = args.max_results or cfg.get("search", {}).get("max_results_per_source", 200)
    do_screening = args.screen or args.screen_only
    do_fulltext = args.fulltext or args.fulltext_only
    do_eligibility = args.eligibility or args.eligibility_only
    do_extract = args.extract or args.extract_only
    do_report = args.report or args.report_only
    do_manuscript = args.manuscript or args.manuscript_only

    if args.resume:
        run_dir = Path(args.resume)
        protocol = Protocol.model_validate_json((run_dir / "protocol.json").read_text(encoding="utf-8"))
        strategy = SearchStrategy.model_validate_json((run_dir / "search_strategy.json").read_text(encoding="utf-8"))
        # Les requêtes sont reconstruites depuis les blocs (sans LLM) pour appliquer les dernières corrections
        blocks, queries = make_queries(strategy.blocks, protocol, cfg.get("search", {}).get("sources") or None)
        strategy = SearchStrategy(blocks=blocks, queries=queries)
        (run_dir / "search_strategy.json").write_text(strategy.model_dump_json(indent=2), encoding="utf-8")
        set_trace_file(run_dir / "trace.jsonl")
        print(f"Reprise du run : {run_dir}")
        if do_screening or do_eligibility or do_extract or args.reappraise_only or do_manuscript or args.eval_repro:
            check_model()
    else:
        check_model()
        run_dir = Path("outputs") / f"run_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir.mkdir(parents=True, exist_ok=True)
        set_trace_file(run_dir / "trace.jsonl")
        (run_dir / "topic.txt").write_text(args.topic, encoding="utf-8")
        print(f"Dossier de run : {run_dir}\nSujet : {args.topic}")

        # --- Planner ---
        print("\n[1/5] Génération du protocole (Qwen3.5 9B)...")
        protocol = generate_protocol(args.topic)
        (run_dir / "protocol.json").write_text(protocol.model_dump_json(indent=2), encoding="utf-8")
        print_protocol(protocol)

        # --- Human-in-the-loop ---
        if not args.yes:
            answer = input("\nValider ce protocole ? [o/N] ").strip().lower()
            if answer not in ("o", "oui", "y", "yes"):
                print("Protocole refusé. Modifiez protocol.json puis relancez avec --resume, ou relancez.")
                return 1

        # --- Search Strategy ---
        print("\n[2/5] Génération de la stratégie de recherche...")
        strategy = build_search_strategy(protocol, cfg.get("search", {}).get("sources") or None)
        (run_dir / "search_strategy.json").write_text(strategy.model_dump_json(indent=2), encoding="utf-8")

    if not (args.index_only or args.screen_only or args.fulltext_only or args.eligibility_only or args.extract_only or args.reappraise_only or args.report_only or args.manuscript_only or args.qa_only or bool(args.review) or args.eval_prepare or args.eval or bool(args.eval_repro)):
        print("\n=== REQUÊTES ===")
        for q in strategy.queries:
            print(f"\n[{q.source}]\n{q.query}")

        if args.no_retrieve:
            print(f"\nTerminé (sans récupération). Fichiers dans : {run_dir}")
            return 0

        # --- Retrieval ---
        if not args.yes and not args.resume:
            if input("\nLancer la récupération des articles ? [o/N] ").strip().lower() not in ("o", "oui", "y", "yes"):
                print(f"Arrêt. Reprenez plus tard avec : python main.py --resume {run_dir}")
                return 0
        print(f"\n[3/5] Récupération des articles (max {max_results} par source)...")
        prisma = run_retrieval(strategy, protocol, run_dir, max_results)
        print_prisma(prisma)

    # --- Index RAG ---
    if not (args.screen_only or args.fulltext_only or args.eligibility_only or args.extract_only or args.reappraise_only or args.report_only or args.manuscript_only or args.qa_only or bool(args.review) or args.eval_prepare or args.eval or bool(args.eval_repro)):
        from rag.index import build_index  # import tardif : chromadb n'est requis qu'ici

        print("\n[4/5] Construction de l'index RAG hybride (Chroma + BM25)...")
        index = build_index(run_dir)
        print(f"  Index prêt : {run_dir / 'chroma'}")
        if args.query:
            print_hits(args.query, index.search(args.query, k=8))

    # --- Screening ---
    if do_screening:
        print("\n[5/5] Screening titre/résumé (relecteurs A et B + arbitre)...")
        summary = run_screening(run_dir, protocol, limit=args.screen_limit)
        print_screening(summary)
        print(f"  Détail et justifications : {run_dir / 'screening' / 'results.csv'}")

    # --- Textes intégraux ---
    if do_fulltext:
        print("\n[texte intégral] Récupération des textes intégraux open access...")
        ft = run_fulltext_retrieval(run_dir)
        print_fulltext(ft)
        print(f"  Manifeste : {run_dir / 'fulltext' / 'manifest.json'}")

    # --- Éligibilité sur texte intégral ---
    if do_eligibility:
        print("\n[éligibilité] Éligibilité sur texte intégral (relecteurs A et B + arbitre)...")
        elig = run_eligibility(run_dir, protocol)
        print_eligibility(elig)
        print(f"  Détail : {run_dir / 'fulltext' / 'eligibility_results.csv'}")

    # --- Extraction et risque de biais ---
    if do_extract:
        print("\n[extraction] Extraction des données et risque de biais (citations vérifiées)...")
        evidence = run_extraction(run_dir, protocol, cfg, limit=args.extract_limit)
        print_evidence(evidence)
        print(f"  Tableaux : {run_dir / 'evidence' / 'evidence_table.csv'} et risk_of_bias.csv")

    if args.reappraise_only:
        print("\n[risque de biais] Nouvelle évaluation du risque de biais + re-vérification des champs (sans ré-extraire)...")
        print_evidence(run_reappraisal(run_dir, protocol, cfg))
        print(f"  Tableaux : {run_dir / 'evidence' / 'evidence_table.csv'} et risk_of_bias.csv")

    # --- Livrables du rapport (sans LLM) ---
    if do_report:
        print("\n[rapport] Tableaux d'évidence (décisions humaines appliquées)...")
        print_evidence(refresh_evidence_tables(run_dir, cfg))
        print("\n[rapport] Diagramme PRISMA, références BibTeX et tableaux...")
        print_report(write_report_data(run_dir))

    if do_manuscript:
        from agents.manuscript import print_manuscript, write_manuscript

        print("\n[manuscrit] Brouillon de manuscrit PRISMA (méthodes par gabarit, synthèse contrôlée)...")
        print_manuscript(write_manuscript(run_dir, protocol, cfg))

    if args.review:
        from agents.review import interactive_review

        print(f"\n[relecture humaine] {args.review}...")
        counts = interactive_review(run_dir, args.review, protocol, limit=args.review_limit)
        if any(v for k, v in counts.items() if k != "skipped"):
            print("  Application des décisions (reprise de l'étape, sans nouvel appel au LLM)...")
            if args.review == "screening":
                print_screening(run_screening(run_dir, protocol))
            else:
                print_eligibility(run_eligibility(run_dir, protocol))

    if args.qa or args.qa_only:
        from agents.critic import print_qa, run_qa

        print("\n[contrôle qualité] Agent critique...")
        print_qa(run_qa(run_dir, protocol))

    if args.eval_prepare or args.eval or args.eval_repro:
        from agents import evaluation as ev

        if args.eval_prepare:
            print("\n[évaluation] Gabarits de vérité terrain...")
            for f in (ev.prepare_screening_gold(run_dir), ev.prepare_extraction_audit(run_dir), ev.instructions_md(run_dir)):
                print(f"  {f}")
        if args.eval_repro:
            print(f"\n[évaluation] Reproductibilité : nouveau screening de {args.eval_repro} articles...")
            ev.run_reproducibility(run_dir, n=args.eval_repro)
        print("\n[évaluation] Rapport...")
        ev.print_evaluation(ev.evaluation_report(run_dir, cfg))

    print(f"\nTerminé. Fichiers dans : {run_dir}")
    return 0


def run_with_graph(args) -> int:
    """Pipeline complet orchestré par LangGraph (graph/pipeline.py)."""
    from graph.pipeline import mermaid, run_graph

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config()
    if args.resume:
        run_dir = Path(args.resume)
    else:
        run_dir = Path("outputs") / f"run_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "topic.txt").write_text(args.topic, encoding="utf-8")
    set_trace_file(run_dir / "trace.jsonl")
    check_model()
    (run_dir / "report").mkdir(exist_ok=True)
    (run_dir / "report" / "pipeline_graph.mmd").write_text(mermaid(cfg), encoding="utf-8")
    print(f"[graphe LangGraph] run : {run_dir}")
    max_results = args.max_results or cfg.get("search", {}).get("max_results_per_source", 200)
    state = run_graph(run_dir, cfg, topic=args.topic, max_results=max_results, auto=args.yes)
    return 1 if state.get("stop_reason") else 0


if __name__ == "__main__":
    sys.exit(main())
