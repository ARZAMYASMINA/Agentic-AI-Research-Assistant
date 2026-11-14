# CONTEXTE DU PROJET — document de reprise

> **À quoi sert ce fichier.** Il résume tout ce qui a été fait dans la conversation précédente pour qu'on puisse continuer dans une nouvelle conversation sans rien perdre : le projet, l'environnement, l'architecture, l'état exact de chaque étape (écrite / exécutée / vérifiée), les résultats chiffrés, les décisions de conception, les bugs rencontrés, les limites connues et la suite logique.
> Le `README.md` du projet est le document « officiel » destiné au dépôt GitHub. Celui-ci est un document de travail interne.

---

## 0. Comment reprendre dans une nouvelle conversation

Collez ceci au début de la nouvelle conversation, avec le contenu de ce fichier :

```text
Je continue un projet existant (pipeline agentique de revues systématiques PRISMA avec Qwen3.5 9B via Ollama).
Tu as accès au dossier C:\Users\MSI\Desktop\projet M9 via le serveur MCP « M9 » (outils à charger avec tool_search).
Ne lis JAMAIS le dossier venv. Tu ne peux pas exécuter de code : tu écris et modifies les fichiers,
j'exécute les commandes dans PowerShell et je te colle les résultats.
Commence par lire README_CONTEXTE.md puis README.md, dis-moi où on en est, et propose la prochaine action.
Réponds en français.
```

**Règles de travail qui ont bien fonctionné** (à garder) :
1. On avance étape par étape : l'assistant écrit le code, l'utilisateur lance les commandes et colle les résultats, l'assistant les **analyse en lisant les fichiers produits** (`results.csv`, `records.jsonl`, `trace.jsonl`…) avant de conclure.
2. Chaque étape est testée sur un petit échantillon (`--screen-limit 10`, `--extract-limit 3`) avant le lancement complet.
3. Les décisions de fond sur le contenu scientifique (critères du protocole, inclusion/exclusion d'articles) appartiennent à l'utilisateur (principe du *human-in-the-loop*). L'assistant recommande, l'utilisateur valide.
4. Honnêteté sur les limites : tout ce qui est écrit mais non exécuté est signalé comme « non testé ».

---

## 1. Le projet

**Intitulé.** *Development of an Agentic AI Pipeline for Conducting Rigorous, Transparent, and Reproducible Systematic Reviews on Any Given Topic.*

**Objectif.** Système multi-agents (LangChain / LangGraph + RAG) qui, à partir d'une question de recherche, exécute une revue systématique de bout en bout selon **PRISMA 2020** : protocole, stratégie de recherche, recherche multi-sources, déduplication, screening (double relecture simulée), extraction de données, risque de biais, synthèse, diagramme PRISMA, brouillon de manuscrit. Supervision humaine aux points critiques, traçabilité complète.

**Agents prévus par l'énoncé** : Coordinator/Planner, Search Strategy, Retrieval (+ RAG), Screening (titre/résumé + texte intégral), Extraction & Appraisal, Synthesis & Writing, Critic/QA.

**Livrables exigés**
- Code source sur GitHub/GitLab + `README.md` détaillé (installation, configuration, exécution, usage, toutes les étapes suivies).
- Template de protocole PRISMA configurable.
- Démonstration de bout en bout sur au moins un vrai sujet.
- Rapport technique + guide utilisateur.
- **Évaluation** : complétude, exactitude, cohérence inter-agents, gain de temps par rapport à une revue manuelle.
- **Vidéo de démonstration de 3 minutes maximum.**
- Équipe de 5 personnes maximum ; **chaque étudiant décrit sa contribution individuelle dans le rapport final**.

**Critères de succès** : revue complète avec intervention humaine minimale ; sorties méthodologiquement saines, référencées, transparentes ; décisions de screening et d'extraction avec justification traçable ; gain de temps important à rigueur égale.

---

## 2. Environnement et conventions

| Élément | Valeur |
|---|---|
| OS | Windows (PowerShell) |
| Python | 3.11.9, environnement virtuel `venv` dans le dossier du projet |
| Dossier du projet | `C:\Users\MSI\Desktop\projet M9` |
| LLM | **Qwen3.5 9B** via **Ollama** (`qwen3.5:9b`), `temperature=0`, `seed=42`, `num_ctx=16384`, réflexion désactivée (`reasoning=False`) |
| Embeddings | `nomic-embed-text` via Ollama (`ollama pull nomic-embed-text`) |
| Accès fichiers de l'assistant | Serveur MCP « M9 » limité au dossier du projet (outils différés : `tool_search` d'abord) |
| Exécution de code par l'assistant | Possible dans Claude Code (desktop, PowerShell/Bash) depuis le 13/11/2026 ; impossible dans l'ancienne interface (MCP seul) |

**Dépendances installées au fil du projet** (si besoin : `pip install -r requirements.txt`) : `python-dotenv pyyaml pydantic langchain langchain-ollama chromadb rank_bm25 numpy requests pypdf pytest python-docx pillow` (+ `bibtexparser`, `langgraph` listés dans `requirements.txt` mais pas encore utilisés).

**Fichier `.env`** (copie de `.env.example`) : `OLLAMA_BASE_URL`, `OLLAMA_MODEL=qwen3.5:9b`, `OLLAMA_EMBED_MODEL=nomic-embed-text`, clés facultatives `NCBI_API_KEY`, `NCBI_EMAIL`, `SEMANTIC_SCHOLAR_API_KEY`, `OPENALEX_EMAIL`.

---

## 3. Arborescence du projet

```text
projet M9/
├── main.py                      # CLI : orchestre toutes les étapes (flags ci-dessous)
├── requirements.txt  .env.example  .gitignore  README.md  README_CONTEXTE.md
├── config/protocol_template.yaml   # sources, outil de biais (QUADAS-2), champs d'extraction (optionnel), LLM
├── agents/
│   ├── schemas.py        # tous les modèles Pydantic (Protocol, ConceptBlock, ScreeningDecision/Record, FieldValue, EvidenceRecord...)
│   ├── llm.py            # get_llm, check_model, structured_call (JSON dans le prompt), trace JSONL, garde SSL
│   ├── planner.py        # sujet -> protocole PICO + critères
│   ├── search_strategy.py# blocs de concepts (LLM) -> requêtes Boolean (code), make_queries/normalize_blocks
│   ├── retrieval.py      # exécute les requêtes, déduplique, écrit prisma_identification.json
│   ├── screening.py      # titre/résumé : relecteurs A/B + arbitre + règle de date
│   ├── fulltext.py       # récupération des textes intégraux open access (+ PDF manuels)
│   ├── eligibility.py    # éligibilité sur texte intégral (RAG par document) + décisions humaines (CSV)
│   ├── extraction.py     # extraction structurée avec citations vérifiées (groupes de champs)
│   ├── appraisal.py      # risque de biais QUADAS-2 / RoB 2, garde-fous
│   ├── evidence.py       # orchestre extraction + biais, tableaux CSV, résumé
│   ├── prisma.py         # effectifs PRISMA + diagramme SVG                       (6a, vérifié)
│   ├── report.py         # PRISMA + BibTeX + tableaux Markdown                    (6a, vérifié)
│   ├── manuscript.py     # brouillon de manuscrit : gabarits + tableaux + SWiM (6b)
│   ├── human.py          # décisions humaines partagées (screening + éligibilité)
│   ├── review.py         # relecture humaine interactive dans le terminal (7)
│   ├── citations.py      # références de l'introduction : recherche PubMed/OpenAlex + vérification (6b)
│   ├── critic.py         # agent critique / QA déterministe -> report/qa_report.md (7)
│   ├── observability.py  # appels LLM par agent, versions des prompts, empreintes des fichiers (7)
│   ├── evaluation.py     # évaluation : vérité terrain, métriques, audit humain, reproductibilité, temps (8)
│   └── swim.py           # synthèse SWiM : LLM relève les paires de valeurs, le code vérifie et rédige (6b)
├── tools/
│   ├── paper.py          # modèle Paper
│   ├── query_builder.py  # requêtes PubMed / Semantic Scholar / arXiv / OpenAlex
│   ├── search_apis.py    # clients HTTP des 4 sources
│   ├── dedup.py          # déduplication DOI -> PMID -> titre normalisé
│   ├── metrics.py        # kappa de Cohen
│   ├── fulltext.py       # Europe PMC XML, PDF OpenAlex/arXiv, parsing JATS/PDF
│   ├── quotes.py         # verify_quote, numbers_in_source (anti-hallucination)
│   ├── bibtex.py         # export BibTeX                                          (6a, vérifié)
│   └── docx_export.py    # Markdown -> DOCX minimal (python-docx)
├── rag/
│   ├── fusion.py         # tokenisation + Reciprocal Rank Fusion
│   ├── index.py          # index global Chroma (dense) + BM25 (sparse), recherche hybride
│   ├── chunks.py         # découpage d'un texte intégral en passages
│   └── doc_index.py      # index hybride en mémoire pour UN document (utilisé à l'éligibilité et à l'extraction)
├── graph/pipeline.py     # orchestration LangGraph : 13 nœuds, interruptions humaines, routage conditionnel
├── tests/                # test_query_builder, test_dedup, test_fusion, test_screening, test_fulltext,
│                         # test_chunks, test_eligibility, test_evidence, test_report
└── outputs/run_<date>/   # un dossier par exécution (voir §5)
```

---

## 4. Chaîne de traitement et commandes

Tout passe par `main.py`. Les étapes suivantes se lancent soit à la suite d'un run neuf, soit **séparément avec `--resume`**.

| Étape | Commande (`<RUN>` = `outputs\run_20261109_125531`) | Appelle le LLM ? |
|---|---|---|
| 1-2 Protocole + stratégie + récupération + index | `python main.py --max-results 50` (demande 2 validations humaines) | oui (planner, concepts) |
| Reprise : requêtes reconstruites sans LLM | `python main.py --resume <RUN>` | non |
| Index seul + test de recherche | `python main.py --resume <RUN> --index-only --query "second reader mammography"` | embeddings seulement |
| 4 Screening titre/résumé | `python main.py --resume <RUN> --screen-only [--screen-limit 10]` | oui |
| 5a Textes intégraux | `python main.py --resume <RUN> --fulltext-only` | non |
| 5b-1 Éligibilité texte intégral | `python main.py --resume <RUN> --eligibility-only` | oui |
| 5b-2 Extraction + risque de biais | `python main.py --resume <RUN> --extract-only [--extract-limit 3]` | oui |
| 5b-2' Risque de biais seul + re-vérification | `python main.py --resume <RUN> --reappraise-only` | oui |
| 6a Livrables du rapport | `python main.py --resume <RUN> --report-only` | non |
| 6b Brouillon de manuscrit (MD + DOCX) | `python main.py --resume <RUN> --manuscript-only` | oui (synthèse, en cache) |
| 7 Pipeline complet LangGraph | `python main.py --graph [--resume <RUN>] [--topic "..."] [--yes]` | oui (seulement ce qui manque) |
| 7 Relecture humaine interactive | `python main.py --resume <RUN> --review screening` (ou `eligibility`) `[--review-limit 10]` | non |
| 7 Contrôle qualité (agent critique) | `python main.py --resume <RUN> --qa-only` | non |
| 8 Gabarits de vérité terrain | `python main.py --resume <RUN> --eval-prepare` | non |
| 8 Reproductibilité (nouveau screening de N articles) | `python main.py --resume <RUN> --eval-repro 30` | oui (~10 min) |
| 8 Rapport d'évaluation | `python main.py --resume <RUN> --eval` | non |

Options : `--topic "..."`, `--yes` (saute les validations), `--no-retrieve`, `--max-results N`, `--screen` / `--fulltext` / `--eligibility` / `--extract` / `--report` (enchaînent l'étape après la précédente).
Tests : `pytest tests/`.

**Chaque étape de décision est reprenable et archive ses anciens résultats** si le protocole, les prompts, les champs ou la disponibilité des textes ont changé (fichiers `*_signature.txt` + `*.bak.jsonl`).

---

## 5. Fichiers produits dans un run

```text
outputs/run_<date>/
├── topic.txt  protocol.json  search_strategy.json  trace.jsonl   # trace de TOUS les appels LLM (prompts, sorties brutes)
├── retrieval/<source>.json  records_deduplicated.json  prisma_identification.json
├── chroma/  index_info.json                                      # index RAG global
├── screening/   decisions.jsonl  results.csv  prisma_screening.json  included_for_fulltext.json  human_overrides.csv
├── fulltext/    text/<id>.txt  pdf/<id>.pdf  manifest.json  prisma_fulltext_retrieval.json
│                eligibility.jsonl  eligibility_results.csv  prisma_eligibility.json
│                included_for_extraction.json  human_overrides.csv
├── evidence/    records.jsonl  evidence_table.csv  risk_of_bias.csv  prisma_evidence.json
└── report/      prisma_flow.png  prisma_flow.json  references.bib  tables.md       (après 6a)
                 qa_report.md  qa.json  pipeline_graph.mmd                        (après 7)
└── evaluation/  INSTRUCTIONS.md  screening_gold.csv  extraction_audit.csv  reproducibility.jsonl
                 evaluation_report.md  evaluation.json                            (après 8)
                 manuscript.md  manuscript.docx  swim.json (comparaisons acceptées/rejetées, cache)  (après 6b)
```

---

## 6. État d'avancement (honnête)

Légende : **✔ vérifié** = exécuté par l'utilisateur et résultats analysés ; **◐ écrit, non vérifié** = code livré mais pas exécuté/rapporté.

| # | Étape | Statut | Remarques |
|---|---|---|---|
| 1 | Structure, config, README initial | ✔ | |
| 2 | Protocole PICO + stratégie de recherche | ✔ | Bug de sortie structurée corrigé (voir §8) |
| 3a | Récupération 4 sources + déduplication + chiffres PRISMA | ✔ | Bug « bloc outcome dans la requête » corrigé |
| 3b | Index RAG hybride (Chroma + BM25 + RRF) | ✔ | Requête de test pertinente |
| 4 | Screening titre/résumé (A/B + arbitre) | ✔ | Complet sur 167 articles |
| 5a | Textes intégraux open access | ✔ | 9 sur 29 obtenus |
| 5b-1 | Éligibilité sur texte intégral + décisions humaines | ✔ | 3 décisions humaines appliquées |
| 5b-2 | Extraction + risque de biais | ✔ | 13 études traitées (13/11/2026) ; risque de biais refait avec `--reappraise-only` (voir §7) |
| 6a | Diagramme PRISMA SVG, BibTeX, tableaux Markdown | ✔ | Exécuté le 13/11/2026 : effectifs cohérents (176→167→29→18→13), 13 entrées BibTeX, aucun avertissement. Libellés d'exclusion raccourcis (code + critère coupé au mot, ex. « I1 not met: … ») |
| 6b | Synthèse narrative + manuscrit | ◐ | Exécuté le 13/11/2026 (`--manuscript-only`, ~1 min 30 s, 4 à 5 appels LLM, cache) : `report/manuscript.md` + `.docx`. Méthodes/effectifs/limites par gabarit : exacts. **Synthèse SWiM par direction des effets** (13/11) : 12 comparaisons vérifiées dans 6 études, toutes relues et exactes ; texte entièrement généré par le code. Reste : Introduction et discussion comparative à écrire par l'équipe ; relecture humaine globale |
| 7 | Agent critique, relecture humaine interactive, LangGraph | ✔ | 13/11/2026 : graphe LangGraph complet (`--graph`) exécuté sur le run de référence (4,6 s, 0 appel LLM, reprise depuis les fichiers) ; agent critique (`--qa-only`) : 0 erreur, 10 points à relire ; relecture interactive (`--review screening|eligibility`) testée (affichage), **pas encore utilisée par l'utilisateur : 24 + 16 décisions en attente** |
| 8 | Évaluation (vérité terrain, métriques, temps) | ◐ | 13/11/2026 : outils écrits et testés (`agents/evaluation.py`) ; rapport `evaluation/evaluation_report.md` ; reproductibilité mesurée (voir §7) ; **gabarits de vérité terrain à remplir par l'équipe** (100 articles de screening, 88 valeurs d'extraction) |
| 9 | Livrables finaux (README, rapport, vidéo, GitHub) | ✘ | À faire |

**Tests unitaires** : `pytest tests/` → **123 tests passent** (13/11/2026, toute la suite). ~~seule la première série (3 tests de `test_query_builder.py`) a été confirmée par l'utilisateur.~~

---

## 7. Run de référence : `outputs\run_20261109_125531`

**Sujet.** *Effectiveness of artificial intelligence for breast cancer detection in screening mammography.*
(`--max-results 50` pendant le développement.)

**Protocole modifié et validé par l'utilisateur** (après un premier jeu de critères trop restrictif qui excluait presque tout) :
- Question : *how does the use of AI algorithms for breast cancer detection perform compared with standard radiologist interpretation* (dépistage par mammographie).
- **I1** étude primaire comparant l'IA (seule ou avec radiologues) à la lecture standard (RCT, cohorte prospective/rétrospective, étude de lecteurs) · **I2** femmes en dépistage (pas de bilan diagnostique) · **I3** IA appliquée à des mammographies de dépistage, quel que soit son rôle (lecteur seul, second lecteur, concurrent, tri) · **I4** mesures quantitatives de précision diagnostique · **I5** au moins un algorithme précis évalué en contexte de dépistage.
- **E1** mammographie diagnostique de patientes symptomatiques · **E2** cas cliniques, éditoriaux, commentaires, revues narratives sans données comparatives · **E3** études animales, fantômes, modélisation théorique · **E4** IA pour une tâche autre que la détection (ex. densité seule) · **E5** langue non anglaise non traduisible · **E6** doublons.
- Règle automatique : année hors `date_from`–`date_to` (2015–…) → exclusion (`RULE:date`).

### Résultats par phase

| Phase | Chiffres |
|---|---|
| **Identification** | PubMed 50 récupérés / 5 690 disponibles ; Semantic Scholar 50 / 496 ; arXiv 26 / 26 ; OpenAlex 50 / 9 128. Total 176, doublons 9, **167 articles uniques**, 22 sans résumé |
| **Screening titre/résumé** | 167 screenés, **138 exclus, 29 passent** (10 include + 19 maybe), 10 arbitrages, accord A/B 0,94, **kappa 0,779**, 24 à revoir |
| **Textes intégraux** | 29 recherchés : **9 texte intégral** (5 PDF + 4 Europe PMC XML), 9 résumé seul, 11 ni l'un ni l'autre |
| **Éligibilité** (après 3 décisions humaines) | 11 non récupérés, 18 évalués (9 texte intégral + 9 résumé seul) → **5 exclus, 7 inclus, 6 « maybe »**, kappa 0,609 |
| **À extraire** | **13 études** : 7 sur texte intégral + 6 sur résumé seul |

**Décisions humaines appliquées** (`fulltext/human_overrides.csv`) : `P0161 exclude` (commentaire sur P0131), `P0123 include` (données de dépistage confirmées), `P0143 maybe` (poster, exclusion sur « absence de chiffres » non justifiée).

**Études retenues**
- *Inclus (7, texte intégral)* : P0015, P0031, P0040, P0065, P0109, P0123, P0165.
- *« Maybe » (6, résumé seul, à trancher)* : P0013, P0043, P0050, P0131, P0148, P0143.
- *Exclus à l'éligibilité (5)* : P0041, P0166 (revues systématiques), P0052 (modèle sans comparaison à des lecteurs), P0156 (revue sur la 3D), P0161 (commentaire).
- *Non récupérés (11)* : P0004, P0026, P0064, P0070, P0073, P0077, P0087, P0091, P0127, P0132 (McKinney, *Nature*), P0155. La plupart sont des commentaires/enquêtes écartables par leur titre ; P0132, P0077, P0087, P0070 pourraient compter.
- L'utilisateur **n'a pas accès** aux PDF de P0013, P0043, P0131, P0148, P0132 (paywall). Mécanisme prévu : déposer `fulltext\pdf\<id>.pdf` puis relancer `--fulltext-only`.

**Extraction testée sur P0013, P0015, P0031 (dernier essai)** : 29/39 champs renseignés (74 %), 25/29 citations exactes, 27/29 valeurs étayées (93 %). Risque de biais : P0013 *unclear* (résumé seul), P0015 *low/low/low/unclear*, P0031 *low/high/high/low → high* (cohérent avec les aveux des auteurs : seuils calibrés sur les mêmes données, biais de vérification).

**Extraction complète des 13 études (13/11/2026)** : 110/169 champs renseignés (65,1 %), 94/110 citations exactes (85,5 %), **101/110 valeurs étayées (91,8 %)** après les corrections ci-dessous. Risque de biais global : **8 unclear (dont les 6 sur résumé seul), 1 low (P0015), 4 high (P0031, P0040, P0065, P0165)**. P0109 et P0123 restent presque entièrement « unclear » (le modèle ne trouve pas de passage dans les extraits). Ré-évaluation du biais : ~210 s pour 7 études ; deux passages successifs donnent les mêmes jugements (déterminisme vérifié).
**À relire par un humain** : P0015 *flow_and_timing* « low » et P0065 *flow_and_timing* « low » s'appuient sur des citations qui ne traitent pas vraiment du domaine (même citation réutilisée pour 3 domaines chez P0015) ; P0065 *patient_selection* « high » sur des critères techniques d'inclusion (discutable). P0165 : la sensibilité/spécificité des radiologues (85,7 % / 93,4 %) est dans le texte mais n'a pas été extraite (contexte RAG limité à 8 000 caractères). P0143 (poster) : 11/13 champs vides, candidat à l'exclusion (décision humaine).

### Décisions humaines du 13/11/2026 (validées par l'utilisateur)
**Éligibilité** (`fulltext/human_overrides.csv`, 8 lignes) : les 6 « maybe » sont tranchés : **P0013, P0131, P0043 inclus** (P0043 = tomosynthèse, contexte de dépistage à confirmer sur le texte intégral) ; **P0050 exclu (I1**, comparé à un autre réseau et non à des radiologues) ; **P0143 exclu (I4**, poster sans données ; décision « maybe » précédente révisée) ; **P0148 exclu (I1**, 29 cancers d'intervalle sans cas normaux, cas limite).
→ **10 études incluses** (7 texte intégral + 3 résumé seul), 8 exclues à l'éligibilité, 11 non récupérées.

**Extraction / risque de biais** (`evidence/human_overrides.csv`, colonnes `id,kind,name,value,note`, `kind` = `field` ou `rob`) : 8 corrections.
- P0015 : flow_and_timing **high**, reference_standard **unclear**, index_test **unclear** (limites reconnues par les auteurs : differential/partial verification, incorporation bias, données d'entraînement partiellement communes) → global **high** (au lieu de low).
- P0065 : patient_selection **low** (« prospective consecutive case series »), flow_and_timing **high** (suivi insuffisant).
- P0165 : sensibilité/spécificité des radiologues ajoutées (85,7 %/93,4 % interne ; 87,2 %/93,4 % externe) ; **AUC corrigée : 0,944 / 0,951** (0,982 = sous-ensemble de la décision déléguée seulement — erreur « valeur exacte mais hors contexte », invisible aux contrôles automatiques).

**Chiffres actuels (10 études, corrections appliquées)** : 96/130 champs (73,8 %), 89/96 valeurs étayées (92,7 %), risque de biais global 5 high / 5 unclear. Dans le rapport, † = corrigé par un humain, ⚠ = non étayé.

### Manuscrit 6b (13/11/2026)
`report/manuscript.md` + `manuscript.docx` : avertissement en tête ; Introduction et « comparaison avec d'autres revues » laissées à l'équipe ; **Méthodes, sélection des études (effectifs, kappa), limites, références et annexe des requêtes générées par gabarit à partir des fichiers du run (exacts, vérifiés)** ; tableaux 6a ; synthèse en 4 sections rédigées par le LLM uniquement à partir des valeurs étayées.
Contrôles automatiques par **proposition** (texte avant chaque citation `[Pxxxx]`) : nombres présents dans les données des études citées, identifiants connus, et affirmations sur le plan d'étude (rétrospectif/prospectif), le niveau de preuve (résumé seul/texte intégral) et le risque de biais. Paragraphe en échec → renvoyé une fois au modèle avec la liste des problèmes (révision gardée seulement si meilleure), puis marqué ⚠.
Résultat : 7 paragraphes, 2 signalés. Sur la 1re version, les nouveaux contrôles détectaient 3 erreurs de sens sur 5 relevées à la main.
**Erreurs NON détectées (relecture humaine)** : inversion de valeurs à l'intérieur d'une même étude, ex. P0043 « specificity improved to 89.64% vs 77.34%; sensitivity higher without AI (85.44%) than with it (87.69%) » (faux : 89,64 % = IA seule, et la sensibilité augmente avec l'IA) ; Gommers (P0013, aide à la lecture) classé en « stand-alone » ; « over half a million » sans chiffre vérifiable. → limite citée dans le manuscrit ; prochaine décision : synthèse sans chiffres (chiffres seulement dans les tableaux) et/ou agent Critic (étape 7).

**Synthèse sans chiffres (13/11/2026, choix de l'utilisateur)** : le prompt interdit tout chiffre (les valeurs restent dans le tableau 3, généré par le code) ; tout chiffre dans la prose est signalé (`check_paragraph(..., allow_numbers=False)`, `CHECK_VERSION = "3-no-numbers"`). Versions précédentes conservées dans le scratchpad (synthesis.v1/v2). Résultat : 7 paragraphes, 1 signalé, **mais 6 erreurs qualitatives relevées à la main** : effets attribués à la mauvaise étude (P0013 ↔ P0043), comparaison affirmée alors qu'une valeur n'est pas rapportée (AUC radiologues P0123, P0165), valeurs rapportées dites « non rapportées » (P0131), direction inversée (spécificité de P0131 : 93,0 % combinée < 96,6 % radiologues, écrite « plus élevée avec l'IA »).
**Conclusion** : avec Qwen3.5 9B, la prose libre n'est fiable ni avec ni sans chiffres. Piste proposée : synthèse **SWiM / effect direction** — le LLM remplit pour chaque étude et chaque critère (sensibilité, spécificité, AUC, temps de lecture…) une petite fiche structurée (rôle de l'IA, valeur IA, valeur comparateur, direction), le code vérifie que les deux valeurs existent dans les données et que la direction correspond à leur comparaison numérique, puis le texte et un tableau de direction des effets sont générés par le code.

**Synthèse SWiM (13/11/2026, choix de l'utilisateur, version retenue)** — `agents/swim.py`.
- Le LLM ne fait qu'une micro-tâche par étude : lister les paires (étiquette + valeur du bras IA, étiquette + valeur du comparateur) pour sensibilité, spécificité, AUC, taux de détection, taux de rappel, temps de lecture.
- **Le code vérifie chaque paire** : les deux valeurs sont sur la ligne de l'issue, chacune est le 1er nombre après son étiquette (pas une borne d'IC), dans des segments « ; » différents, et tous les mots de groupe (with, without, unaided, radiologists, AI, first…) avant la valeur figurent dans l'étiquette (« AI » refusé pour « radiologists without AI »).
- **Le rôle de l'IA est déduit des étiquettes** (with AI / AI-supported → assisted ; AI vs radiologists → standalone) : le modèle avait classé « AI 89.64% vs radiologists 77.34% » (P0043) en « assisted », erreur de la prose libre.
- **Le code relance le modèle une fois** si des paires sont rejetées ou s'il voit dans une ligne un groupe IA et un groupe comparateur sans comparaison renvoyée (P0165, P0131, P0031 étaient vides au 1er essai).
- Direction calculée par le code : ▲/▼/◆ (seuil relatif 2 %, `synthesis.no_difference_relative` dans la config), ◐ = mixte dans l'étude ; décompte par étude et test du signe exact ; tableau 4 + texte + conclusion par gabarit (décomptes exacts, pas d'adjectifs).
- **Résultat** : 12 comparaisons acceptées (6 études), 7 rejetées (toutes à juste titre) ; **les 12 vérifiées à la main : toutes exactes**. Pas de comparaison pour P0015, P0040, P0123 (pas de valeur radiologue) ni P0065 (format « 75.1% (without AI) vs 76.0% (with device) » : étiquette après la valeur → perdu ; corrigeable via `evidence/human_overrides.csv` en réécrivant la ligne « Without AI: 75.1%; With AI: 76.0% »). Partiel : P0131 (AI-1 seulement), P0165 (interne seulement).
- Bug général corrigé dans `structured_call` : le modèle recopiait parfois le **schéma JSON** au lieu de répondre ; avec des champs par défaut, cela validait à vide. Un schéma recopié est maintenant une erreur (nouvel essai).
- Durée : ~2 min (10 études, 14 appels), puis cache (`report/swim.json`).

### Étape 7 (13/11/2026)
- **LangGraph** (`graph/pipeline.py`, `--graph`) : 13 nœuds (plan → strategy → retrieve → index → screen → review_screening → fulltext → eligibility → review_eligibility → extract → report → manuscript → critic). Le dossier du run est l'état durable (chaque nœud reprend depuis ses fichiers) ; `MemorySaver` + `interrupt()` pour la supervision humaine dans une exécution : validation du protocole (écrit `protocol_approved.txt`), puis « relire maintenant / plus tard / stop » quand des décisions sont en attente. Routage : arrêt si 0 article, protocole refusé, 0 étude incluse, ou erreurs du critique. Diagramme exporté : `report/pipeline_graph.mmd` (Mermaid, pour le rapport). `--yes` = sans interruption (la validation n'est PAS écrite, les relectures sont reportées).
- **Le protocole du run de référence n'a pas de `protocol_approved.txt`** : il a été validé dans la conversation précédente ; pour l'enregistrer, lancer `python main.py --graph --resume outputs\run_20261109_125531` et répondre « oui ».
- **Relecture humaine interactive** (`--review screening|eligibility`) : fiche par article (titre, résumé, A/B/arbitre, décision automatique), réponses i/e/m/k/s/q, note obligatoire, critère pour une exclusion → `<étape>/human_overrides.csv` (module commun `agents/human.py` ; **le screening applique maintenant aussi les décisions humaines**), puis l'étape est relancée sans LLM.
- **Agent critique** (`agents/critic.py`, `--qa-only`, dernier nœud du graphe), **sans LLM** par choix (leçons 14-16) : effectifs PRISMA entre étapes/rapport/manuscrit, période du protocole, codes de critères, « maybe » en attente, relectures en attente, kappa < 0,6, valeurs non étayées, biais low/high sans citation, études tout « unclear », SWiM sans comparaison, livrables périmés, erreurs de la trace. Résultat sur le run : **0 erreur, 10 à relire** (24 screening + 16 éligibilité en attente, 6 études avec valeurs non étayées, P0123 tout « unclear », 4 études sans comparaison SWiM).
- **Observabilité** (`agents/observability.py`, dans `qa_report.md`) : appels/échecs/essais/temps par agent depuis `trace.jsonl`, empreintes des prompts et des fichiers clés. Attention : la trace **cumule toutes les exécutions de développement** (ex. 207 appels screening_a pour 167 articles) ; total cumulé ≈ 7 574 s de LLM.

### Étape 8 : évaluation (13/11/2026)
- **Vérité terrain screening** (`evaluation/screening_gold.csv`, créé une seule fois, jamais écrasé) : 100 articles = les 29 retenus par l'agent + 71 exclus tirés au hasard (graine 2026), ordre mélangé, **sans les décisions de l'agent** (aveugle). Deux membres remplissent `reviewer1` / `reviewer2` (include/exclude/maybe), puis `consensus` en cas de désaccord ; noter le temps passé. Métriques : rappel, précision, F1, spécificité, WSS au rappel atteint, kappa agent-humain et humain-humain ; positif = include ou maybe. Limite : rappel estimé seulement sur les exclus échantillonnés.
- **Audit d'extraction** (`evaluation/extraction_audit.csv`) : 88 valeurs des 7 études en texte intégral (hors valeurs déjà corrigées), verdict correct / partial / incorrect / missed / na → exactitude stricte et large, par champ. Audit **non aveugle** (valeur de l'agent visible) : à signaler.
- **Audit des décisions relues** (déjà disponible) : éligibilité 8 relues → 1 renversée (P0123), 6 « maybe » tranchés, 1 confirmée ; biais 5/28 domaines corrigés ; extraction 3/96 valeurs corrigées ; SWiM 12 acceptées / 7 rejetées (+ vérification manuelle 12/12 exactes, §7). Relectures ciblées, pas des estimations d'exactitude.
- **Cohérence** : kappa A/B screening 0,779 (accord 0,94), éligibilité 0,609.
- **Reproductibilité** (`--eval-repro 30`, ~11 s/article, ~6 min) : nouveau screening de 30 articles avec les mêmes réglages (température 0, graine 42) → **29/30 décisions finales identiques (96,7 %), kappa 0,87** ; seul changement P0025 exclude → maybe (relecteur A). Mais la justification de A n'est identique mot pour mot que dans **4/30** cas : **Ollama n'est pas strictement déterministe** malgré température 0 et graine fixe ; la reproductibilité est donc au niveau des décisions, pas du texte — à dire dans le rapport.
- **Temps** : machine = durée moyenne mesurée par appel × appels d'un passage ; manuel = **hypothèses** configurables (`evaluation.manual_minutes`) : 1 min/résumé/relecteur, 10 min/texte intégral/relecteur, 60 min/extraction, 30 min/biais. Résultat : **≈ 77 min machine contre ≈ 1 684 min (28 h) manuelles, ratio ≈ 22** — hors temps humain de supervision, à mesurer par l'équipe.

### Manuscrit complété (13/11/2026, demande de l'utilisateur)
- **Diagramme PRISMA en PNG** (Pillow, `agents/prisma.py::render_png`, ~2 000 px) à la place du SVG : `report/prisma_flow.png`, **intégré au Word comme Figure 1** (l'export DOCX gère maintenant `![légende](image.png)`). L'ancien `prisma_flow.svg` a été supprimé du run (régénérable : `render_svg` existe toujours).
- **Abstract structuré** (Background/objective, Methods, Results, Conclusions, Keywords) rédigé par l'agent **à partir des chiffres vérifiés du run** (pas de prose libre du LLM, pour éviter les erreurs de sens des leçons 14-15) ; mots-clés génériques (valables pour tout sujet).
- **Introduction rédigée par le LLM** à partir du protocole seul (`write_introduction`, cache `report/introduction.json`) : interdiction des chiffres, des citations d'auteurs/organisations et des identifiants d'études incluses ; marqueur `[REFERENCE NEEDED]` exigé dans chaque paragraphe de contexte ; une révision si les contrôles échouent (ici 1 problème → 0). **À relire par l'équipe** : la phrase « rapid advancement and widespread adoption … in clinical practice » n'a pas de marqueur et n'est pas sourcée ; « breast cancer prevention programs » est inexact (le dépistage ne prévient pas) ; l'objectif est répété. Les marqueurs sont par paragraphe, pas par phrase.

### Agent de références de l'introduction (13/11/2026, demande de l'utilisateur) — `agents/citations.py`
- Chaque affirmation suivie de `[REFERENCE NEEDED]` (hors paragraphe d'objectif, qui n'est jamais sourcé) : le LLM propose 1-2 requêtes ; le code ajoute le sujet de la revue (1er terme du bloc population) si la requête ne le contient pas ; recherche **PubMed + OpenAlex**, toutes dates, articles avec résumé, études incluses exclues ; les sources déjà validées sont reproposées ; le LLM choisit le candidat qui soutient l'affirmation et copie la phrase du résumé ; un nouvel essai avec la raison du rejet.
- **Le code n'accepte que si** : verdict « supports », citation retrouvée mot pour mot dans le titre/résumé, **source dans le domaine** (racine d'un terme de population : breast / mammogr / neoplas), et **≥ 3 racines communes** entre l'affirmation et la citation (2 laissait passer « standard » ~ « standalone »).
- Marqueur remplacé par `[R1]`, `[R2]`… ; liste « References (background, introduction) » ; `report/references_background.bib` ; **annexe B** du manuscrit = affirmation / phrase justificative (audit) + affirmations restées sans source ; trace complète dans `report/intro_references.json`.
- Historique des essais (relus à la main) : 1re version 4/6 dont 2 fausses (IRM de la prostate ; article de segmentation cité pour l'objectif de NOTRE revue) → contrôle de sujet + objectif exclu ; puis faux positif « standard/standalone » → seuil 3. **Résultat actuel : 3/6 affirmations sourcées, 2 références (R1 AJR 2026 faux négatifs ; R2 Eur J Radiol Open 2026, revue systématique des modes d'intégration de l'IA), toutes pertinentes** ; 3 restent `[REFERENCE NEEDED]`. Le résultat varie d'une exécution à l'autre (LLM non déterministe, sources en ligne qui évoluent). R2 pour « it remains unclear… » s'appuie sur l'objectif d'une revue : soutien indirect, à relire.
- Word ouvert pendant la génération : `manuscript.docx` verrouillé → sauvegarde automatique sous `manuscript_<date>.docx` + avertissement.

---

## 8. Décisions de conception et leçons (à ne pas refaire)

1. **Sorties structurées d'Ollama.** `with_structured_output` (tool calling puis `json_schema`) faisait écrire un document Markdown à Qwen3.5. Solution retenue dans `agents/llm.py::structured_call` : schéma JSON **dans le prompt**, extraction tolérante du JSON (`extract_json`), validation Pydantic, réessais avec température/seed différents, erreur renvoyée au modèle. Chaque appel est tracé dans `trace.jsonl`. (Les premiers essais réutilisent aussi `format=schema` d'Ollama, sans compter dessus.)
2. **Requêtes de recherche.** Le LLM ne fournit que des **blocs de concepts** (synonymes, MeSH) ; le code construit les requêtes Boolean (reproductible). Le bloc *outcome* est exclu de la requête (il réduit le rappel) ; les MeSH génériques (Female, Adult, Aged…) sont filtrés ; le nom de concept est normalisé (`population`/`intervention`/`outcome`).
3. **Le protocole généré par un 9B peut être trop restrictif** : la première version exigeait RCT/cohorte avec témoin et IA « second lecteur » homologuée, ce qui excluait presque tout. Il faut **toujours relire les critères** à la validation humaine.
4. **Screening.** Deux relecteurs simulés (A favorise le rappel ; B favorise la précision, critères listés dans l'ordre inverse) avec **les mêmes critères numérotés I1…/E1…** et la justification écrite *avant* la décision. Arbitre seulement si l'un exclut et l'autre non. L'arbitre ne doit exclure que sur preuve explicite (pas sur « may/suggests »). Un article sans résumé → « maybe » (jamais deviné). Kappa calculé sur « passe / exclu ».
5. **Éligibilité sur texte intégral** : trois niveaux de preuve explicites (`full_text`, `abstract_only`, `none`). RAG par document : le texte est découpé (`rag/chunks.py`), indexé en mémoire (BM25 + embeddings), et chaque critère retrouve ses passages. Vérification *primaire vs secondaire* demandée en premier (a rattrapé les revues systématiques). Toute exclusion automatique est marquée « à revoir ».
6. **Décisions humaines** : fichier `fulltext/human_overrides.csv` (`id,decision,note,criterion`), appliqué par-dessus sans modifier les décisions automatiques (audit).
7. **Extraction : bug des noms de champs** (important). Le modèle renvoie souvent `sensitivity: AI`, `sensitivity: Radiologists`… au lieu de `sensitivity`. Ces entrées étaient jetées par une correspondance exacte, ce qui faisait croire à un refus du modèle. `finalize_fields` **fusionne maintenant** toutes les entrées dont le nom commence par celui du champ.
8. **Anti-hallucination** (`tools/quotes.py`) :
   - `verify_quote` : la citation existe mot pour mot dans le texte (insensible à la casse, aux espaces, aux apostrophes typographiques ; tolère « ... »).
   - `numbers_in_source` : tous les nombres significatifs (décimaux ou ≥ 3 chiffres) de la valeur figurent dans le texte.
   - `FieldValue.grounded` = citation vérifiée **ou** nombres retrouvés, **et jamais** si un nombre est introuvable (cas du modèle qui calcule `1495 = 72,6 % × 2033`).
   - Risque de biais : un jugement « low/high » sans citation retrouvée est ramené à « unclear » ; résumé seul → toujours « unclear » sans appel au LLM.
9. **Archivage automatique** quand le protocole, les prompts ou les données changent (évite de mélanger d'anciennes et de nouvelles décisions).
11. **Normalisation des sources pour la vérification** (`tools/quotes.py`, 13/11/2026) : (a) décimales avec point médian du style *The Lancet* (`84·2`) converties en point — sinon toutes les valeurs de P0165 étaient « non étayées » ; (b) mots coupés par une espace à l'extraction PDF (`diag nosis`, `eli gible`) : repli de comparaison sans espaces.
12. **Risque de biais : citation obligatoire** (13/11/2026). Le modèle laissait `evidence` vide et mettait la citation dans `rationale` → tout était ramené à « unclear ». Le prompt exige maintenant une citation pour « low »/« high », et `recover_quote` récupère un passage entre guillemets de la justification s'il est retrouvé mot pour mot. Nouvelle option **`--reappraise-only`** : refait seulement le risque de biais et re-vérifie les champs sans ré-extraire (archive l'ancien `records.jsonl`, met la signature à jour).
13. **Corrections humaines de l'extraction et du biais** (`evidence/human_overrides.csv`) : appliquées par-dessus `records.jsonl` (jamais modifié) à la lecture (`current_records`), avec la valeur/jugement automatique conservé dans la note (audit). `--report-only` réécrit d'abord `evidence_table.csv`, `risk_of_bias.csv`, `prisma_evidence.json` avec les études **actuellement incluses** et ces corrections. La signature de l'extraction ne dépend plus de la liste des études : exclure une étude ne force plus une ré-extraction complète ; un changement de niveau de preuve d'un article (PDF ajouté) le fait retraiter seul.
16. **SWiM : le LLM extrait, le code décide.** Faire relever des paires étiquette/valeur vérifiables puis calculer directions, décomptes et texte par le code a donné 12/12 comparaisons exactes là où la prose libre en faussait la plupart. Un modèle peut recopier le schéma JSON : ne jamais laisser une liste avec valeur par défaut valider une réponse vide.
15. **Prose qualitative sans chiffres : pas plus fiable.** Le modèle se trompe d'étude, de groupe ou de sens d'effet. Les comparaisons doivent être structurées (valeurs + direction) et vérifiées par le code avant d'être rédigées.
14. **Synthèse LLM : la vérification des nombres ne suffit pas.** Un 9B inverse des valeurs d'une même étude (IA vs radiologues, avec vs sans IA) : chaque nombre existe dans les données, le contrôle passe, la phrase est fausse. Les faits vérifiables (plan, niveau de preuve, risque de biais) sont contrôlés par proposition ; le reste exige une relecture humaine ou un agent critique.
10. **Problèmes d'environnement rencontrés** : `ModuleNotFoundError: dotenv` (→ installer les paquets) ; `SSL_CERT_FILE` pointant vers un chemin inexistant `%VIRTUAL_ENV%\...` (→ garde dans `llm.py`, ou `Remove-Item Env:SSL_CERT_FILE`) ; `model 'qwen3.5:9b' not found` (→ `ollama pull qwen3.5:9b`, vérifié maintenant par `check_model()`). Un bug connu d'Ollama 0.17.7 avec Qwen3.5 et le tool-calling a été signalé (non pertinent tant qu'on n'utilise pas le tool-calling ; version d'Ollama de l'utilisateur non précisée).

---

## 9. Limites connues (à citer dans le rapport) et écarts avec l'énoncé

**Limites scientifiques / techniques**
- **Rappel inconnu** : aucune vérité terrain humaine n'existe encore ; la précision n'a été estimée qu'en relisant les 29 articles retenus (8 vrais candidats clairs, 3 limites, 3 erreurs sur 14 articles avec résumé) et un petit échantillon d'exclus.
- **Kappa entre A et B** : même modèle, prompts proches → mesure de **cohérence**, pas de justesse. Peu informatif quand presque tout est exclu.
- **Accès aux textes** : seulement l'open access (9/29). 20 rapports sur 29 sans texte intégral.
- **Extraction depuis les tableaux** : les tableaux sont aplatis en texte ; le modèle peut mal attribuer les colonnes (exemple P0031 : NPV des radiologues « non rapporté » alors que la colonne de référence le donne). Le contrôle des nombres ne le détecte pas.
- **Nombres dérivés** : le modèle calcule parfois des totaux malgré l'interdiction (ex. « total: 2014 », « 1495 »/« 558 ») ; ils sont signalés dans `unverified_fields`, pas empêchés.
- **Risque de biais = brouillon automatique** : la vérification prouve qu'une citation *existe*, pas qu'elle *soutient* la conclusion. À relire par un humain.
- Les articles sur résumé seul ont une extraction partielle et un risque de biais toujours « unclear ».
- Échantillon de développement : 50 articles maximum par source.

**Écarts avec l'énoncé** (mis à jour 13/11/2026 : LangGraph, critique, relecture interactive et observabilité sont faits — voir §7 « Étape 7 » ; les points ci-dessous datent d'avant)
- **LangGraph n'est pas utilisé.** `graph/` est vide ; l'orchestration actuelle est séquentielle via les options de `main.py`. LangChain n'est utilisé que pour `ChatOllama`. L'énoncé attend LangChain/LangGraph pour l'orchestration, l'état et le routage.
- **Pas d'agent Critic/QA ni de Coordinator** distincts (le « Planner » ne fait que générer le protocole).
- **Pas d'interface de relecture humaine interactive** : seulement le CSV d'éligibilité ; les 24 décisions de screening « à revoir » n'ont pas de mécanisme de correction équivalent.
- **Pas encore de synthèse narrative ni de manuscrit**, pas d'export DOCX, pas d'évaluation chiffrée, pas de vidéo.

---

## 10. Suite logique : prochaines actions, dans l'ordre

### Immédiat (aucune écriture de code nécessaire)
1. `pytest tests/` : relever tout échec (les nouveaux tests de 5b, 6a n'ont jamais été exécutés ; erreurs possibles dans les tests eux-mêmes).
2. Extraction complète : `python main.py --resume outputs\run_20261109_125531 --extract-only` (13 articles, reprenable ; **noter la durée**, qu'on n'a jamais mesurée, utile pour l'évaluation).
3. Livrables : `python main.py --resume outputs\run_20261109_125531 --report-only` ; ouvrir `report\prisma_flow.png` dans un navigateur ; lire `report\tables.md`. Corriger les avertissements affichés (« ATTENTION : … »).
4. Relire `evidence\evidence_table.csv` (colonne `unverified_fields`) et `risk_of_bias.csv`.

### Étape 6b : synthèse narrative et manuscrit
- Assembler `report\manuscript.md` : **Méthodes** générées par gabarit à partir des données réelles (protocole, requêtes par base, critères, double relecture, kappa, outil QUADAS-2, limites de texte intégral), **Résultats** (effectifs PRISMA, tableaux de 6a), **Discussion/Limites** (gabarit + paragraphes courts).
- Paragraphes de synthèse rédigés par le LLM **uniquement à partir du tableau d'évidence**, chaque affirmation citant un identifiant (ex. `[P0015]`), puis **contrôle automatique** : tous les nombres du paragraphe doivent figurer dans les données fournies (réutiliser `numbers_in_source`) ; sinon le paragraphe est marqué.
- Mention claire en tête : *brouillon assisté par IA, à vérifier par un humain*.
- Export Markdown + BibTeX, puis DOCX (`python-docx`).

### Étape 7 : supervision, critique, orchestration
- **LangGraph** : graphe d'état (protocole → stratégie → récupération → screening → texte intégral → éligibilité → extraction → rapport) avec **points d'interruption humains** (validation du protocole, revue des cas « à revoir », décision sur les « maybe »), reprise, et routage conditionnel.
- **Agent Critic/QA** : vérifie la cohérence des effectifs PRISMA, le respect du protocole, les valeurs non étayées, les affirmations non soutenues dans le manuscrit, les écarts entre relecteurs.
- **Relecture humaine interactive** (CLI) pour le screening et l'éligibilité, avec le même principe d'audit que `human_overrides.csv`.
- Observabilité : agrégation de `trace.jsonl`, versions des prompts, empreintes des fichiers de run.

### Étape 8 : évaluation (exigée par l'énoncé)
- **Vérité terrain humaine** : deux membres de l'équipe screenent ~100 articles tirés au hasard (sur les 167 + quelques exclus) ; calculer **rappel, précision, F1, WSS@95** de l'agent, et le kappa agent–humain.
- **Exactitude de l'extraction** : extraction manuelle des champs clés sur les 7 études en texte intégral ; taux d'accord par champ.
- **Cohérence inter-agents** : kappa A/B (déjà calculé), stabilité entre deux exécutions (seed).
- **Gain de temps** : durée machine (logs) contre estimation manuelle (temps par résumé × nombre d'articles).
- Relancer avec `--max-results` plus grand (200) quand tout est validé ; idéalement **un second sujet** pour montrer la généralité (« any given topic »).

### Étape 9 : livrables finaux
- README final (installation, configuration, exécution, toutes les étapes), rapport technique et guide utilisateur, **contribution individuelle de chaque membre**, **vidéo ≤ 3 min**, dépôt GitHub propre (`.gitignore` exclut déjà `venv/`, `.env`, `outputs/` ; prévoir un dossier d'exemple avec un run de démonstration).

---

## 11. Aide-mémoire rapide

```powershell
cd "C:\Users\MSI\Desktop\projet M9"
venv\Scripts\activate
ollama list                               # doit afficher qwen3.5:9b et nomic-embed-text
pytest tests/

$RUN = "outputs\run_20261109_125531"
python main.py --resume $RUN --screen-only                  # reprend là où ça s'est arrêté
python main.py --resume $RUN --fulltext-only
python main.py --resume $RUN --eligibility-only
python main.py --resume $RUN --extract-only
python main.py --resume $RUN --report-only
```

**Corrections extraction / biais** : éditer `evidence\human_overrides.csv` (`id,kind,name,value,note` ; kind = field|rob) puis `--report-only`.
**Décisions humaines** : éditer `fulltext\human_overrides.csv` (4 colonnes, ligne d'en-tête conservée, pas de virgule dans `note`) puis relancer `--eligibility-only`.
**PDF manuel** : déposer `fulltext\pdf\P0013.pdf` (identifiant exact) puis `--fulltext-only`, `--eligibility-only`, `--extract-only`.

---

## 12. Pièges pour l'assistant qui reprendra

- Les outils M9 sont **différés** : appeler `tool_search` (ex. « M9 write_file », « M9 edit_file ») avant de les utiliser. Ne jamais lire `venv`.
- **`edit_file` : vérifier que le diff affiché contient bien un changement.** Trois fois, un édit « no-op » (`oldText` identique à `newText`) a été envoyé par erreur et a laissé croire qu'un test ou une correction était ajouté. Relire le diff à chaque fois.
- Quand on remplace une chaîne contenant `\n` écrit littéralement dans du code Python, l'échappement JSON doit être `\\n` ; pour éviter les erreurs, privilégier des ancres d'une seule ligne ou réécrire le fichier entier avec `write_file`.
- L'assistant ne peut pas exécuter le code : **ne jamais affirmer qu'un code fonctionne**. Tout est « non testé » tant que l'utilisateur n'a pas collé le résultat.
- Analyser les sorties réelles (fichiers de run) avant de diagnostiquer ; deux diagnostics ont été faux au début (prompt trop strict alors que c'était un bug de noms de champs ; protocole trop strict plutôt que screening défaillant).
- L'utilisateur communique en français et colle les résultats de console. Il valide l'étape puis dit « next » ; il n'a pas toujours rapporté le temps d'exécution ni les résultats de `pytest`.
