"""Accès au LLM local (Ollama / Qwen3.5 9B) + appels structurés tracés."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv

# Protection : une variable SSL pointant vers un fichier inexistant fait planter httpx
for _var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    _path = os.environ.get(_var)
    if _path and not os.path.exists(_path):
        logging.getLogger(__name__).warning(
            "%s pointe vers un fichier inexistant (%s) : variable ignorée.", _var, _path
        )
        os.environ.pop(_var, None)

from langchain_ollama import ChatOllama  # noqa: E402
from pydantic import BaseModel  # noqa: E402

load_dotenv()
log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_trace_path: Path | None = None


def set_trace_file(path: Path) -> None:
    """Active la trace JSONL de tous les appels LLM (transparence / reproductibilité)."""
    global _trace_path
    _trace_path = Path(path)
    _trace_path.parent.mkdir(parents=True, exist_ok=True)


def _trace(record: dict) -> None:
    if _trace_path is None:
        return
    with _trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_llm(temperature: float = 0.0) -> ChatOllama:
    kwargs = dict(
        model=os.getenv("OLLAMA_MODEL", "qwen3.5:9b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "16384")),
        seed=int(os.getenv("OLLAMA_SEED", "42")),
    )
    try:
        # Qwen3.5 est un modèle de raisonnement : on coupe la réflexion pour la vitesse
        return ChatOllama(**kwargs, reasoning=False)
    except Exception:  # version de langchain-ollama sans le paramètre `reasoning`
        log.warning("Paramètre `reasoning` non supporté, mise à jour de langchain-ollama conseillée.")
        return ChatOllama(**kwargs)


def check_model() -> None:
    """Vérifie qu'Ollama répond et que le modèle configuré est installé."""
    import ollama

    host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
    try:
        listed = ollama.Client(host=host).list()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Ollama injoignable sur {host} ({e}). Lancez Ollama puis réessayez.")

    models = listed["models"] if isinstance(listed, dict) else listed.models
    names = [getattr(m, "model", None) or m["model"] for m in models]
    if model not in names and f"{model}:latest" not in names:
        installed = "\n  ".join(names) or "(aucun modèle)"
        raise SystemExit(
            f"Modèle '{model}' introuvable dans Ollama.\n"
            f"Modèles installés :\n  {installed}\n"
            f"Solution : `ollama pull {model}` ou corrigez OLLAMA_MODEL dans .env"
        )
    print(f"Ollama OK : modèle '{model}' disponible.")


def extract_json(text: str) -> dict:
    """Extrait un objet JSON d'une réponse libre (balises <think>, blocs ```json, texte autour)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Aucun objet JSON trouvé dans la réponse")
    return json.loads(text[start : end + 1])


JSON_INSTRUCTIONS = (
    "\n\nRespond with ONLY a single valid JSON object that follows this JSON schema. "
    "No markdown, no headings, no explanations, no text before or after the JSON.\n"
    "JSON schema:\n{schema}"
)


def structured_call(
    schema: type[T],
    system: str,
    user: str,
    agent: str,
    retries: int = 3,
    llm: ChatOllama | None = None,
) -> T:
    """Appelle le LLM et renvoie un objet validé par le schéma Pydantic.

    Méthode indépendante du support natif d'Ollama : schéma dans le prompt, extraction
    tolérante du JSON, validation Pydantic, et renvoi de l'erreur au modèle en cas d'échec.
    """
    llm = llm or get_llm()
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    base_user = user + JSON_INSTRUCTIONS.format(schema=schema_json)
    base_seed = int(os.getenv("OLLAMA_SEED", "42"))
    last_error: Exception | None = None
    feedback = ""

    for attempt in range(1, retries + 1):
        t0 = time.time()
        raw = ""
        try:
            # Les essais suivants varient température/seed, sinon la sortie serait identique.
            update = {"temperature": 0.0 if attempt == 1 else 0.3 * (attempt - 1), "seed": base_seed + attempt}
            if attempt < retries:
                update["format"] = schema.model_json_schema()  # contrainte Ollama si elle est supportée
            attempt_llm = llm.model_copy(update=update)
            messages = [("system", system), ("human", base_user + feedback)]
            response = attempt_llm.invoke(messages)
            raw = response.content if isinstance(response.content, str) else str(response.content)
            data = extract_json(raw)
            if isinstance(data, dict) and ("$defs" in data or ("properties" in data and data.get("type") == "object")):
                # Le modèle a recopié le schéma : avec des champs par défaut, cela passerait la validation à vide.
                raise ValueError("you returned the JSON schema itself instead of data that follows it")
            result = schema.model_validate(data)
            _trace(
                {
                    "agent": agent,
                    "attempt": attempt,
                    "model": llm.model,
                    "seconds": round(time.time() - t0, 2),
                    "system": system,
                    "user": base_user + feedback,
                    "raw_output": raw,
                    "output": result.model_dump(),
                }
            )
            return result
        except Exception as e:  # noqa: BLE001
            last_error = e
            log.warning("[%s] tentative %d/%d échouée : %s", agent, attempt, retries, str(e)[:300])
            _trace({"agent": agent, "attempt": attempt, "error": str(e)[:2000], "raw_output": raw[:4000]})
            feedback = (
                f"\n\nYour previous answer was invalid ({str(e)[:400]}). "
                "Reply again with ONLY the JSON object, nothing else."
            )

    raise RuntimeError(f"[{agent}] échec après {retries} tentatives") from last_error
