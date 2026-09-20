"""
Fábricas de Embeddings e LLM
=============================================================================
Conceito — SEÇÃO 1 e 2 do curso

Centraliza a seleção de provedor sem que os outros módulos precisem
conhecer detalhes de API key ou nome de modelo.

Embeddings : NVIDIA NIM — nvidia/nv-embedqa-e5-v5 (1024 dims)

LLM — ordem de preferência (avaliada uma vez na inicialização):
  1. NVIDIA NIM — deepseek-ai/deepseek-v4-pro   (mais capaz)
  2. NVIDIA NIM — deepseek-ai/deepseek-v4-flash  (mais rápido)
  3. OpenRouter  — deepseek/deepseek-v4-flash    (fallback externo)

Feature flags (variáveis de ambiente):
  FEATURE_NVIDIA_LLM=false   → desabilita NVIDIA da cadeia LLM (padrão: false)
  FEATURE_NVIDIA_LLM=true    → habilita NVIDIA na cadeia LLM

A checagem de disponibilidade na NVIDIA é feita via GET /v1/models
na primeira chamada a get_llm() e o resultado fica em cache por
processo — sem overhead nas chamadas subsequentes.

Em runtime, se um modelo NVIDIA falhar (timeout, 5xx, etc.), o
FallbackLLM avança automaticamente para o próximo na cadeia.
"""
import os
import requests
import threading
from typing import Any

import config  # garante load_dotenv() antes de qualquer os.getenv()

from reliability import record_fallback, record_info

# ── Feature Flags ─────────────────────────────────────────────────────────────
# Para habilitar NVIDIA: defina FEATURE_NVIDIA_LLM=true no .env
FEATURE_NVIDIA_LLM = os.getenv("FEATURE_NVIDIA_LLM", "false").strip().lower() == "true"

# ── Chaves ────────────────────────────────────────────────────────────────────

_raw_nvidia_key      = os.getenv("NVIDIA_API_KEY", "")
_raw_openrouter_key  = os.getenv("OPENROUTER_API_KEY", "")

_nvidia_key_valid    = bool(_raw_nvidia_key) and _raw_nvidia_key != "sua_chave_aqui"

# NVIDIA na cadeia de LLM (geração de respostas) — controlado pela feature flag
NVIDIA_AVAILABLE     = FEATURE_NVIDIA_LLM and _nvidia_key_valid
OPENROUTER_AVAILABLE = bool(_raw_openrouter_key) and _raw_openrouter_key != "sua_chave_aqui"

# Mantido por compatibilidade com código legado
GOOGLE_AVAILABLE = False

# ── Embeddings — NVIDIA NIM ───────────────────────────────────────────────────
# Embeddings NÃO dependem de FEATURE_NVIDIA_LLM: a flag controla apenas o uso
# da NVIDIA na cadeia de LLM. Embeddings exigem só chave válida + biblioteca.
try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    _nvidia_embeddings_lib = True
except ImportError:
    _nvidia_embeddings_lib = False

NVIDIA_EMBEDDINGS_AVAILABLE = _nvidia_key_valid and _nvidia_embeddings_lib

# ── Embeddings locais (sentence-transformers) ────────────────────────────────
# EMBEDDING_PROVIDER=local  → modelo local (padrão quando não há chave NVIDIA)
# EMBEDDING_PROVIDER=nvidia → NVIDIA NIM (comportamento original do minicurso)
EMBEDDING_PROVIDER    = os.getenv(
    "EMBEDDING_PROVIDER", "nvidia" if NVIDIA_EMBEDDINGS_AVAILABLE else "local"
).strip().lower()
EMBEDDING_MODEL_LOCAL = os.getenv("EMBEDDING_MODEL_LOCAL", "BAAI/bge-m3")
# Modelo da NVIDIA NIM: nv-embedqa-e5-v5 (1024 dims, ~512 tokens) ou nemotron-3-embed-1b (2048 dims, ~4096 tokens)
EMBEDDING_MODEL_NVIDIA = os.getenv("EMBEDDING_MODEL_NVIDIA", "nvidia/nv-embedqa-e5-v5")
EMBEDDING_DEVICE      = os.getenv("EMBEDDING_DEVICE", "") or None   # None = auto

_local_embeddings_singleton = None
_local_embeddings_lock = threading.Lock()


class LocalEmbeddings:
    """
    Embeddings locais via sentence-transformers, com a mesma interface que o
    LangChain espera (embed_documents / embed_query). Vetores normalizados
    (norma 1), então distância cosseno == produto interno.

    Modelo padrão: BAAI/bge-m3 — multilíngue, 1024 dims (mesma dimensão do
    nv-embedqa-e5-v5, então os índices HNSW de create_indexes.sql continuam
    válidos). Troque via EMBEDDING_MODEL_LOCAL.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_LOCAL, device: str | None = EMBEDDING_DEVICE,
                 batch_size: int = 32):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self.dims = self._model.get_sentence_embedding_dimension()
        # bge-m3 aceita até 8192 tokens; limitamos para caber na memória em lotes
        self._model.max_seq_length = min(self._model.max_seq_length or 2048, 2048)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=len(texts) > 64, convert_to_numpy=True,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    # Compatibilidade com Runnable do LangChain (alguns retrievers chamam __call__)
    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


def _get_local_embeddings() -> "LocalEmbeddings":
    global _local_embeddings_singleton
    if _local_embeddings_singleton is None:
        with _local_embeddings_lock:
            if _local_embeddings_singleton is None:
                record_info("embeddings", f"local: {EMBEDDING_MODEL_LOCAL}")
                print(f"[providers] Carregando embeddings locais: {EMBEDDING_MODEL_LOCAL} ...")
                _local_embeddings_singleton = LocalEmbeddings()
                d = _local_embeddings_singleton.dims
                if d != config.EMBEDDING_DIMS:
                    print(f"[providers] AVISO: modelo local tem {d} dims, mas config.EMBEDDING_DIMS="
                          f"{config.EMBEDDING_DIMS}. Ajuste config.py e reingira os dados.")
    return _local_embeddings_singleton

# ── Constantes de modelos LLM ─────────────────────────────────────────────────

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL_PRO    = "deepseek-ai/deepseek-v4-pro"
NVIDIA_MODEL_FLASH  = "deepseek-ai/deepseek-v4-flash"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL    = "deepseek/deepseek-v4-flash"

# ── Cache da lista de provedores disponíveis ──────────────────────────────────
# Lista de tuplas (base_url, model_id, api_key, label) em ordem de preferência.
# Preenchida na primeira chamada; estável pelo tempo de vida do processo.

_provider_chain: list[tuple[str, str, str, str]] | None = None
_provider_chain_lock = threading.Lock()


def _fetch_nvidia_models(timeout: float = 6.0) -> set[str]:
    """Retorna o conjunto de model IDs disponíveis na NVIDIA NIM, ou {} em erro."""
    if not NVIDIA_AVAILABLE:
        return set()
    try:
        resp = requests.get(
            f"{NVIDIA_NIM_BASE_URL}/models",
            headers={"Authorization": f"Bearer {_raw_nvidia_key}"},
            timeout=timeout,
        )
        if resp.ok:
            return {m["id"] for m in resp.json().get("data", [])}
        print(f"[providers] NVIDIA /models retornou {resp.status_code}")
    except Exception as exc:
        print(f"[providers] Aviso: não foi possível consultar NVIDIA NIM /models — {exc}")
    return set()


def _build_provider_chain() -> list[tuple[str, str, str, str]]:
    """
    Constrói a lista de provedores em ordem de preferência:
      1. NVIDIA pro  (se disponível)
      2. NVIDIA flash (se disponível)
      3. OpenRouter flash (fallback)

    Cada entrada: (base_url, model_id, api_key, label)
    """
    chain: list[tuple[str, str, str, str]] = []

    if NVIDIA_AVAILABLE:
        available = _fetch_nvidia_models()
        if NVIDIA_MODEL_PRO in available:
            chain.append((
                NVIDIA_NIM_BASE_URL, NVIDIA_MODEL_PRO,
                _raw_nvidia_key, f"NVIDIA NIM: {NVIDIA_MODEL_PRO}",
            ))
        if NVIDIA_MODEL_FLASH in available:
            chain.append((
                NVIDIA_NIM_BASE_URL, NVIDIA_MODEL_FLASH,
                _raw_nvidia_key, f"NVIDIA NIM: {NVIDIA_MODEL_FLASH}",
            ))

    if OPENROUTER_AVAILABLE:
        chain.append((
            OPENROUTER_BASE_URL, OPENROUTER_MODEL,
            _raw_openrouter_key, f"OpenRouter: {OPENROUTER_MODEL}",
        ))

    return chain


def _get_provider_chain() -> list[tuple[str, str, str, str]]:
    global _provider_chain
    if _provider_chain is not None:
        return _provider_chain
    with _provider_chain_lock:
        if _provider_chain is None:
            if not FEATURE_NVIDIA_LLM:
                print("[providers] FEATURE_NVIDIA_LLM=false — NVIDIA desabilitada por feature flag")
            _provider_chain = _build_provider_chain()
            labels = [lbl for *_, lbl in _provider_chain]
            print(f"[providers] Cadeia LLM: {' → '.join(labels) if labels else 'vazia'}")
        return _provider_chain


def _promote_provider(working_index: int) -> None:
    """
    Remove da cadeia global todos os provedores antes de `working_index`.
    Chamado quando um fallback ocorre em runtime — garante que as próximas
    requisições comecem diretamente no provedor que está funcionando,
    sem repetir timeouts desnecessários.
    """
    global _provider_chain
    if working_index <= 0 or _provider_chain is None:
        return
    with _provider_chain_lock:
        if _provider_chain is not None and working_index < len(_provider_chain):
            removed  = [lbl for *_, lbl in _provider_chain[:working_index]]
            _provider_chain = _provider_chain[working_index:]
            new_primary = _provider_chain[0][3] if _provider_chain else "nenhum"
            print(f"[providers] Promovido para primário: {new_primary} "
                  f"(removidos: {removed})")
            record_info("llm", f"Novo provedor primário após fallback: {new_primary}")


# ── FallbackLLM ───────────────────────────────────────────────────────────────

class FallbackLLM:
    """
    Wrapper que tenta cada provedor na cadeia em ordem.
    Se um falhar (timeout, erro 5xx, etc.) avança para o próximo.

    Expõe `.invoke()` e `.stream()` — a mesma interface do ChatOpenAI —
    para que o resto do código não precise saber sobre a cadeia.
    """

    def __init__(self, temperature: float = 0.3):
        self._temperature = temperature
        self._chain = _get_provider_chain()

    def _make_llm(self, base_url: str, model: str, api_key: str):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=self._temperature,
            request_timeout=30,
            max_retries=1,          # 1 retry interno; o FallbackLLM faz o fallback externo
        )

    def _try_chain(self, method: str, *args, **kwargs) -> Any:
        import openai

        # Lê a cadeia atual no momento da chamada (pode ter sido promovida)
        chain = _get_provider_chain()

        last_exc: Exception | None = None
        for idx, (base_url, model, api_key, label) in enumerate(chain):
            try:
                record_info("llm", label)
                llm = self._make_llm(base_url, model, api_key)
                result = getattr(llm, method)(*args, **kwargs)
                if idx > 0:
                    # Sucesso após fallback — promove este provedor para primário
                    # nas próximas requisições, evitando novos timeouts desnecessários.
                    record_fallback(
                        "llm",
                        f"{chain[idx-1][3]} falhou",
                        fallback_to=label,
                    )
                    _promote_provider(idx)
                return result
            except (openai.APITimeoutError, openai.APIStatusError, Exception) as exc:
                last_exc = exc
                if idx < len(chain) - 1:
                    next_label = chain[idx + 1][3]
                    print(f"[providers] {label} falhou ({type(exc).__name__}) — tentando {next_label}")
                    record_fallback("llm", f"{label}: {type(exc).__name__}", fallback_to=next_label)
                else:
                    print(f"[providers] {label} falhou ({type(exc).__name__}) — sem mais provedores")

        raise RuntimeError(
            f"Todos os provedores LLM falharam. Último erro: {last_exc}"
        ) from last_exc

    def invoke(self, *args, **kwargs) -> Any:
        return self._try_chain("invoke", *args, **kwargs)

    def stream(self, *args, **kwargs):
        # stream é lazy — não podemos fazer fallback após iniciar; usa invoke internamente
        return self._try_chain("stream", *args, **kwargs)

    # Suporte a chamada direta (compatibilidade com chain | llm)
    def __call__(self, *args, **kwargs) -> Any:
        return self._try_chain("invoke", *args, **kwargs)

    # Pipe operator para LangChain (chain | llm | parser)
    def __or__(self, other):
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(lambda x: self.invoke(x)) | other

    def __ror__(self, other):
        from langchain_core.runnables import RunnableLambda
        return other | RunnableLambda(lambda x: self.invoke(x))


# ── API pública ───────────────────────────────────────────────────────────────

def get_embeddings():
    """
    Retorna o modelo de embeddings conforme EMBEDDING_PROVIDER:
      local  → sentence-transformers (BAAI/bge-m3, 1024 dims) — sem API
      nvidia → NVIDIA NIM nvidia/nv-embedqa-e5-v5 (1024 dims)
    """
    if EMBEDDING_PROVIDER == "local":
        return _get_local_embeddings()
    if not NVIDIA_EMBEDDINGS_AVAILABLE:
        record_fallback("embeddings", "NVIDIA_API_KEY não configurada ou inválida")
        raise EnvironmentError(
            "NVIDIA_API_KEY não configurada ou inválida. "
            "Defina NVIDIA_API_KEY no arquivo .env — obrigatório para embeddings."
        )
    modelo = EMBEDDING_MODEL_NVIDIA
    record_info("embeddings", f"NVIDIA NIM: {modelo} ({config.EMBEDDING_DIMS} dims)")
    return NVIDIAEmbeddings(
        model=modelo,
        api_key=_raw_nvidia_key,
        truncate="END",
    )


def get_llm(temperature: float = 0.3) -> FallbackLLM:
    """
    Retorna um FallbackLLM que tenta os provedores nesta ordem:
      1. NVIDIA NIM deepseek-v4-pro
      2. NVIDIA NIM deepseek-v4-flash
      3. OpenRouter deepseek-v4-flash

    A disponibilidade na NVIDIA é checada via /v1/models na primeira chamada
    e fica em cache — sem overhead nas chamadas seguintes.
    Se um provedor falhar em runtime (timeout/5xx), avança para o próximo.
    """
    chain = _get_provider_chain()
    if not chain:
        raise EnvironmentError(
            "Nenhum provedor LLM disponível. Configure NVIDIA_API_KEY ou OPENROUTER_API_KEY no .env."
        )
    return FallbackLLM(temperature=temperature)


def get_llm_provider_info() -> dict:
    """Retorna informações sobre o provedor LLM primário ativo."""
    chain = _get_provider_chain()
    if not chain:
        return {"label": "nenhum", "model": "", "provider": "none"}
    base_url, model, _, label = chain[0]
    return {
        "label":    label,
        "model":    model,
        "provider": "nvidia" if "nvidia" in base_url else "openrouter",
        "chain":    [lbl for *_, lbl in chain],
    }


# =============================================================================
# Execução direta: testa os provedores disponíveis
# python providers.py
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  SEÇÃO 1 — Provedores de Embeddings e LLM")
    print("=" * 60)
    print(f"\n  NVIDIA embeddings    : {NVIDIA_EMBEDDINGS_AVAILABLE}")
    print(f"  NVIDIA LLM           : {NVIDIA_AVAILABLE}  (flag FEATURE_NVIDIA_LLM)")
    print(f"  OpenRouter disponível: {OPENROUTER_AVAILABLE}  (LLM fallback)")

    info = get_llm_provider_info()
    print(f"\n  Provedor primário : {info['label']}")
    print(f"  Cadeia completa   : {' → '.join(info['chain'])}")

    emb = get_embeddings()
    vetor = emb.embed_query("teste de embedding em português")
    print(f"\n  Embedding gerado: {len(vetor)} dims | primeiros 5: {[round(v,4) for v in vetor[:5]]}")

    print("\n  Testando LLM (com fallback automático)...")
    llm = get_llm()
    resposta = llm.invoke("Responda em uma frase: o que é um embedding?")
    print(f"\n  LLM respondeu: {resposta.content}")
