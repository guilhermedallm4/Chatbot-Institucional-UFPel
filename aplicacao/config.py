"""
Configurações centralizadas do projeto RAG.
Único ponto de verdade para credenciais, conexão e constantes.
"""
import os
import warnings
from dotenv import load_dotenv

# Deve ser o primeiro módulo importado por qualquer outro arquivo do projeto,
# garantindo que os valores do .env estejam disponíveis antes de qualquer
# os.getenv() fora deste arquivo.
load_dotenv()

warnings.filterwarnings("ignore", message=".*langchain-community.*")

# =============================================================================
# BANCO DE DADOS
# =============================================================================

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",   "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",   "semanticdb"),
    "user":     os.getenv("DB_USER",   "gdlima"),
    "password": os.getenv("DB_PASS",   "12345"),
}

CONNECTION_STRING = (
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

# =============================================================================
# VECTOR STORE
# =============================================================================

# Coleção legado usada pelos exemplos didáticos de store.py
COLLECTION_NAME = "documentos_institucionais"

# Limiar mínimo de relevância: score LangChain = 1 − distância_cosseno
# 1.0 = idêntico ao query  |  0.0 = sem relação semântica
RELEVANCE_THRESHOLD = 0.2

# Dimensão dos vetores — deve bater com o modelo de embeddings em uso:
#   BAAI/bge-m3 (local) = 1024 | nvidia/nv-embedqa-e5-v5 = 1024 | nvidia/nemotron-3-embed-1b = 2048
# Os loaders em crawler/ (load_*.py) ajustam vector(N)/halfvec(N) dos schemas a este valor.
# HNSW sobre `vector` vai até 2000 dims; acima disso o índice usa a projeção halfvec (até 4000).
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", "1024"))

# =============================================================================
# COLLECTIONS POR TIPO — Portal UFPel
# =============================================================================
# Para adicionar um novo tipo: inclua uma entrada em COLLECTION_MAP e
# uma descrição correspondente em COLLECTION_DESCRIPTIONS.
# O roteador e a ingestão segmentada detectam automaticamente os novos tipos.

# Mapeamento tipo (campo 'tipo' no JSON crawleado) → nome da coleção pgvector
COLLECTION_MAP: dict[str, str] = {
    "disciplina":   "ufpel_disciplinas",
    "projeto":      "ufpel_projetos",
    "servidor":     "ufpel_servidores",
    "unidade":      "ufpel_unidades",
    "curso":        "ufpel_cursos",
    "gestao":       "ufpel_gestao",
    "sobre":        "ufpel_sobre",
    "portal_geral": "ufpel_portal_geral",
}

# Descrições exibidas ao LLM no prompt de roteamento.
# Chave = nome da coleção (valor de COLLECTION_MAP).
COLLECTION_DESCRIPTIONS: dict[str, str] = {
    "ufpel_disciplinas": (
        "disciplinas acadêmicas: ementa, objetivos, conteúdo programático, "
        "carga horária, departamento responsável"
    ),
    "ufpel_projetos": (
        "projetos de pesquisa e extensão: título, resumo, coordenador, "
        "área CNPq, ênfase, unidade de origem"
    ),
    "ufpel_servidores": (
        "professores e servidores técnico-administrativos: nome, cargo, "
        "unidade de lotação, titulação, regime de trabalho"
    ),
    "ufpel_unidades": (
        "unidades, centros, institutos e departamentos: endereço, "
        "direção/chefia, contato, estrutura organizacional"
    ),
    "ufpel_cursos": (
        "cursos de graduação e pós-graduação: duração, turno, vagas, "
        "grade curricular, coordenação"
    ),
    "ufpel_gestao": (
        "estrutura de gestão institucional: Reitoria, Vice-Reitoria, "
        "Pró-Reitorias, Direções, Coordenações, organogramas e contatos"
    ),
    "ufpel_sobre": (
        "informações sobre a UFPel: história, missão, visão, valores, "
        "estrutura institucional e dados gerais da universidade"
    ),
    "ufpel_portal_geral": (
        "informações institucionais gerais que não se encaixam nas "
        "categorias acima (editais, notícias, eventos, etc.)"
    ),
}

ALL_COLLECTIONS: list[str] = list(COLLECTION_MAP.values())


def collection_for_tipo(tipo: str) -> str:
    """Retorna o nome da coleção pgvector para um dado tipo de página."""
    return COLLECTION_MAP.get(tipo, COLLECTION_MAP["portal_geral"])

# =============================================================================
# CHUNKING
# =============================================================================

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# Overlap entre chunks adjacentes no chunking semântico (chunking_semantico.py),
# que agrupa seções inteiras e por isso não herda o CHUNK_OVERLAP do splitter.
SEMANTIC_CHUNK_OVERLAP = 20

# =============================================================================
# RAG — Parâmetros de retrieval e geração
# =============================================================================

# Candidatos recuperados antes de reranking (deve ser >= RAG_TOP_K_FINAL)
RAG_TOP_K_RETRIEVE: int = 10

# Documentos enviados ao LLM para síntese (mínimo recomendado: 4)
# Aumente para cobrir mais resultados; diminua para reduzir custo de tokens.
RAG_TOP_K_FINAL: int = 4

# Máximo de caracteres por documento no contexto do LLM
# Valores menores reduzem custo; valores maiores preservam mais contexto.
RAG_MAX_CHARS_PER_DOC: int = 1500

# Limiar mínimo de relevância para aceitar um resultado (0–1, cosine)
# Abaixo disso o pipeline tenta fallback multi-coleção.
RAG_RELEVANCE_THRESHOLD: float = 0.20

# Paginação de resultados: a busca vetorial traz até RAG_FETCH_ALL_K candidatos
# acima do limiar de relevância ("traga todos"), mas só RAG_TOP_K_FINAL são
# enviados ao LLM por vez. O restante fica no cache de sessão
# (conversation_state.py) e é apresentado quando o aluno pede
# "fale outras disciplinas", "mostre mais", etc.
RAG_FETCH_ALL_K: int = 40
