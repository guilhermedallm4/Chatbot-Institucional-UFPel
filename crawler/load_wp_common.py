"""
Infraestrutura de carga compartilhada — portal e PPGC
=============================================================================
Base de `load_portal_computacao.py` e `load_ppgc.py`. Mesmo contrato do
`load_computacao.py` (institucional), para que os três carregadores se
comportem igual: refresh completo do relacional, embeddings por faceta, e
`--dry-run` que mostra exatamente o texto que seria vetorizado.

Duas camadas
------------
  1. Relacional — TRUNCATE + INSERT em lote. Idempotente por construção:
     rodar duas vezes dá o mesmo estado. `preservar` protege as tabelas que
     NÃO podem ser recriadas a partir do crawl (o texto extraído dos PDFs do
     PPGC custa horas de processamento e não está no JSON do crawler).

  2. emb_* — um embedding por faceta da entidade, com FK de volta.

Sobre o texto que vai para o embedding
--------------------------------------
Todo trecho vetorizado é AUTOCONTIDO e começa se identificando:

    Notícia do PPGC/UFPel — assunto: edital
    Título: Seleção de Aluno Regular 2026/2 — Mestrado/Doutorado
    Publicada em 28 de maio de 2026

Um chunk recuperado entra no prompt sozinho, sem o registro vizinho. Se ele
não disser o que é, de quando é e a que se refere, o modelo preenche as
lacunas — e é aí que nasce resposta errada com data errada. Por isso a data
entra por extenso: "maio de 2026" casa semanticamente numa pergunta como "o
que saiu em maio?", enquanto "2026-05-28" não ativa praticamente nada.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import psycopg2
from psycopg2.extras import Json, execute_values

# Permite importar config/providers de aplicacao/
_APP_DIR = Path(__file__).resolve().parent.parent / "aplicacao"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import config                                            # noqa: E402

log = logging.getLogger("load_wp")

#: Tamanho do chunk vetorizado. Mesmo valor de load_computacao.py — o
#: nemotron-3-embed-1b aceita muito mais, mas chunk grande DILUI o sinal: um
#: "inteligência artificial" perdido em 20 mil caracteres pontua quase como
#: texto irrelevante.
CHUNK_CHARS = 2000
CHUNK_OVERLAP = 200

#: Texto abaixo disto não vira embedding. Seções só com cabeçalho ("Editais",
#: 7 chars) injetariam ruído e competiriam com os acertos reais.
MIN_TEXT_CHARS = 40

EMBED_BATCH = 100
EMBED_DELAY = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Conexão e utilitários
# ─────────────────────────────────────────────────────────────────────────────

def connect():
    conn = psycopg2.connect(**config.DB_CONFIG)
    conn.autocommit = False
    return conn


def chunk_text(texto: str, size: int = CHUNK_CHARS,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Divide texto longo em chunks, quebrando em fronteiras naturais.

    Tenta parágrafo → linha → fim de frase → espaço, nessa ordem, para não
    cortar no meio de uma ideia.
    """
    texto = (texto or "").strip()
    if len(texto) <= size:
        return [texto] if texto else []

    chunks: list[str] = []
    inicio = 0
    while inicio < len(texto):
        fim = min(inicio + size, len(texto))
        if fim < len(texto):
            janela = texto[inicio:fim]
            corte = max(janela.rfind("\n\n"), janela.rfind("\n"),
                        janela.rfind(". "), janela.rfind(" "))
            if corte > size * 0.5:
                fim = inicio + corte + 1
        pedaco = texto[inicio:fim].strip()
        if pedaco:
            chunks.append(pedaco)
        if fim >= len(texto):
            break
        inicio = max(fim - overlap, inicio + 1)
    return chunks


def juntar(*partes: Optional[str], sep: str = "\n") -> str:
    return sep.join(p for p in partes if p)


def rotulo(nome: str, valor: Any) -> Optional[str]:
    """"Nível: Mestrado e Doutorado" — omite a linha inteira se o valor é nulo."""
    if valor in (None, "", [], {}):
        return None
    if isinstance(valor, (list, tuple)):
        valor = ", ".join(str(v) for v in valor if v)
        if not valor:
            return None
    return f"{nome}: {valor}"


def vetor_literal(vetor: Sequence[float]) -> str:
    """Serializa um embedding no literal aceito pelo pgvector."""
    return "[" + ",".join(f"{v:.7g}" for v in vetor) + "]"


def emitir(destino: list[dict], fk_col: str, fk_val: Any, escopo: str,
           titulo: str, url: Optional[str], texto: Optional[str],
           metadata: Optional[dict] = None) -> None:
    """
    Acrescenta um (ou mais) registros de embedding, fatiando o texto se preciso.

    Quando o texto excede `CHUNK_CHARS`, cada pedaço vira um escopo próprio
    ('corpo#0', 'corpo#1'…), preservando a unicidade (entidade, escopo) que o
    UNIQUE das tabelas emb_* exige.

    O `#` não é decorativo: `busca_semantica.buscar(escopos=[...])` filtra por
    `split_part(escopo, '#', 1)`, então tudo à direita do primeiro `#` é
    discriminador interno. Por isso as seções de página usam `secao#3` e não
    `secao3` — com o segundo, cada seção viraria um escopo distinto e filtrar
    por 'secao' não pegaria nenhuma. O CABEÇALHO é repetido em todos os
    pedaços: sem isso, o chunk 3 de uma notícia chegaria ao prompt sem título
    nem data, e a síntese não teria como citar a fonte.
    """
    texto = (texto or "").strip()
    if len(texto) < MIN_TEXT_CHARS:
        return
    partes = chunk_text(texto)
    if len(partes) == 1:
        destino.append({fk_col: fk_val, "escopo": escopo, "titulo": titulo,
                        "url": url, "texto": partes[0],
                        "metadata": metadata or {}})
        return
    cabecalho = texto.split("\n\n", 1)[0][:400]
    for i, parte in enumerate(partes):
        corpo = parte if i == 0 else f"{cabecalho}\n(continuação {i + 1})\n{parte}"
        destino.append({fk_col: fk_val, "escopo": f"{escopo}#{i}",
                        "titulo": titulo, "url": url, "texto": corpo,
                        "metadata": {**(metadata or {}), "chunk": i}})


# ─────────────────────────────────────────────────────────────────────────────
# Camada relacional
# ─────────────────────────────────────────────────────────────────────────────

def ajustar_dims(sql: str) -> str:
    """
    O DDL foi escrito para o nemotron-3-embed-1b (2048 dims). Quando o provedor
    de embeddings é outro (ex.: BAAI/bge-m3 local, 1024 dims — config.EMBEDDING_DIMS),
    troca a dimensão em `vector(N)` e `halfvec(N)` para que a coluna e o índice
    HNSW batam com o vetor que providers.get_embeddings() realmente produz.
    """
    dims = int(config.EMBEDDING_DIMS)
    if dims != 2048:
        log.info("[Schema] dimensão dos embeddings: 2048 → %d (config.EMBEDDING_DIMS)", dims)
        sql = re.sub(r"\b(vector|halfvec)\(2048\)", rf"\g<1>({dims})", sql)
    return sql


def aplicar_schema(conn, arquivo: Path) -> None:
    if not arquivo.exists():
        raise FileNotFoundError(f"schema não encontrado: {arquivo}")
    log.info("[Schema] aplicando %s", arquivo.name)
    with conn.cursor() as cur:
        cur.execute(ajustar_dims(arquivo.read_text(encoding="utf-8")))
    conn.commit()
    log.info("[Schema] aplicado")


def _colunas(rows: list[dict]) -> list[str]:
    """União das chaves de todas as linhas, preservando a ordem de aparição."""
    cols: list[str] = []
    for row in rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    return cols


def truncar(conn, tabelas: Iterable[str]) -> None:
    """
    Esvazia as tabelas e REINICIA as sequences.

    `RESTART IDENTITY` não é cosmético: sem ele, as chaves BIGSERIAL
    (`faq_id`, `disciplina_id`) continuam de onde pararam a cada recarga e
    crescem indefinidamente. Com ele, uma recarga do mesmo JSON produz
    exatamente os mesmos ids — o que torna o dataset comparável entre
    execuções e as URLs de citação estáveis.
    """
    alvos = list(tabelas)
    if not alvos:
        return
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {', '.join(alvos)} RESTART IDENTITY CASCADE")
    conn.commit()
    log.info("[Truncate] %d tabelas esvaziadas (CASCADE atinge as emb_*)",
             len(alvos))


def mapear_ids(conn, tabela: str, id_col: str,
               chave_cols: Sequence[str]) -> dict[tuple, Any]:
    """
    Lê de volta `{chave_natural: id_gerado}` de uma tabela com BIGSERIAL.

    Necessário porque as tabelas derivadas (`port_faq`, `ppgc_faq`,
    `ppgc_disciplina`) não têm id no JSON do crawler — a chave é natural
    (pagina_id + ordem) e o id é gerado pelo banco. As tabelas emb_*
    referenciam o id gerado, então ele precisa vir do banco, e não de uma
    suposição sobre a ordem de inserção ou sobre o estado da sequence.
    """
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(chave_cols)}, {id_col} FROM {tabela}")
        return {tuple(linha[:-1]): linha[-1] for linha in cur.fetchall()}


def inserir_tabela(conn, tabela: str, rows: list[dict],
                   upsert_em: Optional[Sequence[str]] = None,
                   nao_sobrescrever: Sequence[str] = ()) -> int:
    """
    INSERT em lote. Com `upsert_em`, faz UPSERT em vez de inserir e ignorar.

    `nao_sobrescrever` lista colunas que o UPSERT deve DEIXAR COMO ESTÃO. É o
    que preserva o texto extraído dos PDFs: o crawl atualiza título, mime e
    origem do documento sem tocar em `texto_extraido`, `n_paginas`, `sha256`
    e `extraido_em`.
    """
    if not rows:
        return 0
    cols = _colunas(rows)
    valores = [tuple(row.get(c) for c in cols) for row in rows]
    if upsert_em:
        atualizaveis = [c for c in cols
                        if c not in upsert_em and c not in nao_sobrescrever]
        acao = (f"UPDATE SET " + ", ".join(f"{c} = EXCLUDED.{c}"
                                           for c in atualizaveis)
                if atualizaveis else "NOTHING")
        sql = (f"INSERT INTO {tabela} ({', '.join(cols)}) VALUES %s "
               f"ON CONFLICT ({', '.join(upsert_em)}) DO {acao}")
    else:
        sql = (f"INSERT INTO {tabela} ({', '.join(cols)}) VALUES %s "
               f"ON CONFLICT DO NOTHING")
    with conn.cursor() as cur:
        execute_values(cur, sql, valores, page_size=500)
    return len(rows)


def carregar_relacional(conn, dados: dict, ordem: Sequence[str], *,
                        truncate: bool = True,
                        preservar: dict[str, dict] = None) -> dict[str, int]:
    """
    Carrega as tabelas relacionais na ordem de dependência.

    `preservar` mapeia tabela → {"upsert_em": [...], "nao_sobrescrever": [...]}.
    Tabelas listadas aí não entram no TRUNCATE e são carregadas por UPSERT.
    """
    preservar = preservar or {}
    tabelas = list(ordem)
    if truncate:
        # ordem inversa de dependência; as preservadas ficam de fora
        truncar(conn, [t for t in reversed(tabelas) if t not in preservar])

    resumo: dict[str, int] = {}
    for tabela in tabelas:
        rows = dados.get(tabela) or []
        cfg = preservar.get(tabela) or {}
        try:
            n = inserir_tabela(conn, tabela, rows,
                               upsert_em=cfg.get("upsert_em"),
                               nao_sobrescrever=cfg.get("nao_sobrescrever", ()))
            conn.commit()
        except psycopg2.Error as exc:
            conn.rollback()
            log.error("[%s] falha na carga: %s", tabela, str(exc).splitlines()[0])
            raise
        resumo[tabela] = n
        if n:
            log.info("[%-26s] %6d linhas%s", tabela, n,
                     "  (upsert, preserva extração)" if cfg else "")
    return resumo


# ─────────────────────────────────────────────────────────────────────────────
# Camada vetorial
# ─────────────────────────────────────────────────────────────────────────────

def carregar_embeddings(conn, registros: dict[str, list[dict]],
                        emb_tables: dict[str, str],
                        batch: int = EMBED_BATCH,
                        delay: float = EMBED_DELAY) -> dict[str, int]:
    """
    Gera embeddings em lote e insere nas tabelas emb_*.

    Falha de um lote não aborta a carga: o lote é registrado e o processo
    segue, para que uma indisponibilidade momentânea da API não custe o
    trabalho todo.
    """
    from providers import get_embeddings                  # noqa: E402

    embeddings = get_embeddings()
    resumo: dict[str, int] = {}

    for tabela, fk_col in emb_tables.items():
        rows = registros.get(tabela) or []
        if not rows:
            resumo[tabela] = 0
            continue

        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {tabela}")
        conn.commit()

        inseridos = 0
        for i in range(0, len(rows), batch):
            lote = rows[i:i + batch]
            try:
                vetores = embeddings.embed_documents([r["texto"] for r in lote])
            except Exception as exc:                       # noqa: BLE001
                log.error("[%s] lote %d-%d falhou: %s",
                          tabela, i, i + len(lote), str(exc)[:120])
                continue

            valores = [
                (r[fk_col], r["escopo"], r["titulo"], r.get("url"), r["texto"],
                 vetor_literal(v), Json(r.get("metadata") or {}))
                for r, v in zip(lote, vetores)
            ]
            sql = (f"INSERT INTO {tabela} "
                   f"({fk_col}, escopo, titulo, url, texto, embedding, metadata) "
                   f"VALUES %s ON CONFLICT ({fk_col}, escopo) DO UPDATE SET "
                   f"texto = EXCLUDED.texto, embedding = EXCLUDED.embedding, "
                   f"metadata = EXCLUDED.metadata, titulo = EXCLUDED.titulo, "
                   f"url = EXCLUDED.url, atualizado_em = now()")
            with conn.cursor() as cur:
                execute_values(cur, sql, valores,
                               template="(%s,%s,%s,%s,%s,%s::vector,%s)",
                               page_size=batch)
            conn.commit()
            inseridos += len(valores)
            log.info("[%-22s] %d/%d vetorizados", tabela, inseridos, len(rows))
            if delay and i + batch < len(rows):
                time.sleep(delay)

        resumo[tabela] = inseridos
    return resumo


# ─────────────────────────────────────────────────────────────────────────────
# Relatórios
# ─────────────────────────────────────────────────────────────────────────────

def relatorio_dry_run(dados: dict, ordem: Sequence[str],
                      registros: dict[str, list[dict]],
                      amostras: int = 2) -> None:
    """Contagens + os textos que seriam vetorizados, sem tocar no banco."""
    print("\n" + "=" * 78)
    print("CAMADA RELACIONAL")
    print("=" * 78)
    total = 0
    for tabela in ordem:
        n = len(dados.get(tabela) or [])
        total += n
        print(f"  {tabela:<30} {n:>7}")
    print(f"  {'TOTAL':<30} {total:>7}")

    print("\n" + "=" * 78)
    print("CAMADA VETORIAL — amostra do texto que seria enviado ao embedder")
    print("=" * 78)
    for tabela, rows in registros.items():
        chars = sum(len(r["texto"]) for r in rows)
        print(f"\n── {tabela}  ({len(rows)} trechos, {chars:,} caracteres)"
              .replace(",", "."))
        for r in rows[:amostras]:
            print(f"   escopo={r['escopo']}  url={r.get('url')}")
            print("   " + "-" * 70)
            for linha in r["texto"].splitlines()[:14]:
                print("   | " + linha[:100])
            if len(r["texto"].splitlines()) > 14:
                print("   | …")
    print()


def carregar_json(caminho: Path) -> dict:
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    if not isinstance(dados, dict):
        raise ValueError(f"{caminho}: esperado um objeto {{tabela: [linhas]}}")
    return dados
