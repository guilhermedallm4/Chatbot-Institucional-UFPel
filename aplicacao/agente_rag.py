"""
Agente RAG institucional — LFM2.5 + PostgreSQL (relacional + pgvector)
=============================================================================
Conceito — Módulo Agente

O banco tem DUAS camadas sobre o mesmo crawl (ver crawler/README_computacao.md
e crawler/README_portal_ppgc.md):

  relacional  curso, disciplina, servidor, projeto, turma, port_*, ppgc_* e as
              views vw_* (superfície plana para o LLM escrever SQL)
  vetorial    tabelas emb_* — um embedding por FACETA da entidade (escopo),
              SEMPRE com a chave da entidade de volta (curso_codigo, servidor_id…)

É essa chave que permite a MESCLA: um acerto semântico vira JOIN relacional
(buscar_semantica → chave → detalhar / consultar_sql) e um filtro relacional
restringe a busca vetorial (metadata / WHERE chave IN (...)).

Ferramentas que o LFM2.5 pode chamar:
  buscar_por_nome     nome próprio conhecido → trigram (pg_trgm + unaccent)
  buscar_semantica    tema/assunto/procedimento → ANN nas emb_* (HNSW halfvec)
  consultar_sql       lista, contagem, filtro por data, "todos os X" → SELECT nas vw_*
  detalhar            chave → ficha completa da entidade (todas as tabelas ligadas)
  descrever_tabela    colunas de uma tabela/view quando o esquema do prompt não basta
  ler_pagina          página oficial *.ufpel.edu.br (verificar/atualizar)
  buscar_web          internet (opcional, --web)

Uso:
  python agente_rag.py                          # chat interativo
  python agente_rag.py -p "Quem coordena o curso de Ciência da Computação?"
  python agente_rag.py -p "..." --json          # resposta + rastro em JSON
  python agente_rag.py --pensar                 # liga o raciocínio <think> (--mostrar-pensamento exibe)
  python agente_rag.py --web                    # habilita busca na internet
Avaliação em lote: veja avaliar_agente.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras

import config  # carrega .env
from agente_lfm import AgenteLFM, Tool

for _n in ("httpx", "httpcore", "sentence_transformers", "urllib3", "transformers", "huggingface_hub"):
    logging.getLogger(_n).setLevel(logging.WARNING)

# =============================================================================
# Conexões
# =============================================================================

_RO_CONFIG = {
    **config.DB_CONFIG,
    "user": os.getenv("DB_READONLY_USER") or config.DB_CONFIG["user"],
    "password": os.getenv("DB_READONLY_PASS") or config.DB_CONFIG["password"],
}


def _conn():
    c = psycopg2.connect(**_RO_CONFIG)
    c.autocommit = True
    return c


def _rows(sql: str, params: Any = None) -> list[dict]:
    with _conn() as c, c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET statement_timeout = '20s'")
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


_embeddings = None


def _emb():
    global _embeddings
    if _embeddings is None:
        from providers import get_embeddings
        _embeddings = get_embeddings()
    return _embeddings


DIMS = int(config.EMBEDDING_DIMS)
MAX_TOOL_CHARS = 4500


def _trunc(s: Any, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + " [...]"


def _fmt_rows(rows: list[dict], max_cell: int = 300, max_chars: int = MAX_TOOL_CHARS) -> str:
    if not rows:
        return "0 linhas."
    cols = list(rows[0].keys())
    linhas = [" | ".join(cols)]
    for r in rows:
        linhas.append(" | ".join(
            _trunc(json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else v, max_cell)
            for v in r.values()))
    return _trunc(f"{len(rows)} linha(s):\n" + "\n".join(linhas), max_chars)


def _fmt_secoes(secoes: list[tuple[str, Any]], max_chars: int = MAX_TOOL_CHARS) -> str:
    partes = []
    for titulo, conteudo in secoes:
        if conteudo in (None, "", [], {}):
            continue
        if isinstance(conteudo, list):
            if conteudo and isinstance(conteudo[0], dict):
                corpo = "\n".join("  - " + "; ".join(f"{k}={_trunc(v, 200)}" for k, v in r.items() if v not in (None, ""))
                                  for r in conteudo)
            else:
                corpo = "\n".join(f"  - {_trunc(x, 300)}" for x in conteudo)
        elif isinstance(conteudo, dict):
            corpo = "\n".join(f"  {k}: {_trunc(v, 500)}" for k, v in conteudo.items() if v not in (None, ""))
        else:
            corpo = _trunc(conteudo, 1500)
        partes.append(f"## {titulo}\n{corpo}")
    return _trunc("\n".join(partes), max_chars) or "Sem dados."


# =============================================================================
# Fontes vetoriais (emb_*) e entidades relacionais
# =============================================================================
# nome curto → (tabela emb, coluna-chave, acervo, entidade para detalhar)
FONTES: dict[str, tuple[str, str, str, str]] = {
    "curso":            ("emb_curso",               "curso_codigo",      "institucional", "curso"),
    "disciplina":       ("emb_disciplina",          "disciplina_codigo", "institucional", "disciplina"),
    "servidor":         ("emb_servidor",            "servidor_id",       "institucional", "servidor"),
    "projeto":          ("emb_projeto",             "projeto_id",        "institucional", "projeto"),
    "turma":            ("emb_turma",               "turma_id",          "institucional", "turma"),
    "portal_noticia":   ("emb_port_post",           "post_id",           "portal",        "portal_noticia"),
    "portal_pagina":    ("emb_port_pagina",         "pagina_id",         "portal",        "portal_pagina"),
    "portal_faq":       ("emb_port_faq",            "faq_id",            "portal",        "portal_faq"),
    "ppgc_pagina":      ("emb_ppgc_pagina",         "pagina_id",         "ppgc",          "ppgc_pagina"),
    "ppgc_noticia":     ("emb_ppgc_post",           "post_id",           "ppgc",          "ppgc_noticia"),
    "ppgc_edital":      ("emb_ppgc_edital",         "edital_id",         "ppgc",          "ppgc_edital"),
    "ppgc_normativo":   ("emb_ppgc_normativo",      "normativo_id",      "ppgc",          "ppgc_normativo"),
    "ppgc_faq":         ("emb_ppgc_faq",            "faq_id",            "ppgc",          "ppgc_faq"),
    "ppgc_linha":       ("emb_ppgc_linha_pesquisa", "linha_slug",        "ppgc",          "ppgc_linha"),
    "ppgc_disciplina":  ("emb_ppgc_disciplina",     "disciplina_id",     "ppgc",          "ppgc_disciplina"),
    "ppgc_documento":   ("emb_ppgc_documento",      "documento_url",     "ppgc",          "ppgc_documento"),
}
ACERVOS = ("institucional", "portal", "ppgc")


def _resolver_fontes(fonte: str) -> list[str]:
    f = (fonte or "").strip().lower()
    if not f or f == "todas":
        return list(FONTES)
    if f in ACERVOS:
        return [k for k, v in FONTES.items() if v[2] == f]
    if f in FONTES:
        return [f]
    # tolera nome da tabela emb_*
    for k, v in FONTES.items():
        if v[0] == f or v[0] == f"emb_{f}":
            return [k]
    return []


# =============================================================================
# Ferramentas
# =============================================================================

def buscar_semantica(consulta: str, fonte: str = "", top_k: int = 6) -> str:
    """Busca vetorial (pgvector) por TEMA/assunto/procedimento, quando não há um nome exato: 'quem pesquisa visão computacional?', 'como peço segunda chamada?', 'disciplinas sobre paralelismo'. fonte restringe: um acervo ('institucional' | 'portal' | 'ppgc') ou uma fonte ('curso','disciplina','servidor','projeto','turma','portal_noticia','portal_pagina','portal_faq','ppgc_pagina','ppgc_noticia','ppgc_edital','ppgc_normativo','ppgc_faq','ppgc_linha','ppgc_disciplina'); '' = todas. Cada resultado traz a CHAVE da entidade — use em detalhar() ou num WHERE do SQL (busca híbrida)."""
    top_k = max(1, min(int(top_k), 12))
    fontes = _resolver_fontes(fonte)
    if not fontes:
        return f"Fonte '{fonte}' desconhecida. Use um acervo ({', '.join(ACERVOS)}) ou uma fonte: {', '.join(FONTES)}."
    vec = _emb().embed_query(consulta)
    vec_lit = "[" + ",".join(f"{v:.6f}" for v in vec) + "]"
    dist = f"embedding::halfvec({DIMS}) <=> %(q)s::halfvec({DIMS})"  # mesma expressão do índice HNSW
    subs = []
    for f in fontes:
        tabela, fk, acervo, _ = FONTES[f]
        subs.append(
            f"(SELECT '{f}' AS fonte, {fk}::text AS chave, escopo, titulo, url, texto, metadata, "
            f"1 - ({dist}) AS score FROM {tabela} ORDER BY {dist} LIMIT %(k)s)"
        )
    # subselect externo: com uma única fonte, "(SELECT ... ORDER BY) ORDER BY" seria inválido no PostgreSQL
    sql = "SELECT * FROM (" + " UNION ALL ".join(subs) + ") u ORDER BY score DESC LIMIT %(k)s"
    try:
        rows = _rows(sql, {"q": vec_lit, "k": top_k})
    except Exception as e:  # noqa: BLE001
        return f"Erro na busca semântica: {str(e).splitlines()[0]}"
    if not rows:
        return "Nenhum trecho encontrado."
    out = []
    for i, r in enumerate(rows, 1):
        meta = {k: v for k, v in (r["metadata"] or {}).items() if k in ("vinculo_ativo", "vigente", "nivel", "tipo", "ano", "cursos", "data_publicacao", "acervo")}
        out.append(
            f"[{i}] fonte={r['fonte']} chave={r['chave']} escopo={r['escopo']} score={r['score']:.2f}\n"
            f"{r['titulo']}\nURL: {r['url'] or '-'}"
            + (f"\nmeta: {json.dumps(meta, ensure_ascii=False, default=str)}" if meta else "")
            + f"\n{_trunc(r['texto'].replace(chr(10), ' '), 600)}\n"
        )
    return _trunc("\n".join(out), MAX_TOOL_CHARS)


# entidade → (tabela, coluna-chave, coluna-nome, coluna-url)
_NOMES: dict[str, tuple[str, str, str, str]] = {
    "curso":          ("curso",              "codigo_ufpel", "nome",   "url"),
    "disciplina":     ("disciplina",         "codigo",       "nome",   "url"),
    "servidor":       ("servidor",           "servidor_id",  "nome",   "url"),
    "projeto":        ("projeto",            "projeto_id",   "titulo", "url"),
    "ppgc_edital":    ("ppgc_edital",        "edital_id",    "titulo", "url"),
    "ppgc_normativo": ("ppgc_normativo",     "normativo_id", "titulo", "url_pagina"),
    "ppgc_linha":     ("ppgc_linha_pesquisa", "slug",        "nome",   "url"),
    "ppgc_disciplina": ("ppgc_disciplina",   "disciplina_id", "nome",  "url"),
    "portal_noticia": ("port_post",          "post_id",      "titulo", "url"),
    "ppgc_noticia":   ("ppgc_post",          "post_id",      "titulo", "url"),
    "portal_pagina":  ("port_pagina",        "pagina_id",    "titulo", "url"),
    "ppgc_pagina":    ("ppgc_pagina",        "pagina_id",    "titulo", "url"),
    "grupo_pesquisa": ("port_grupo_pesquisa", "grupo_id",    "nome",   "url"),
}


def buscar_por_nome(nome: str, entidade: str = "", top_k: int = 6) -> str:
    """Localiza registros pelo NOME/TÍTULO com tolerância a erros (trigram): pessoa, disciplina, curso, projeto, edital, norma, notícia, página. entidade: 'curso','disciplina','servidor','projeto','ppgc_edital','ppgc_normativo','ppgc_linha','ppgc_disciplina','portal_noticia','ppgc_noticia','portal_pagina','ppgc_pagina' ou '' (todas). Retorna entidade, CHAVE (para detalhar/SQL), nome e URL."""
    top_k = max(1, min(int(top_k), 12))
    entidade = (entidade or "").strip().lower()
    acervo_map = {  # tolera o nome do acervo no lugar da entidade
        "institucional": ["curso", "disciplina", "servidor", "projeto"],
        "portal": [e for e in _NOMES if e.startswith("portal_") or e == "grupo_pesquisa"],
        "ppgc": [e for e in _NOMES if e.startswith("ppgc_")],
    }
    if entidade in _NOMES:
        alvos = [entidade]
    elif entidade in acervo_map:
        alvos = acervo_map[entidade]
    elif not entidade or entidade == "todas":
        alvos = list(_NOMES)
    else:
        return f"Entidade '{entidade}' desconhecida. Use: {', '.join(_NOMES)} ou um acervo ({', '.join(acervo_map)})."
    subs = []
    for ent in alvos:
        tab, chave, col_nome, col_url = _NOMES[ent]
        subs.append(
            f"(SELECT '{ent}' AS entidade, {chave}::text AS chave, {col_nome} AS nome, {col_url} AS url, "
            f"similarity(unaccent({col_nome}), unaccent(%(n)s)) AS sim FROM {tab} "
            f"WHERE unaccent({col_nome}) %% unaccent(%(n)s) OR unaccent({col_nome}) ILIKE '%%' || unaccent(%(n)s) || '%%' "
            f"ORDER BY sim DESC LIMIT %(k)s)"
        )
    sql = "SELECT * FROM (" + " UNION ALL ".join(subs) + ") u ORDER BY sim DESC LIMIT %(k)s"
    try:
        with _conn() as c, c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET pg_trgm.similarity_threshold = 0.15")
            cur.execute(sql, {"n": nome, "k": top_k})
            rows = cur.fetchall()
    except Exception as e:  # noqa: BLE001
        return f"Erro na busca por nome: {str(e).splitlines()[0]}"
    if not rows:
        return f"Nada parecido com '{nome}'. Tente buscar_semantica ou outra grafia."
    return "\n".join(f"[{i}] {r['entidade']} chave={r['chave']} sim={r['sim']:.2f} | {r['nome']} | URL: {r['url'] or '-'}"
                     for i, r in enumerate(rows, 1))


def _detalhar_curso(k: str) -> str:
    c = _rows("SELECT * FROM curso WHERE codigo_ufpel = %s", (k,))
    if not c:
        return f"curso '{k}' não encontrado."
    return _fmt_secoes([
        ("Curso", c[0]),
        ("Conceitos", _rows("SELECT indicador, ano, nota FROM curso_conceito WHERE curso_codigo=%s", (k,))),
        ("Versões de currículo", _rows("SELECT versao, is_atual FROM curriculo_versao WHERE curso_codigo=%s ORDER BY versao", (k,))),
        ("Matriz (versão atual) por semestre", _rows(
            "SELECT semestre_rotulo, COUNT(*) AS n_disciplinas, STRING_AGG(disciplina_nome, '; ' ORDER BY disciplina_nome) AS disciplinas "
            "FROM vw_disciplina_curso WHERE curso_codigo=%s AND versao_atual GROUP BY semestre_rotulo, semestre_num ORDER BY semestre_num NULLS LAST", (k,))),
        ("Vagas por processo", _rows("SELECT processo, ano, semestre, SUM(vagas) AS vagas FROM curso_vaga WHERE curso_codigo=%s AND cota<>'TOTAL' GROUP BY 1,2,3 ORDER BY 2,3", (k,))),
        ("Professores com vínculo ativo", _rows(
            "SELECT nome, titulacao, array_to_string(areas_atuacao,'; ') AS areas FROM vw_professor_computacao "
            "WHERE %s = ANY(cursos_codigos) AND vinculo_ativo ORDER BY nome LIMIT 60", (k,))),
        ("Seções da aba Informações (use consultar_sql em curso_info_secao para o texto)",
         _rows("SELECT secao, LEFT(texto, 200) AS inicio FROM curso_info_secao WHERE curso_codigo=%s ORDER BY ordem", (k,))),
    ])


def _detalhar_disciplina(k: str) -> str:
    d = _rows("SELECT * FROM disciplina WHERE codigo = %s", (k,))
    if not d:
        return f"disciplina '{k}' não encontrada."
    return _fmt_secoes([
        ("Disciplina", d[0]),
        ("Conteúdo", {r["secao"]: r["texto"] for r in _rows("SELECT secao, texto FROM disciplina_conteudo WHERE disciplina_codigo=%s ORDER BY ordem", (k,))}),
        ("Cursos/matriz em que aparece", _rows("SELECT curso_nome, versao, versao_atual, semestre_rotulo, carater, creditos, horas FROM vw_disciplina_curso WHERE disciplina_codigo=%s ORDER BY curso_nome, versao", (k,))),
        ("Pré-requisitos (por curso/versão)", _rows("SELECT curso_codigo, versao, prereq_nome FROM matriz_prerequisito WHERE disciplina_codigo=%s", (k,))),
        ("Turmas ofertadas", _rows("SELECT ano, semestre, codigo_turma, vagas, matriculados, array_to_string(professores,', ') AS professores, horarios, array_to_string(cursos,', ') AS cursos FROM vw_turma WHERE disciplina_codigo=%s ORDER BY ano DESC, semestre DESC", (k,))),
        ("Equivalências", _rows("SELECT equivalente_nome, curso_nome FROM disciplina_equivalencia WHERE disciplina_codigo=%s LIMIT 15", (k,))),
        ("Bibliografia básica", [r["referencia"] for r in _rows("SELECT referencia FROM disciplina_bibliografia WHERE disciplina_codigo=%s AND tipo ILIKE 'b%%' ORDER BY ordem LIMIT 6", (k,))]),
    ])


def _detalhar_servidor(k: str) -> str:
    s = _rows("SELECT servidor_id, nome, cargo, categoria, titulacao, lotacao_nome, regime_jornada, situacao, vinculo_ativo, "
              "data_ingresso_ufpel, data_saida_cargo, email, lattes_url, url, LEFT(curriculo_resumo, 900) AS curriculo_resumo FROM servidor WHERE servidor_id=%s", (k,))
    if not s:
        return f"servidor '{k}' não encontrado."
    return _fmt_secoes([
        ("Servidor", s[0]),
        ("Funções", _rows("SELECT funcao, unidade, data_inicio FROM servidor_funcao WHERE servidor_id=%s", (k,))),
        ("Formação", _rows("SELECT nivel, area, instituicao, ano FROM servidor_formacao WHERE servidor_id=%s ORDER BY ordem", (k,))),
        ("Áreas de atuação", [r["area"] for r in _rows("SELECT area FROM servidor_area_atuacao WHERE servidor_id=%s ORDER BY ordem", (k,))]),
        ("Cursos de Computação em que atua", _rows("SELECT c.nome, c.nivel, sc.origem FROM servidor_curso sc JOIN curso c ON c.codigo_ufpel=sc.curso_codigo WHERE sc.servidor_id=%s", (k,))),
        ("Projetos vigentes", _rows("SELECT p.titulo, sp.papel, p.enfase, p.data_inicio, p.data_fim FROM servidor_projeto sp JOIN projeto p ON p.projeto_id=sp.projeto_id WHERE sp.servidor_id=%s ORDER BY p.data_inicio DESC LIMIT 20", (k,))),
        ("Disciplinas ministradas (3 últimos semestres)", _rows("SELECT ano, semestre, codigo_turma, d.nome AS disciplina, curso_nome FROM servidor_disciplina_ministrada m JOIN disciplina d ON d.codigo=m.disciplina_codigo WHERE m.servidor_id=%s ORDER BY ano DESC, semestre DESC LIMIT 25", (k,))),
    ])


def _detalhar_projeto(k: str) -> str:
    p = _rows("SELECT * FROM projeto WHERE projeto_id=%s", (k,))
    if not p:
        return f"projeto '{k}' não encontrado."
    return _fmt_secoes([
        ("Projeto", p[0]),
        ("Seções", {r["secao"]: _trunc(r["texto"], 800) for r in _rows("SELECT secao, texto FROM projeto_info_secao WHERE projeto_id=%s ORDER BY ordem", (k,))}),
        ("Professores envolvidos", _rows("SELECT s.nome, sp.papel, s.lotacao_nome FROM servidor_projeto sp JOIN servidor s ON s.servidor_id=sp.servidor_id WHERE sp.projeto_id=%s", (k,))),
        ("Equipe (total)", _rows("SELECT COUNT(*) AS membros, COUNT(*) FILTER (WHERE is_servidor) AS servidores FROM projeto_equipe WHERE projeto_id=%s", (k,))),
    ])


def _detalhar_simples(sql: str, k: Any, rotulo: str, extra: Optional[list[tuple[str, Any]]] = None) -> str:
    r = _rows(sql, (k,))
    if not r:
        return f"{rotulo} '{k}' não encontrado."
    return _fmt_secoes([(rotulo, r[0])] + (extra or []))


DETALHADORES = {
    "curso": _detalhar_curso,
    "disciplina": _detalhar_disciplina,
    "servidor": _detalhar_servidor,
    "projeto": _detalhar_projeto,
    "turma": lambda k: _detalhar_simples("SELECT * FROM vw_turma WHERE turma_id=%s", k, "Turma"),
    "ppgc_edital": lambda k: _detalhar_simples(
        "SELECT * FROM vw_ppgc_edital WHERE edital_id=%s", k, "Edital",
        [("Documentos do edital", _rows("SELECT tipo_documento, titulo, secao, url FROM ppgc_edital_documento WHERE edital_id=%s ORDER BY ordem", (k,)))]),
    "ppgc_normativo": lambda k: _detalhar_simples(
        "SELECT normativo_id, tipo, numero, ano, titulo, ementa, vigente, url_pagina, url_pdf, LEFT(texto, 2500) AS texto FROM ppgc_normativo WHERE normativo_id=%s", k, "Norma"),
    "ppgc_linha": lambda k: _detalhar_simples("SELECT * FROM ppgc_linha_pesquisa WHERE slug=%s", k, "Linha de pesquisa"),
    "ppgc_disciplina": lambda k: _detalhar_simples("SELECT * FROM ppgc_disciplina WHERE disciplina_id=%s", int(k), "Disciplina PPGC"),
    "portal_noticia": lambda k: _detalhar_simples("SELECT titulo, url, data_por_extenso, secao_slug, escopo_curso, LEFT(texto, 3000) AS texto FROM port_post WHERE post_id=%s", int(k), "Notícia (portal)"),
    "ppgc_noticia": lambda k: _detalhar_simples("SELECT titulo, url, data_por_extenso, assunto, LEFT(texto, 3000) AS texto FROM ppgc_post WHERE post_id=%s", int(k), "Notícia (PPGC)"),
    "portal_pagina": lambda k: _detalhar_simples(
        "SELECT titulo, url, caminho, secao_slug, idioma, resumo FROM port_pagina WHERE pagina_id=%s", int(k), "Página (portal)",
        [("Seções", _rows("SELECT titulo_secao, LEFT(texto, 500) AS texto, url FROM port_pagina_secao WHERE pagina_id=%s ORDER BY ordem", (int(k),)))]),
    "ppgc_pagina": lambda k: _detalhar_simples(
        "SELECT titulo, url, caminho, categoria, resumo FROM ppgc_pagina WHERE pagina_id=%s", int(k), "Página (PPGC)",
        [("Seções", _rows("SELECT titulo_secao, LEFT(texto, 500) AS texto, url FROM ppgc_pagina_secao WHERE pagina_id=%s ORDER BY ordem", (int(k),)))]),
    "portal_faq": lambda k: _detalhar_simples("SELECT * FROM vw_port_faq WHERE faq_id=%s", int(k), "FAQ (portal)"),
    "ppgc_faq": lambda k: _detalhar_simples("SELECT faq_id, secao, pergunta, resposta, url FROM ppgc_faq WHERE faq_id=%s", int(k), "FAQ (PPGC)"),
    "grupo_pesquisa": lambda k: _detalhar_simples("SELECT * FROM port_grupo_pesquisa WHERE grupo_id=%s", int(k), "Grupo de pesquisa"),
}


def detalhar(entidade: str, chave: str) -> str:
    """Ficha COMPLETA de uma entidade a partir da chave retornada por buscar_por_nome/buscar_semantica (mescla todas as tabelas ligadas): curso → matriz, vagas, professores; disciplina → ementa, pré-requisitos, turmas, equivalências; servidor → formação, áreas, cursos, projetos, disciplinas ministradas; projeto → seções e equipe; turma; ppgc_edital → documentos/PDFs; ppgc_normativo; ppgc_linha; ppgc_disciplina; portal_noticia; ppgc_noticia; portal_pagina; ppgc_pagina; portal_faq; ppgc_faq."""
    fn = DETALHADORES.get((entidade or "").strip().lower())
    if fn is None:
        return f"Entidade '{entidade}' desconhecida. Use: {', '.join(DETALHADORES)}."
    try:
        return fn(str(chave).strip())
    except Exception as e:  # noqa: BLE001
        return f"Erro ao detalhar {entidade} '{chave}': {str(e).splitlines()[0]}"


_SQL_PROIBIDO = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|vacuum|call|do|execute|"
    r"pg_sleep|pg_read_file|pg_ls_dir|lo_import|lo_export|dblink|set\s+role|reset)\b", re.I)


def consultar_sql(sql: str) -> str:
    """Executa UM SELECT (somente leitura) e retorna até 40 linhas. Use para listas, contagens, filtros exatos, datas, ordenações e para a busca HÍBRIDA (WHERE chave IN (...) com chaves vindas de buscar_semantica). Escreva contra as views vw_* do esquema do prompt; nomes de pessoas/disciplinas estão em MAIÚSCULAS → unaccent(col) ILIKE unaccent('%termo%'). Sempre inclua LIMIT."""
    s = sql.strip().rstrip(";").strip()
    sem_literais = re.sub(r"'(?:[^']|'')*'", "''", s)  # ignora ';' e palavras dentro de strings ('; ', 'drop')
    if ";" in sem_literais:
        return "Erro: envie apenas UMA instrução SQL, sem ';' no meio."
    if not re.match(r"^(select|with)\b", s, re.I):
        return "Erro: apenas consultas SELECT/WITH são permitidas."
    if _SQL_PROIBIDO.search(sem_literais):
        return "Erro: a consulta contém comando não permitido (somente leitura)."
    if not re.search(r"\blimit\s+\d+", s, re.I):
        s += " LIMIT 40"
    try:
        with _conn() as c, c.cursor() as cur:
            cur.execute("SET statement_timeout = '20s'")
            cur.execute("SET default_transaction_read_only = on")  # 2ª barreira além do papel rag_leitor
            cur.execute(s)
            cols = [d.name for d in cur.description]
            rows = cur.fetchmany(40)
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0]
        dica = "Dica: confira colunas com descrever_tabela('<view>') e use ILIKE/unaccent para textos."
        m_col = re.search(r'column "?([\w.]+)"? does not exist', msg)
        if m_col:
            dica = _sugerir_coluna(m_col.group(1).split(".")[-1], s)
        return f"Erro SQL: {msg}\n{dica}"
    if not rows:
        return "Consulta executada: 0 linhas."
    return _fmt_rows([dict(zip(cols, r)) for r in rows])


def _sugerir_coluna(coluna: str, sql: str) -> str:
    """Para 'column X does not exist': lista as colunas mais parecidas nas tabelas/views citadas no SQL."""
    tabelas = set(t.lower() for t in re.findall(r"\b(?:from|join)\s+([a-zA-Z_][\w]*)", sql, re.I))
    if not tabelas:
        return "Dica: essa coluna não existe — chame descrever_tabela('<view ou tabela>') para ver as colunas."
    try:
        rows = _rows(
            "SELECT table_name, column_name, similarity(column_name, %(c)s) AS sim FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name = ANY(%(t)s) ORDER BY sim DESC LIMIT 4",
            {"c": coluna, "t": list(tabelas)})
    except Exception:  # noqa: BLE001
        rows = []
    if rows and rows[0]["sim"] > 0.3:
        return ("Dica: a coluna '" + coluna + "' não existe. Você quis dizer: "
                + ", ".join(f"{r['table_name']}.{r['column_name']}" for r in rows if r["sim"] > 0.3)
                + "? Corrija e reenvie a consulta.")
    return f"Dica: a coluna '{coluna}' não existe em {', '.join(sorted(tabelas))} — chame descrever_tabela para ver as colunas."


def descrever_tabela(nome: str) -> str:
    """Lista as colunas (e tipos) de uma tabela ou view do banco, com o total de linhas. Use quando o esquema do prompt não bastar ou um SQL falhar por coluna inexistente."""
    nome = re.sub(r"[^a-z0-9_]", "", (nome or "").lower())
    cols = _rows("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='public' AND table_name=%s ORDER BY ordinal_position", (nome,))
    if not cols:
        parecidas = _rows("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name ILIKE %s ORDER BY 1 LIMIT 15", (f"%{nome[:12]}%",))
        return f"'{nome}' não existe." + (f" Parecidas: {', '.join(r['table_name'] for r in parecidas)}" if parecidas else "")
    try:
        n = _rows(f"SELECT COUNT(*) AS n FROM {nome}")[0]["n"]
    except Exception:  # noqa: BLE001
        n = "?"
    return f"{nome} ({n} linhas): " + ", ".join(f"{c['column_name']} {c['data_type']}" for c in cols)


_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
DOMINIOS_PERMITIDOS = ("ufpel.edu.br",)


def ler_pagina(url: str) -> str:
    """Lê o texto de uma página oficial da UFPel (somente *.ufpel.edu.br, inclui wp.ufpel.edu.br e institucional.ufpel.edu.br). Use para conferir/atualizar um dado a partir da URL retornada pelas outras ferramentas, ou quando o banco não tem o dado e a página oficial provavelmente tem."""
    import requests
    from bs4 import BeautifulSoup

    host = urlparse(url).hostname or ""
    if not any(host == d or host.endswith("." + d) for d in DOMINIOS_PERMITIDOS):
        return f"Erro: só é permitido ler páginas em {', '.join(DOMINIOS_PERMITIDOS)} (recebido: {host})."
    try:
        r = requests.get(url, headers={"User-Agent": _UA}, timeout=20)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return f"Erro ao acessar {url}: {e}"
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    texto = re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n")).strip()
    return _trunc(texto, MAX_TOOL_CHARS) or "Página sem texto extraível."


def buscar_web(consulta: str, max_results: int = 5) -> str:
    """Busca na internet (DuckDuckGo). Use apenas para o que NÃO está na base da UFPel (ex.: definição de um conceito, notícia externa)."""
    from ddgs import DDGS
    try:
        res = DDGS().text(consulta, max_results=max(1, min(int(max_results), 8)), region="br-pt")
    except Exception as e:  # noqa: BLE001
        return f"Erro na busca: {e}"
    if not res:
        return "Nenhum resultado."
    return _trunc("\n".join(f"[{i}] {r.get('title','')}\nURL: {r.get('href','')}\n{r.get('body','')}\n"
                            for i, r in enumerate(res, 1)), MAX_TOOL_CHARS)


# =============================================================================
# Prompt de sistema — esquema (views) lido do banco + regras de negócio
# =============================================================================

REGRAS_NEGOCIO = """\
Regras de negócio (decidem a resposta certa):
- Professores ATUAIS → WHERE vinculo_ativo = TRUE (vw_professor_computacao / servidor). Sem o filtro entram ex-professores
  que só ministraram turmas recentes. Professores "da computação" → cardinality(cursos_codigos) > 0.
- Grade/matriz sem versão explícita → WHERE versao_atual (vw_disciplina_curso). Versões antigas só existem na oferta (vw_turma.versoes).
- Turmas: vw_turma tem UMA linha por turma (cursos/versões agregados); semestre corrente = 2026/2.
- Normas/regimento/resoluções do PPGC → SEMPRE vw_ppgc_normativo_vigente (norma revogada não pode ser citada como regra).
- "Edital aberto/mais recente" → vw_ppgc_edital_mais_recente; tipos de edital: ingresso_regular, aluno_especial, bolsa, outros;
  nivel: mestrado | doutorado | ambos | NULL. Documentos/PDFs de um edital → vw_ppgc_edital_documento (tipo_documento='edital','resultado',...).
- Prazos e requisitos (créditos, proficiência, qualificação) → vw_ppgc_requisito WHERE nivel = 'mestrado' | 'doutorado' (valores diferem!).
- Notícias: portal da Computação = vw_port_noticia; do PPGC = ppgc_post; os dois juntos = vw_computacao_noticia (coluna acervo). data_publicacao é DATE; ano/mes são inteiros.
- Calendário: agenda do PPGC → vw_ppgc_agenda (data_inicio); calendário acadêmico da graduação (2021–2023) → vw_port_calendario.
- Procedimentos ("como faço para…") → buscar_semantica em portal_faq / ppgc_faq. Fato exato ("até quando…", "quantos…") → consultar_sql.
- Qual acervo: dúvidas de aluno de GRADUAÇÃO (prova, segunda chamada, atestado, matrícula, estágio, TCC, monitoria) → portal
  (fonte='portal_faq' ou 'portal'); mestrado/doutorado/bolsa/edital de seleção/defesa → ppgc. Em dúvida, busque sem fonte (todas).
- Vagas de ingresso na graduação (SISU, PAVE, cotas) → tabela curso_vaga (curso_codigo, processo, ano, semestre, cota, vagas) ou detalhar('curso', codigo). Não é assunto do PPGC.
- "Última / mais recente / próximos" → SEMPRE consultar_sql com ORDER BY data (data_publicacao DESC, data_inicio ASC). Busca semântica NÃO sabe o que é recente.
- Contagens ("quantos/quantas") → consultar_sql com COUNT(*) na view certa; nunca conte itens de uma busca semântica.
- Turmas/oferta → vw_turma (colunas ano, semestre, professores[], cursos[], horarios). Professor com mais turmas: unnest(professores).
- Escreva SQL contra as views vw_* listadas abaixo; só use tabela-base se a view não tiver a coluna, e antes chame descrever_tabela.
- Só existem as ferramentas listadas (não chame views nem tabelas como se fossem funções).
- Campos ausentes são NULL (nunca a string 'Não há informações disponíveis'). Códigos de curso: 3900=Ciência da Computação, 3910=Engenharia de Computação, 7057=Mestrado, 8102=Doutorado, 9130=Especialização."""

EXEMPLOS_SQL = """\
Exemplos de SQL:
- professores atuais da computação e suas áreas:
  SELECT nome, titulacao, lotacao_nome, array_to_string(areas_atuacao,'; ') AS areas FROM vw_professor_computacao WHERE cardinality(cursos_codigos)>0 AND vinculo_ativo ORDER BY nome LIMIT 80
- grade do 4º semestre de Ciência da Computação (versão vigente):
  SELECT disciplina_codigo, disciplina_nome, creditos, horas, carater FROM vw_disciplina_curso WHERE curso_codigo='3900' AND versao_atual AND semestre_num=4 ORDER BY disciplina_nome LIMIT 40
- pré-requisitos de uma disciplina no curso: SELECT d.nome, p.prereq_nome FROM matriz_prerequisito p JOIN disciplina d ON d.codigo=p.disciplina_codigo WHERE p.curso_codigo='3900' AND unaccent(d.nome) ILIKE unaccent('%estrutura de dados%') LIMIT 20
- vagas: SELECT processo, ano, semestre, SUM(vagas) FROM curso_vaga WHERE curso_codigo='3900' AND cota<>'TOTAL' GROUP BY 1,2,3 ORDER BY 2,3 LIMIT 20
- turmas de um professor em 2026/2: SELECT disciplina_nome, codigo_turma, horarios, array_to_string(cursos,', ') FROM vw_turma WHERE ano=2026 AND semestre=2 AND EXISTS (SELECT 1 FROM unnest(professores) p WHERE unaccent(p) ILIKE unaccent('%netto%')) LIMIT 20
- quantas turmas em 2026/2: SELECT COUNT(*) FROM vw_turma WHERE ano=2026 AND semestre=2
- professor com mais turmas em 2026/2: SELECT p AS professor, COUNT(*) AS turmas FROM vw_turma, unnest(professores) p WHERE ano=2026 AND semestre=2 GROUP BY p ORDER BY turmas DESC LIMIT 5
- linhas de pesquisa do PPGC: SELECT nome, descricao, url FROM ppgc_linha_pesquisa LIMIT 10
- requisitos/prazos: SELECT requisito, prazo FROM vw_ppgc_requisito WHERE nivel='doutorado' LIMIT 20
- normas vigentes: SELECT tipo, numero, ano, titulo, url FROM vw_ppgc_normativo_vigente ORDER BY ano DESC LIMIT 20
- próximos prazos do PPGC: SELECT data_inicio, titulo FROM vw_ppgc_agenda WHERE data_inicio >= CURRENT_DATE ORDER BY data_inicio LIMIT 10
- edital de ingresso mais recente do mestrado: SELECT titulo, periodo_letivo, numero_oficial, url FROM vw_ppgc_edital_mais_recente WHERE tipo='ingresso_regular' AND nivel IN ('mestrado','ambos') LIMIT 5
- últimas notícias: SELECT acervo, data_por_extenso, titulo, url FROM vw_computacao_noticia ORDER BY data_publicacao DESC LIMIT 10
- híbrido (tema + fato): 1) buscar_semantica('aprendizado de máquina', fonte='projeto') → chaves; 2) SELECT professor_nome, titulo, papel FROM vw_projeto_professor WHERE projeto_id IN ('<chave1>','<chave2>') LIMIT 40"""

SYSTEM_PROMPT_TEMPLATE = """\
Você é o assistente institucional dos cursos de Computação da UFPel (Universidade Federal de Pelotas). Responde em português, \
de forma objetiva, SOMENTE com base nos dados obtidos pelas ferramentas. Data de hoje: {data}.

Três acervos no mesmo banco PostgreSQL (coleta de julho–setembro/2026):
- institucional (institucional.ufpel.edu.br): {n_inst}
- portal da Computação (wp.ufpel.edu.br/computacao): {n_portal}
- PPGC, a pós-graduação (…/computacao/ppgc): {n_ppgc}

Como escolher a ferramenta (SEMPRE consulte pelo menos uma antes de responder sobre a UFPel):
1. Nome próprio conhecido (pessoa, disciplina, curso, projeto, edital, norma)? → buscar_por_nome → detalhar(entidade, chave).
2. Tema, assunto, "algo ligado a…", "como faço para…"? → buscar_semantica (restrinja com fonte quando souber o acervo).
3. Lista, contagem, soma, filtro exato, datas, "todos os X", "os últimos N"? → consultar_sql nas views abaixo.
4. Tema E fato juntos ("professores com projeto de IA e sua titulação")? → HÍBRIDO: buscar_semantica → chaves → consultar_sql/detalhar.
5. Ficha completa de algo já localizado? → detalhar(entidade, chave). Coluna desconhecida? → descrever_tabela.
6. Conferir/atualizar pela página oficial? → ler_pagina(url).
{regra_web}
Regras de resposta:
- Nunca invente nomes, números, datas ou URLs. Se as ferramentas não trouxerem a informação, diga que não foi encontrada na base.
- Ao final, liste as fontes (URLs) dos registros usados.
- Não repita a mesma chamada com os mesmos argumentos; se falhar, mude a estratégia (outra ferramenta, outros termos, outra view).
- Prefira UMA consulta SQL bem feita a várias chamadas de detalhar. Seja conciso.

{regras}

Views para SQL (colunas):
{views}
Tabelas-base (use descrever_tabela para ver colunas): {tabelas}

{exemplos}"""


def _contagem(sql: str) -> str:
    try:
        return str(_rows(sql)[0]["n"])
    except Exception:  # noqa: BLE001
        return "?"


def _views_texto() -> str:
    rows = _rows("SELECT table_name, string_agg(column_name, ', ' ORDER BY ordinal_position) AS cols "
                 "FROM information_schema.columns WHERE table_schema='public' AND table_name LIKE 'vw\\_%' "
                 "AND table_name NOT IN ('vw_turma_completa','vw_port_menu','vw_port_documento','vw_port_pagina') "
                 "GROUP BY table_name ORDER BY table_name")
    return "\n".join(f"- {r['table_name']}({r['cols']})" for r in rows) or "- (nenhuma view encontrada — rode os loaders em crawler/)"


def _tabelas_texto() -> str:
    rows = _rows("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE' "
                 "AND table_name NOT LIKE 'emb\\_%' AND table_name NOT LIKE 'langchain%' AND table_name NOT LIKE '%\\_info' "
                 "AND table_name NOT IN ('doc_completos','disciplinas','projetos','servidores','cursos','unidades','gestao','sobre','portal_geral') ORDER BY 1")
    return ", ".join(r["table_name"] for r in rows)


def montar_system_prompt(web: bool = False) -> str:
    regra_web = ("7. Assunto fora da UFPel (conceitos gerais, notícias externas)? → buscar_web.\n" if web else
                 "Perguntas fora do escopo institucional: responda brevemente que só trata de dados da Computação/UFPel.\n")
    n_inst = (f"{_contagem('SELECT COUNT(*) n FROM curso')} cursos, {_contagem('SELECT COUNT(*) n FROM disciplina')} disciplinas, "
              f"{_contagem('SELECT COUNT(*) n FROM servidor')} servidores ({_contagem('SELECT COUNT(*) n FROM servidor WHERE vinculo_ativo')} com vínculo ativo), "
              f"{_contagem('SELECT COUNT(*) n FROM projeto')} projetos vigentes, {_contagem('SELECT COUNT(*) n FROM turma')} turmas de 2026/2")
    n_portal = (f"{_contagem('SELECT COUNT(*) n FROM port_post')} notícias, {_contagem('SELECT COUNT(*) n FROM port_pagina')} páginas, "
                f"{_contagem('SELECT COUNT(*) n FROM port_faq')} perguntas de FAQ, {_contagem('SELECT COUNT(*) n FROM port_calendario_evento')} eventos de calendário (2021–2023), "
                f"{_contagem('SELECT COUNT(*) n FROM port_grupo_pesquisa')} grupos de pesquisa")
    n_ppgc = (f"{_contagem('SELECT COUNT(*) n FROM ppgc_edital')} editais, {_contagem('SELECT COUNT(*) n FROM ppgc_normativo WHERE vigente')} normas vigentes, "
              f"{_contagem('SELECT COUNT(*) n FROM ppgc_post')} notícias, {_contagem('SELECT COUNT(*) n FROM ppgc_pagina')} páginas, {_contagem('SELECT COUNT(*) n FROM ppgc_faq')} FAQ, "
              f"{_contagem('SELECT COUNT(*) n FROM ppgc_requisito')} requisitos, {_contagem('SELECT COUNT(*) n FROM ppgc_linha_pesquisa')} linhas de pesquisa, "
              f"{_contagem('SELECT COUNT(*) n FROM ppgc_docente')} docentes, {_contagem('SELECT COUNT(*) n FROM ppgc_calendario_evento')} eventos de agenda")
    return SYSTEM_PROMPT_TEMPLATE.format(
        data=datetime.now().strftime("%d/%m/%Y"), n_inst=n_inst, n_portal=n_portal, n_ppgc=n_ppgc,
        regra_web=regra_web, regras=REGRAS_NEGOCIO, views=_views_texto(), tabelas=_tabelas_texto(), exemplos=EXEMPLOS_SQL,
    )


def montar_tools(web: bool = False) -> list[Tool]:
    tools = [
        Tool.from_function(buscar_por_nome, param_docs={
            "nome": "nome ou título (ou parte)", "entidade": "curso | disciplina | servidor | projeto | ppgc_edital | ppgc_normativo | ppgc_linha | ppgc_disciplina | portal_noticia | ppgc_noticia | portal_pagina | ppgc_pagina | '' (todas)",
            "top_k": "quantidade de resultados (1-12)"}),
        Tool.from_function(buscar_semantica, param_docs={
            "consulta": "tema ou pergunta em linguagem natural",
            "fonte": "acervo (institucional | portal | ppgc) ou fonte específica; '' = todas",
            "top_k": "quantidade de resultados (1-12)"}),
        Tool.from_function(consultar_sql, param_docs={"sql": "uma instrução SELECT ... LIMIT n"}),
        Tool.from_function(detalhar, param_docs={
            "entidade": "curso | disciplina | servidor | projeto | turma | ppgc_edital | ppgc_normativo | ppgc_linha | ppgc_disciplina | portal_noticia | ppgc_noticia | portal_pagina | ppgc_pagina | portal_faq | ppgc_faq",
            "chave": "chave retornada por buscar_por_nome/buscar_semantica (codigo, servidor_id, projeto_id, edital_id, post_id...)"}),
        Tool.from_function(descrever_tabela, param_docs={"nome": "nome da tabela ou view"}),
        Tool.from_function(ler_pagina, param_docs={"url": "URL em *.ufpel.edu.br"}),
    ]
    if web:
        tools.append(Tool.from_function(buscar_web, param_docs={"consulta": "termos de busca", "max_results": "1-8"}))
    return tools


def _view_como_ferramenta(chamada) -> Optional[str]:
    """
    Modelos pequenos às vezes chamam uma view como se fosse função:
        vw_ppgc_edital_mais_recente(tipo='ingresso_regular', nivel='mestrado')
    Se o nome for uma tabela/view existente, converte em SELECT * ... WHERE col = valor LIMIT 20.
    """
    nome = re.sub(r"[^a-z0-9_]", "", (chamada.name or "").lower())
    if not nome:
        return None
    cols = {r["column_name"] for r in _rows(
        "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name=%s", (nome,))}
    if not cols:
        return None
    conds, params = [], []
    for k, v in (chamada.arguments or {}).items():
        if k in cols and isinstance(v, (str, int, float)):
            conds.append(f"{k}::text ILIKE %s")
            params.append(str(v))
    where = (" WHERE " + " AND ".join(conds)) if conds else ""
    try:
        rows = _rows(f"SELECT * FROM {nome}{where} LIMIT 20", params or None)
    except Exception as e:  # noqa: BLE001
        return f"'{nome}' é uma view/tabela, não uma ferramenta. Use consultar_sql. Erro ao consultar: {str(e).splitlines()[0]}"
    return (f"[aviso] '{nome}' é uma view/tabela, não uma ferramenta — executei consultar_sql(\"SELECT * FROM {nome}{where} LIMIT 20\") por você. "
            f"Da próxima vez chame consultar_sql diretamente.\n" + _fmt_rows(rows))


def criar_agente(web: bool = False, mostrar_pensamento: bool = False, pensar: bool = False,
                 stream: bool = True, max_rodadas: int = 6, verbose_tools: bool = True, **kw) -> AgenteLFM:
    """Fábrica: agente LFM2.5 já configurado com as ferramentas dos três acervos."""
    agente = _criar_agente_base(web, mostrar_pensamento, pensar, stream, max_rodadas, verbose_tools, **kw)
    agente.on_unknown_tool = _view_como_ferramenta
    return agente


def _criar_agente_base(web, mostrar_pensamento, pensar, stream, max_rodadas, verbose_tools, **kw) -> AgenteLFM:
    return AgenteLFM(
        system_prompt=montar_system_prompt(web),
        tools=montar_tools(web),
        model_id=os.getenv("AGENT_MODEL_ID", kw.pop("model_id", "LiquidAI/LFM2.5-2.6B")),
        max_new_tokens=int(os.getenv("AGENT_MAX_NEW_TOKENS", kw.pop("max_new_tokens", 2048))),
        temperature=float(os.getenv("AGENT_TEMPERATURE", kw.pop("temperature", 0.1))),
        pensar=pensar, mostrar_pensamento=mostrar_pensamento, max_tool_rounds=max_rodadas,
        stream=stream, verbose_tools=verbose_tools,
        device=os.getenv("AGENT_DEVICE", kw.pop("device", "auto")),
        dtype=os.getenv("AGENT_DTYPE", kw.pop("dtype", "auto")),
        exigir_ferramenta=kw.pop("exigir_ferramenta", True), max_calls_per_round=kw.pop("max_calls_per_round", 4),
        max_tool_result_chars=kw.pop("max_tool_result_chars", MAX_TOOL_CHARS), **kw,
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Agente RAG institucional UFPel (LFM2.5 + PostgreSQL relacional/pgvector)")
    ap.add_argument("-p", "--pergunta", action="append", help="Pergunta única (pode repetir a flag)")
    ap.add_argument("--json", action="store_true", help="Imprime resposta + rastro de ferramentas em JSON")
    ap.add_argument("--web", action="store_true", help="Habilita busca na internet (buscar_web)")
    ap.add_argument("--mostrar-pensamento", action="store_true", help="Exibe o raciocínio <think>")
    ap.add_argument("--pensar", action="store_true",
                    help="Liga o raciocínio <think> (na avaliação: mesma acurácia e 4x mais lento; padrão desligado)")
    ap.add_argument("--sem-pensar", action="store_true", help=argparse.SUPPRESS)  # compatibilidade (já é o padrão)
    ap.add_argument("--max-rodadas", type=int, default=6, help="Máximo de rodadas de ferramentas por pergunta")
    ap.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"],
                    help="Onde rodar o modelo (padrão: auto — GPU se houver; AGENT_DEVICE no .env)")
    ap.add_argument("--prompt", action="store_true", help="Só imprime o prompt de sistema (esquema) e sai")
    args = ap.parse_args()

    if args.prompt:
        print(montar_system_prompt(args.web))
        return

    if args.device:
        os.environ["AGENT_DEVICE"] = args.device
        if args.device == "cpu":
            os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
    agente = criar_agente(web=args.web, mostrar_pensamento=args.mostrar_pensamento, pensar=args.pensar,
                          stream=not args.json, max_rodadas=args.max_rodadas, verbose_tools=not args.json)

    if args.pergunta:
        turnos = []
        for q in args.pergunta:
            agente.limpar()
            turnos.append(agente.perguntar(q))
        if args.json:
            print(json.dumps([t.to_dict() for t in turnos], ensure_ascii=False, indent=2))
        return

    print("\nAgente Computação/UFPel pronto. Comandos: /limpar (nova conversa), /sair\n", file=sys.stderr)
    while True:
        try:
            q = input("\033[1mVocê:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/sair", "/quit", "/exit"):
            break
        if q == "/limpar":
            agente.limpar()
            print("(histórico limpo)")
            continue
        print("\033[1mAgente:\033[0m ", end="", flush=True)
        agente.perguntar(q)
        print()


if __name__ == "__main__":
    main()
