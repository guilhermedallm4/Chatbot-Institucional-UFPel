"""
Agente RAG institucional — LFM2.5 + PostgreSQL/pgvector
=============================================================================
Conceito — Módulo Agente

No pipeline RAG clássico (pipeline.py) o fluxo é fixo: embedding da pergunta
→ busca vetorial → prompt → LLM. Aqui o LLM *decide* como buscar:

  • buscar_semantica  — busca vetorial (pgvector) quando a pergunta é vaga
                        ou conceitual ("quem pesquisa visão computacional?")
  • buscar_por_nome   — trigram (pg_trgm) quando há um nome/título explícito
  • consultar_sql     — SELECT livre nas tabelas estruturadas quando a
                        pergunta é tabular/agregada ("quantos projetos ativos?",
                        "disciplinas com 4 créditos")
  • obter_registro    — JSON completo de um documento (matriz, turmas, equipe)
  • ler_pagina        — página do portal UFPel ao vivo (verificação/atualização)
  • buscar_web        — internet (opcional, --web)

O modelo pode encadear ferramentas: SQL para achar o doc_id → obter_registro
para os detalhes → resposta com a URL da fonte.

Uso:
  python agente_rag.py                          # chat interativo
  python agente_rag.py -p "Quem coordena o curso de Ciência da Computação?"
  python agente_rag.py -p "..." --json          # resposta + rastro em JSON
  python agente_rag.py --mostrar-pensamento     # exibe o <think>
  python agente_rag.py --web                    # habilita busca na internet
Avaliação em lote: veja avaliar_agente.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras

import config  # carrega .env
from agente_lfm import AgenteLFM, Tool, AgentTurn

# reliability.py liga logging INFO global; silencia o ruído do HuggingFace Hub
import logging
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


def _conn(readonly: bool = False):
    c = psycopg2.connect(**(_RO_CONFIG if readonly else config.DB_CONFIG))
    c.autocommit = True
    return c


_embeddings = None


def _emb():
    global _embeddings
    if _embeddings is None:
        from providers import get_embeddings
        _embeddings = get_embeddings()
    return _embeddings


# Tabelas físicas por tipo (criadas por crawler/ingest_ufpel.py)
TIPO_TABELA_VETOR = {"disciplina": "disciplinas", "projeto": "projetos", "servidor": "servidores", "curso": "cursos"}
TIPO_TABELA_INFO = {
    "disciplina": "disciplinas_info", "projeto": "projetos_info", "servidor": "servidores_info", "curso": "cursos_info",
}

MAX_TOOL_CHARS = 6000
VAZIO = "Não há informações disponíveis"


def _trunc(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + " [...]"


def _compacto(obj, max_chars: int = 4000) -> str:
    """JSON legível, removendo campos 'Não há informações disponíveis'."""
    def limpa(o):
        if isinstance(o, dict):
            return {k: limpa(v) for k, v in o.items() if v not in (VAZIO, "", None, [], {}) and not str(k).endswith("_url")}
        if isinstance(o, list):
            return [limpa(x) for x in o if x not in (VAZIO, "", None)]
        return o
    return _trunc(json.dumps(limpa(obj), ensure_ascii=False, indent=1), max_chars)


# =============================================================================
# Ferramentas
# =============================================================================

def buscar_semantica(consulta: str, tipo: str = "", top_k: int = 5) -> str:
    """Busca vetorial (pgvector) nos documentos da UFPel. Use para perguntas conceituais ou vagas, quando não há um nome exato (ex.: 'quem pesquisa processamento de vídeo?', 'disciplinas sobre programação paralela'). tipo filtra: 'disciplina', 'projeto', 'servidor' ou 'curso' ('' = todos). Retorna trechos com título, doc_id, URL e score."""
    top_k = max(1, min(int(top_k), 10))
    tipos = [tipo] if tipo in TIPO_TABELA_VETOR else list(TIPO_TABELA_VETOR)
    vec = _emb().embed_query(consulta)
    vec_lit = "[" + ",".join(f"{v:.6f}" for v in vec) + "]"

    partes = []
    for t in tipos:
        partes.append(
            f"SELECT '{t}' AS tipo, v.doc_id, v.titulo, v.conteudo, 1 - (v.embedding <=> %(v)s::vector) AS score, "
            f"d.dados->>'url' AS url FROM {TIPO_TABELA_VETOR[t]} v LEFT JOIN doc_completos d ON d.doc_id = v.doc_id"
        )
    sql = " UNION ALL ".join(partes) + " ORDER BY score DESC LIMIT %(k)s"
    with _conn(readonly=True) as c, c.cursor() as cur:
        cur.execute(sql, {"v": vec_lit, "k": top_k})
        rows = cur.fetchall()
    if not rows:
        return "Nenhum documento encontrado."
    out = []
    for i, (t, doc_id, titulo, conteudo, score, url) in enumerate(rows, 1):
        out.append(
            f"[{i}] ({t}, score={score:.2f}) {titulo}\ndoc_id: {doc_id}\nURL: {url or '-'}\n"
            f"{_trunc(conteudo.replace(chr(10), ' '), 700)}\n"
        )
    return _trunc("\n".join(out), MAX_TOOL_CHARS)


def buscar_por_nome(nome: str, tipo: str = "", top_k: int = 5) -> str:
    """Busca documentos pelo NOME/TÍTULO com tolerância a erros de digitação (trigram). Use quando a pergunta cita um nome de pessoa, disciplina, curso ou projeto (ex.: 'Daniel Palomino', 'Algoritmos e Estrutura de Dados'). tipo: 'disciplina', 'projeto', 'servidor', 'curso' ou '' (todos). Retorna título, tipo, doc_id e URL."""
    top_k = max(1, min(int(top_k), 10))
    with _conn(readonly=True) as c, c.cursor() as cur:
        cur.execute("SET pg_trgm.similarity_threshold = 0.15")
        filtro = "AND tipo = %(t)s" if tipo in TIPO_TABELA_VETOR else ""
        cur.execute(
            f"""SELECT doc_id, tipo, titulo, dados->>'url', similarity(unaccent(titulo), unaccent(%(n)s)) AS sim
                FROM doc_completos
                WHERE (unaccent(titulo) %% unaccent(%(n)s) OR unaccent(titulo) ILIKE '%%' || unaccent(%(n)s) || '%%') {filtro}
                ORDER BY sim DESC LIMIT %(k)s""",
            {"n": nome, "t": tipo, "k": top_k},
        )
        rows = cur.fetchall()
    if not rows:
        return f"Nenhum documento com nome parecido com '{nome}'. Tente buscar_semantica."
    return "\n".join(f"[{i}] ({t}, sim={sim:.2f}) {titulo} | doc_id: {doc_id} | URL: {url or '-'}"
                     for i, (doc_id, t, titulo, url, sim) in enumerate(rows, 1))


def obter_registro(doc_id: str) -> str:
    """Retorna TODOS os dados estruturados de um documento (JSON) a partir do doc_id obtido em outra ferramenta: matriz curricular e professores de um curso, ementa/turmas de uma disciplina, equipe de um projeto, currículo/projetos de um servidor."""
    with _conn(readonly=True) as c, c.cursor() as cur:
        cur.execute("SELECT tipo, titulo, dados FROM doc_completos WHERE doc_id = %s", (doc_id.strip(),))
        row = cur.fetchone()
    if not row:
        return f"doc_id '{doc_id}' não encontrado."
    tipo, titulo, dados = row
    return f"{tipo}: {titulo}\nURL: {dados.get('url', '-')}\n{_compacto(dados, MAX_TOOL_CHARS - 200)}"


_SQL_PROIBIDO = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|vacuum|call|do|execute|"
    r"pg_sleep|pg_read_file|pg_ls_dir|lo_import|lo_export|dblink|set\s+role|reset)\b", re.I)


def consultar_sql(sql: str) -> str:
    """Executa um SELECT (somente leitura) no PostgreSQL da UFPel e retorna as linhas. Use para contagens, filtros, ordenações e listas (ex.: 'quantos projetos ativos por unidade?', 'disciplinas com 4 créditos'). Veja o esquema no prompt. Sempre use LIMIT (máx. 30 linhas retornadas). Compare textos com ILIKE '%termo%' — nomes estão em MAIÚSCULAS e podem ter acentos; unaccent() está disponível."""
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        return "Erro: envie apenas UMA instrução SQL, sem ';' no meio."
    if not re.match(r"^(select|with)\b", s, re.I):
        return "Erro: apenas consultas SELECT/WITH são permitidas."
    if _SQL_PROIBIDO.search(s):
        return "Erro: a consulta contém comando não permitido (somente leitura)."
    if not re.search(r"\blimit\s+\d+", s, re.I):
        s += " LIMIT 30"
    try:
        with _conn(readonly=True) as c, c.cursor() as cur:
            cur.execute("SET statement_timeout = '15s'")
            cur.execute("SET default_transaction_read_only = on")  # 2ª barreira além do papel rag_leitor
            cur.execute(s)
            cols = [d.name for d in cur.description]
            rows = cur.fetchmany(30)
    except Exception as e:  # noqa: BLE001
        msg = str(e).split("\n")[0]
        return f"Erro SQL: {msg}\nDica: confira nomes de tabelas/colunas no esquema e use ILIKE para textos."
    if not rows:
        return "Consulta executada: 0 linhas."
    linhas = [" | ".join(cols)]
    for r in rows:
        linhas.append(" | ".join(_trunc("" if v is None else (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v), 300) for v in r))
    return _trunc(f"{len(rows)} linha(s):\n" + "\n".join(linhas), MAX_TOOL_CHARS)


_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
DOMINIOS_PERMITIDOS = ("ufpel.edu.br",)


def ler_pagina(url: str) -> str:
    """Lê o texto de uma página do portal da UFPel (somente domínios *.ufpel.edu.br). Use para conferir ou atualizar uma informação a partir da URL retornada pelas outras ferramentas, ou quando o banco não tem o dado e a página oficial provavelmente tem."""
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
# Prompt de sistema (com esquema do banco)
# =============================================================================

ESQUEMA_SQL = """\
Tabelas (PostgreSQL, somente leitura). Todas têm doc_id (TEXT, chave) e url (TEXT):
- cursos_info(nome, nivel, grau, modalidade, turno, unidade, programa, coordenador, codigo_ufpel,
    matriz_curricular JSONB[{semestre, "Código", "Disciplina / Pré-requisitos", "Caráter", "Cr.", "Horas"}],
    professores JSONB[{nome, unidade, url}], turmas_ofertadas JSONB[{disciplina, professores[], turma, vagas, matriculados}])
- disciplinas_info(codigo, nome, creditos, carga_horaria, ch_teorica, ch_pratica, periodicidade, unidade_responsavel,
    ementa, objetivos, conteudo_programatico, bibliografia, turmas_ofertadas JSONB, cursos_relacionados JSONB[texto])
- servidores_info(nome, categoria, cargo, titulacao, lotacao, regime_jornada, situacao, email, curriculo_resumo,
    formacao_academica JSONB[texto], areas_atuacao JSONB[texto], lattes_url, projetos_ativos JSONB[{titulo,...}],
    disciplinas_ministradas JSONB[{ano_semestre, turma, disciplina}], cursos_relacionados JSONB[texto])
- projetos_info(titulo, resumo, enfase, situacao, data_inicio, data_fim, coordenador, unidade_origem, area_cnpq,
    equipe_vigente JSONB[{nome, ...}], professores_relacionados JSONB[texto])
- documentos(doc_id, tipo, titulo, url, dados JSONB)  -- visão unificada de todos os tipos
Dicas: textos em MAIÚSCULAS → use ILIKE '%termo%' ou unaccent(col) ILIKE unaccent('%termo%');
listas JSONB → jsonb_array_elements(col) / col::text ILIKE '%termo%'; conte com COUNT(*); sempre LIMIT.
Exemplos:
- "quais professores pesquisam X" → SELECT nome, lotacao, url FROM servidores_info
    WHERE unaccent(curriculo_resumo || ' ' || areas_atuacao::text || ' ' || projetos_ativos::text) ILIKE unaccent('%X%') LIMIT 20
  (ou buscar_semantica(consulta, tipo='servidor')); complemente com projetos: SELECT titulo, coordenador FROM projetos_info WHERE unaccent(titulo || ' ' || resumo) ILIKE unaccent('%X%') LIMIT 20
- "disciplinas sobre X e seus créditos" → SELECT nome, codigo, creditos, carga_horaria, url FROM disciplinas_info WHERE unaccent(nome || ' ' || ementa) ILIKE unaccent('%X%') LIMIT 20
- "quantos projetos ativos por unidade" → SELECT unidade_origem, COUNT(*) FROM projetos_info WHERE situacao ILIKE 'ativo' GROUP BY 1 ORDER BY 2 DESC LIMIT 20
Uma consulta SQL bem feita costuma responder de uma vez; prefira SQL a várias chamadas de obter_registro."""

SYSTEM_PROMPT_TEMPLATE = """\
Você é o assistente institucional da UFPel (Universidade Federal de Pelotas). Responde em português, de forma \
objetiva, SOMENTE com base nos dados obtidos pelas ferramentas. Data de hoje: {data}.

Base de dados disponível (portal institucional, coletado em julho/2026): {contagens}.

Como escolher a ferramenta:
1. A pergunta cita um nome (pessoa, disciplina, curso, projeto)? → buscar_por_nome, depois obter_registro para detalhes.
2. Pergunta conceitual/temática sem nome exato? → buscar_semantica (filtre por tipo quando souber).
3. Contagem, filtro, ordenação ou lista completa? → consultar_sql com o esquema abaixo.
4. Precisa da matriz curricular, turmas, equipe ou currículo completo? → obter_registro(doc_id).
5. Quer confirmar/atualizar um dado a partir da URL oficial? → ler_pagina(url).
{regra_web}
Regras:
- Nunca invente nomes, números ou URLs. Se as ferramentas não trouxerem a informação, diga que não foi encontrada na base.
- Ao final da resposta, liste as fontes (URLs) dos documentos usados.
- Não repita a mesma chamada com os mesmos argumentos; se uma busca falhar, mude a estratégia (outra ferramenta ou termos).
- Seja conciso: responda o que foi perguntado e cite os dados relevantes.

{esquema}"""


def _contagens() -> str:
    try:
        with _conn(readonly=True) as c, c.cursor() as cur:
            cur.execute("SELECT tipo, COUNT(*) FROM doc_completos GROUP BY tipo ORDER BY tipo")
            return ", ".join(f"{n} {t}s" for t, n in cur.fetchall()) or "base vazia"
    except Exception as e:  # noqa: BLE001
        return f"(não foi possível consultar o banco: {e})"


def montar_system_prompt(web: bool = False) -> str:
    regra_web = ("6. Assunto fora da UFPel (conceitos gerais, notícias)? → buscar_web.\n" if web else
                 "Perguntas fora do escopo institucional da UFPel: responda brevemente que só trata de dados da UFPel.\n")
    return SYSTEM_PROMPT_TEMPLATE.format(
        data=datetime.now().strftime("%d/%m/%Y"), contagens=_contagens(), regra_web=regra_web, esquema=ESQUEMA_SQL,
    )


def montar_tools(web: bool = False) -> list[Tool]:
    tools = [
        Tool.from_function(buscar_por_nome, param_docs={
            "nome": "nome ou título (ou parte dele)", "tipo": "disciplina | projeto | servidor | curso | '' (todos)",
            "top_k": "quantidade de resultados (1-10)"}),
        Tool.from_function(buscar_semantica, param_docs={
            "consulta": "pergunta ou tema em linguagem natural", "tipo": "disciplina | projeto | servidor | curso | '' (todos)",
            "top_k": "quantidade de resultados (1-10)"}),
        Tool.from_function(consultar_sql, param_docs={"sql": "uma instrução SELECT ... LIMIT n"}),
        Tool.from_function(obter_registro, param_docs={"doc_id": "identificador retornado por outra ferramenta"}),
        Tool.from_function(ler_pagina, param_docs={"url": "URL em *.ufpel.edu.br"}),
    ]
    if web:
        tools.append(Tool.from_function(buscar_web, param_docs={"consulta": "termos de busca", "max_results": "1-8"}))
    return tools


def criar_agente(web: bool = False, mostrar_pensamento: bool = False, pensar: bool = True,
                 stream: bool = True, max_rodadas: int = 6, verbose_tools: bool = True, **kw) -> AgenteLFM:
    """Fábrica: agente LFM2.5 já configurado com as ferramentas da base UFPel."""
    return AgenteLFM(
        system_prompt=montar_system_prompt(web),
        tools=montar_tools(web),
        model_id=os.getenv("AGENT_MODEL_ID", kw.pop("model_id", "LiquidAI/LFM2.5-2.6B")),
        max_new_tokens=int(os.getenv("AGENT_MAX_NEW_TOKENS", kw.pop("max_new_tokens", 2048))),
        temperature=float(os.getenv("AGENT_TEMPERATURE", kw.pop("temperature", 0.1))),
        pensar=pensar, mostrar_pensamento=mostrar_pensamento, max_tool_rounds=max_rodadas,
        stream=stream, verbose_tools=verbose_tools, **kw,
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Agente RAG institucional UFPel (LFM2.5 + pgvector)")
    ap.add_argument("-p", "--pergunta", action="append", help="Pergunta única (pode repetir a flag)")
    ap.add_argument("--json", action="store_true", help="Imprime resposta + rastro de ferramentas em JSON")
    ap.add_argument("--web", action="store_true", help="Habilita busca na internet (buscar_web)")
    ap.add_argument("--mostrar-pensamento", action="store_true", help="Exibe o raciocínio <think>")
    ap.add_argument("--sem-pensar", action="store_true", help="Desliga o raciocínio (mais rápido, menos preciso)")
    ap.add_argument("--max-rodadas", type=int, default=6, help="Máximo de rodadas de ferramentas por pergunta")
    args = ap.parse_args()

    agente = criar_agente(web=args.web, mostrar_pensamento=args.mostrar_pensamento, pensar=not args.sem_pensar,
                          stream=not args.json, max_rodadas=args.max_rodadas, verbose_tools=not args.json)

    if args.pergunta:
        turnos = []
        for q in args.pergunta:
            agente.limpar()
            turnos.append(agente.perguntar(q))
        if args.json:
            print(json.dumps([t.to_dict() for t in turnos], ensure_ascii=False, indent=2))
        return

    print("\nAgente UFPel pronto. Comandos: /limpar (nova conversa), /sair\n", file=sys.stderr)
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
