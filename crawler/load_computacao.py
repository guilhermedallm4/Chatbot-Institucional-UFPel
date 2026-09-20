"""
Carga do dataset de Computação no PostgreSQL (relacional + pgvector)
=============================================================================
Consome o JSON de crawl_computacao.py e popula as duas camadas de
schema_computacao.sql:

  1. Tabelas relacionais — carga por REFRESH COMPLETO (TRUNCATE + INSERT em
     lote). Idempotente por construção: rodar duas vezes dá o mesmo estado,
     sem precisar de ON CONFLICT em nenhuma tabela.

  2. Tabelas emb_* — um embedding por "faceta" de cada entidade
     (escopo), com FK de volta para a entidade relacional.

Sobre o texto que vai para o embedding
--------------------------------------
Cada trecho vetorizado é AUTOCONTIDO: começa identificando a entidade
("Disciplina ALGORITMOS E PROGRAMAÇÃO (22000294) — Ementa"). Um chunk
recuperado precisa fazer sentido sozinho dentro do prompt, sem depender do
registro vizinho — é o que mais afeta a qualidade da resposta.

Textos longos são divididos em chunks de ~1200 caracteres. O modelo
nv-embedqa-e5-v5 corta a entrada em ~512 tokens; sem essa divisão, uma seção
de "Objetivos" com 4 mil caracteres perderia metade do conteúdo
silenciosamente (truncate="END"). Cada chunk recebe escopo próprio
('objetivos#1', 'objetivos#2'), preservando a unicidade (entidade, escopo).

Uso
---
    # schema + carga completa (relacional + embeddings)
    python load_computacao.py --input computacao.json --schema

    # só a camada relacional (rápido, sem custo de API)
    python load_computacao.py --input computacao.json --skip-embeddings

    # inspeção: mostra contagens e os textos que seriam vetorizados
    python load_computacao.py --input computacao.json --dry-run

    # só recalcular embeddings (relacional intacto)
    python load_computacao.py --input computacao.json --only-embeddings
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import psycopg2
from psycopg2.extras import Json, execute_values

# Permite importar config/providers de aplicacao/
_APP_DIR = Path(__file__).resolve().parent.parent / "aplicacao"
sys.path.insert(0, str(_APP_DIR))

import config                                       # noqa: E402
from crawl_computacao import Dataset                 # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("load_computacao")

SCHEMA_FILE = Path(__file__).with_name("schema_computacao.sql")

#: Tamanho do chunk vetorizado.
#
#: O nemotron-3-embed-1b aceita ~4096 tokens (65.536 caracteres), muito mais
#: que os ~512 tokens do nv-embedqa-e5-v5 para o qual 1200 foi calibrado. Ainda
#: assim não usamos o limite: chunk grande DILUI o sinal — um "inteligência
#: artificial" perdido em 20 mil caracteres pontua igual a texto irrelevante.
#: 2000 reduz a fragmentação das seções longas de projeto (metodologia chega a
#: 9 mil caracteres) mantendo densidade de sinal para a recuperação.
CHUNK_CHARS = 2000
CHUNK_OVERLAP = 200

#: Texto abaixo disto não vira embedding. Algumas seções do portal contêm só
#: o cabeçalho ("Bibliografia Básica:", 20 chars) sem nenhum item — vetorizar
#: isso injeta ruído no índice e ainda concorre com os acertos reais.
MIN_TEXT_CHARS = 25

#: Lotes de embedding e pausa entre eles (respeita rate limit da API).
#: A API aceita lotes de até 256 itens; 100 dá boa vazão sem tornar caro o
#: reprocessamento quando um lote falha.
EMBED_BATCH = 100
EMBED_DELAY = 0.3

#: Tabelas vetoriais e a coluna de FK de cada uma.
EMB_TABLES: dict[str, str] = {
    "emb_curso":      "curso_codigo",
    "emb_disciplina": "disciplina_codigo",
    "emb_servidor":   "servidor_id",
    "emb_projeto":    "projeto_id",
    "emb_turma":      "turma_id",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────────────────────────────────────

def _connect():
    conn = psycopg2.connect(**config.DB_CONFIG)
    conn.autocommit = False
    return conn


def _chunk_text(texto: str, size: int = CHUNK_CHARS,
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


def _juntar(*partes: Optional[str], sep: str = "\n") -> str:
    return sep.join(p for p in partes if p)


def _lista(itens: Iterable[Optional[str]], prefixo: str = "  - ") -> Optional[str]:
    linhas = [f"{prefixo}{i}" for i in itens if i]
    return "\n".join(linhas) if linhas else None


def _vetor_literal(vetor: list[float]) -> str:
    """Serializa um embedding no literal aceito pelo pgvector."""
    return "[" + ",".join(f"{v:.7g}" for v in vetor) + "]"


# ─────────────────────────────────────────────────────────────────────────────
# Camada relacional
# ─────────────────────────────────────────────────────────────────────────────

def _ajustar_dims(sql: str) -> str:
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


def aplicar_schema(conn) -> None:
    """Executa schema_computacao.sql (DROP + CREATE de tudo)."""
    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(f"schema não encontrado: {SCHEMA_FILE}")
    log.info("[Schema] aplicando %s", SCHEMA_FILE.name)
    with conn.cursor() as cur:
        cur.execute(_ajustar_dims(SCHEMA_FILE.read_text(encoding="utf-8")))
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
    Esvazia as tabelas de dados em ordem inversa de dependência.

    As emb_* têm FK para as entidades, então caem no CASCADE — por isso o
    refresh relacional exige recarregar os embeddings depois.
    """
    alvos = [t for t in reversed(list(tabelas))]
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {', '.join(alvos)} CASCADE")
    conn.commit()
    log.info("[Truncate] %d tabelas esvaziadas (CASCADE atinge as emb_*)", len(alvos))


def inserir_tabela(conn, tabela: str, rows: list[dict]) -> int:
    """INSERT em lote. Retorna o número de linhas inseridas."""
    if not rows:
        return 0
    cols = _colunas(rows)
    valores = [tuple(row.get(c) for c in cols) for row in rows]
    sql = (f"INSERT INTO {tabela} ({', '.join(cols)}) VALUES %s "
           f"ON CONFLICT DO NOTHING")
    with conn.cursor() as cur:
        execute_values(cur, sql, valores, page_size=500)
    return len(rows)


def carregar_relacional(conn, dados: dict, truncate: bool = True) -> dict[str, int]:
    """Carrega todas as tabelas relacionais na ordem de dependência."""
    tabelas = [t for t in Dataset.ORDER]
    if truncate:
        truncar(conn, tabelas)

    resumo: dict[str, int] = {}
    for tabela in tabelas:
        rows = dados.get(tabela) or []
        try:
            n = inserir_tabela(conn, tabela, rows)
            conn.commit()
        except psycopg2.Error as exc:
            conn.rollback()
            log.error("[%s] falha na carga: %s", tabela, str(exc).splitlines()[0])
            raise
        resumo[tabela] = n
        if n:
            log.info("[%-34s] %6d linhas", tabela, n)
    return resumo


# ─────────────────────────────────────────────────────────────────────────────
# Camada vetorial — construção dos textos
# ─────────────────────────────────────────────────────────────────────────────

class Indices:
    """Índices em memória para montar textos autocontidos sem consultar o banco."""

    def __init__(self, dados: dict):
        self.curso = {r["codigo_ufpel"]: r for r in dados.get("curso", [])}
        self.disciplina = {r["codigo"]: r for r in dados.get("disciplina", [])}
        self.servidor = {r["servidor_id"]: r for r in dados.get("servidor", [])}
        self.projeto = {r["projeto_id"]: r for r in dados.get("projeto", [])}
        self.turma = {r["turma_id"]: r for r in dados.get("turma", [])}

        # disciplina → [(curso_nome, semestre_rotulo, carater)]
        self.disc_cursos: dict[str, list[tuple]] = defaultdict(list)
        for m in dados.get("curso_matriz", []):
            curso = self.curso.get(m["curso_codigo"], {})
            self.disc_cursos[m["disciplina_codigo"]].append(
                (curso.get("nome"), m.get("semestre_rotulo"), m.get("carater")))

        # servidor → nomes de cursos
        self.serv_cursos: dict[str, set[str]] = defaultdict(set)
        for sc in dados.get("servidor_curso", []):
            nome = self.curso.get(sc["curso_codigo"], {}).get("nome")
            if nome:
                self.serv_cursos[sc["servidor_id"]].add(nome)

        # servidor → áreas de atuação
        self.serv_areas: dict[str, list[str]] = defaultdict(list)
        for a in sorted(dados.get("servidor_area_atuacao", []),
                        key=lambda r: r.get("ordem") or 0):
            self.serv_areas[a["servidor_id"]].append(a["area"])

        # servidor → formação
        self.serv_formacao: dict[str, list[str]] = defaultdict(list)
        for f in sorted(dados.get("servidor_formacao", []),
                        key=lambda r: r.get("ordem") or 0):
            self.serv_formacao[f["servidor_id"]].append(f["texto_original"])

        # servidor → projetos vigentes (título, ênfase, papel)
        self.serv_projetos: dict[str, list[tuple]] = defaultdict(list)
        for sp in dados.get("servidor_projeto", []):
            proj = self.projeto.get(sp["projeto_id"])
            if proj:
                self.serv_projetos[sp["servidor_id"]].append(
                    (proj["titulo"], sp.get("enfase") or proj.get("enfase"), sp.get("papel")))

        # projeto → servidores
        self.proj_servidores: dict[str, list[str]] = defaultdict(list)
        for sp in dados.get("servidor_projeto", []):
            nome = self.servidor.get(sp["servidor_id"], {}).get("nome")
            if nome:
                self.proj_servidores[sp["projeto_id"]].append(nome)

        # curso → conceitos e totais de vagas
        self.curso_conceitos: dict[str, list[str]] = defaultdict(list)
        for c in dados.get("curso_conceito", []):
            ano = f" ({c['ano']})" if c.get("ano") else ""
            self.curso_conceitos[c["curso_codigo"]].append(f"{c['indicador']}{ano}: {c['nota']}")

        self.curso_vagas: dict[str, list[str]] = defaultdict(list)
        for v in dados.get("curso_vaga", []):
            if v.get("cota") == "TOTAL":
                periodo = f" {v['ano']}/{v['semestre']}" if v.get("ano") else ""
                self.curso_vagas[v["curso_codigo"]].append(
                    f"{v['processo']}{periodo}: {v['vagas']} vagas")

        # turma → professores / horários / vínculo curricular
        self.turma_profs: dict[str, list[str]] = defaultdict(list)
        for tp in dados.get("turma_professor", []):
            self.turma_profs[tp["turma_id"]].append(tp["nome"])

        self.turma_horarios: dict[str, list[str]] = defaultdict(list)
        for th in sorted(dados.get("turma_horario", []), key=lambda r: r.get("ordem") or 0):
            if th.get("dia_semana") and th.get("hora_inicio"):
                self.turma_horarios[th["turma_id"]].append(
                    f"{th['dia_semana']} {th['hora_inicio']}-{th.get('hora_fim') or ''}".strip("-"))

        self.turma_curriculo: dict[str, list[tuple]] = defaultdict(list)
        for tc in dados.get("turma_curriculo", []):
            nome = self.curso.get(tc["curso_codigo"], {}).get("nome")
            self.turma_curriculo[tc["turma_id"]].append(
                (nome, tc.get("versao"), tc.get("semestre_rotulo")))

        # seções (curso/disciplina/projeto)
        self.curso_secoes = self._por_chave(dados.get("curso_info_secao", []), "curso_codigo")
        self.disc_secoes = self._por_chave(dados.get("disciplina_conteudo", []), "disciplina_codigo")
        self.proj_secoes = self._por_chave(dados.get("projeto_info_secao", []), "projeto_id")

    @staticmethod
    def _por_chave(rows: list[dict], chave: str) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = defaultdict(list)
        for r in sorted(rows, key=lambda x: x.get("ordem") or 0):
            out[r[chave]].append(r)
        return out


def _emitir(destino: list[dict], fk_col: str, fk_val: str, escopo: str,
            titulo: str, url: Optional[str], texto: Optional[str],
            metadata: dict) -> None:
    """
    Adiciona um ou mais registros vetoriais, dividindo textos longos.

    O primeiro chunk mantém o escopo puro ('ementa'); os seguintes recebem
    sufixo ('ementa#2'), preservando UNIQUE(entidade, escopo).
    """
    if not texto or len(texto.strip()) < MIN_TEXT_CHARS:
        return
    pedacos = _chunk_text(texto)
    total = len(pedacos)
    for i, pedaco in enumerate(pedacos, start=1):
        # Reafirma a identidade da entidade em cada chunk: um trecho
        # recuperado precisa ser interpretável fora do documento original.
        cabecalho = f"{titulo}" + (f" (parte {i}/{total})" if total > 1 else "")
        destino.append({
            fk_col: fk_val,
            "escopo": escopo if i == 1 else f"{escopo}#{i}",
            "titulo": titulo,
            "url": url,
            "texto": f"{cabecalho}\n{pedaco}",
            "metadata": {**metadata, "chunk": i, "chunks": total},
        })


def montar_emb_curso(dados: dict, ix: Indices) -> list[dict]:
    out: list[dict] = []
    for c in dados.get("curso", []):
        cod = c["codigo_ufpel"]
        rotulo = f"Curso {c['nome']} ({c.get('grau') or c.get('nivel') or ''})".strip()
        meta = {"tipo": "curso", "curso_codigo": cod, "nivel": c.get("nivel"),
                "grau": c.get("grau"), "modalidade": c.get("modalidade"),
                "unidade": c.get("unidade_nome")}

        ficha = _juntar(
            f"Nível: {c.get('nivel') or 'não informado'} | Grau: {c.get('grau') or 'não informado'}",
            f"Modalidade: {c.get('modalidade') or 'não informado'} | Turno: {c.get('turno') or 'não informado'}",
            f"Unidade responsável: {c.get('unidade_nome') or 'não informado'}",
            f"Coordenador: {c.get('coordenador_nome') or 'não informado'}",
            f"Programa: {c['programa']}" if c.get("programa") else None,
            f"Código UFPel: {cod}"
            + (f" | Código e-MEC: {c['codigo_emec']}" if c.get("codigo_emec") else "")
            + (f" | Código CAPES: {c['codigo_capes']}" if c.get("codigo_capes") else ""),
            ("Conceitos: " + "; ".join(ix.curso_conceitos[cod])) if ix.curso_conceitos[cod] else None,
            ("Vagas: " + "; ".join(ix.curso_vagas[cod])) if ix.curso_vagas[cod] else None,
        )
        _emitir(out, "curso_codigo", cod, "ficha", rotulo, c.get("url"), ficha, meta)

        for s in ix.curso_secoes.get(cod, []):
            _emitir(out, "curso_codigo", cod, s["secao_slug"],
                    f"{rotulo} — {s['secao']}", c.get("url"), s["texto"],
                    {**meta, "secao": s["secao"]})
    return out


def montar_emb_disciplina(dados: dict, ix: Indices) -> list[dict]:
    out: list[dict] = []
    for d in dados.get("disciplina", []):
        cod = d["codigo"]
        rotulo = f"Disciplina {d['nome']} (código {cod})"
        cursos = sorted({nome for nome, _, _ in ix.disc_cursos.get(cod, []) if nome})
        meta = {"tipo": "disciplina", "disciplina_codigo": cod,
                "cursos": cursos, "unidade": d.get("unidade_nome"),
                "creditos": d.get("creditos")}

        secoes = {s["secao_slug"]: s["texto"] for s in ix.disc_secoes.get(cod, [])}
        ficha = _juntar(
            f"Créditos: {d.get('creditos') or 'não informado'} | "
            f"Carga horária: {d.get('carga_horaria') or 'não informado'} horas",
            f"Tipo: {d.get('tipo_atividade') or 'não informado'} | "
            f"Periodicidade: {d.get('periodicidade') or 'não informado'}",
            f"Unidade responsável: {d.get('unidade_nome') or 'não informado'}",
            ("Ofertada em: " + "; ".join(
                f"{nome} ({sem}, {car})" for nome, sem, car in ix.disc_cursos.get(cod, []) if nome)
             ) if ix.disc_cursos.get(cod) else None,
            f"Ementa: {secoes['ementa']}" if secoes.get("ementa") else None,
        )
        _emitir(out, "disciplina_codigo", cod, "ficha", rotulo, d.get("url"), ficha, meta)

        for s in ix.disc_secoes.get(cod, []):
            _emitir(out, "disciplina_codigo", cod, s["secao_slug"],
                    f"{rotulo} — {s['secao']}", d.get("url"), s["texto"],
                    {**meta, "secao": s["secao"]})
    return out


def montar_emb_servidor(dados: dict, ix: Indices) -> list[dict]:
    out: list[dict] = []
    for s in dados.get("servidor", []):
        sid = s["servidor_id"]
        rotulo = f"Professor(a) {s['nome']}"
        cursos = sorted(ix.serv_cursos.get(sid, ()))
        areas = ix.serv_areas.get(sid, [])
        # vinculo_ativo no metadata permite restringir a busca semântica ao
        # corpo docente atual sem precisar de JOIN com servidor.
        meta = {"tipo": "servidor", "servidor_id": sid, "cursos": cursos,
                "lotacao": s.get("lotacao_nome"), "titulacao": s.get("titulacao"),
                "areas": areas, "vinculo_ativo": s.get("vinculo_ativo", True)}

        ficha = _juntar(
            f"Cargo: {s.get('cargo') or 'não informado'} | "
            f"Titulação: {s.get('titulacao') or 'não informado'}",
            f"Lotação: {s.get('lotacao_nome') or 'não informado'}",
            f"Regime: {s.get('regime_jornada')}" if s.get("regime_jornada") else None,
            # Explícito no texto: sem isso o LLM apresentaria um ex-professor
            # como docente atual ao sintetizar o trecho recuperado.
            None if s.get("vinculo_ativo", True) else
            (f"Vínculo NÃO corrente: encerrado em {s['data_saida_cargo']}."
             if s.get("data_saida_cargo") else "Vínculo NÃO corrente na UFPel."),
            ("Professor(a) dos cursos de Computação: " + ", ".join(cursos)) if cursos else None,
            ("Áreas de atuação: " + "; ".join(areas)) if areas else None,
        )
        _emitir(out, "servidor_id", sid, "ficha", rotulo, s.get("url"), ficha, meta)

        _emitir(out, "servidor_id", sid, "curriculo_resumo",
                f"{rotulo} — Resumo do currículo", s.get("url"),
                s.get("curriculo_resumo"), meta)

        _emitir(out, "servidor_id", sid, "areas_atuacao",
                f"{rotulo} — Áreas de atuação e pesquisa", s.get("url"),
                _lista(areas), meta)

        _emitir(out, "servidor_id", sid, "formacao",
                f"{rotulo} — Formação acadêmica", s.get("url"),
                _lista(ix.serv_formacao.get(sid, [])), meta)

        # Projetos no texto do professor: faz "quais professores têm projetos
        # de inteligência artificial" acertar já na busca por servidor, sem
        # depender de um segundo salto por emb_projeto.
        projetos = ix.serv_projetos.get(sid, [])
        _emitir(out, "servidor_id", sid, "projetos",
                f"{rotulo} — Projetos vigentes", s.get("url"),
                _lista(f"{titulo} ({enfase or 'ênfase não informada'}"
                       + (f", {papel.lower()}" if papel else "") + ")"
                       for titulo, enfase, papel in projetos),
                {**meta, "n_projetos": len(projetos)})
    return out


def montar_emb_projeto(dados: dict, ix: Indices) -> list[dict]:
    out: list[dict] = []
    for p in dados.get("projeto", []):
        pid = p["projeto_id"]
        rotulo = f"Projeto {p['titulo']}"
        equipe = sorted(set(ix.proj_servidores.get(pid, ())))
        meta = {"tipo": "projeto", "projeto_id": pid, "enfase": p.get("enfase"),
                "area_cnpq": p.get("area_cnpq"), "unidade": p.get("unidade_origem"),
                "professores": equipe}

        ficha = _juntar(
            f"Ênfase: {p.get('enfase') or 'não informado'} | "
            f"Área CNPq: {p.get('area_cnpq') or 'não informado'}",
            f"Coordenador: {p.get('coordenador_nome') or 'não informado'}",
            f"Unidade de origem: {p.get('unidade_origem') or 'não informado'}",
            f"Vigência: {p.get('data_inicio') or '?'} a {p.get('data_fim') or 'sem data final'}",
            ("Professores envolvidos: " + ", ".join(equipe)) if equipe else None,
            f"Resumo: {p['resumo']}" if p.get("resumo") else None,
        )
        _emitir(out, "projeto_id", pid, "ficha", rotulo, p.get("url"), ficha, meta)

        for s in ix.proj_secoes.get(pid, []):
            _emitir(out, "projeto_id", pid, s["secao_slug"],
                    f"{rotulo} — {s['secao']}", p.get("url"), s["texto"],
                    {**meta, "secao": s["secao"]})
    return out


def montar_emb_turma(dados: dict, ix: Indices) -> list[dict]:
    out: list[dict] = []
    for t in dados.get("turma", []):
        tid = t["turma_id"]
        disc = ix.disciplina.get(t["disciplina_codigo"], {})
        nome_disc = disc.get("nome") or t["disciplina_codigo"]
        rotulo = (f"Turma {t['codigo_turma']} de {nome_disc} "
                  f"({t['ano']}/{t['semestre']})")
        vinculos = ix.turma_curriculo.get(tid, [])
        profs = sorted(set(ix.turma_profs.get(tid, ())))
        horarios = ix.turma_horarios.get(tid, [])
        meta = {"tipo": "turma", "turma_id": tid,
                "disciplina_codigo": t["disciplina_codigo"],
                "ano": t["ano"], "semestre": t["semestre"],
                "cursos": sorted({n for n, _, _ in vinculos if n}),
                "versoes": sorted({v for _, v, _ in vinculos if v})}

        texto = _juntar(
            f"Disciplina: {nome_disc} (código {t['disciplina_codigo']})",
            ("Ofertada para: " + "; ".join(
                f"{nome} — {sem} (versão de currículo {versao})"
                for nome, versao, sem in vinculos if nome)) if vinculos else None,
            ("Professores: " + ", ".join(profs)) if profs else None,
            ("Horários: " + ", ".join(horarios)) if horarios else None,
            f"Vagas: {t.get('vagas') or 'não informado'} | "
            f"Matriculados: {t.get('matriculados') or 'não informado'}",
        )
        _emitir(out, "turma_id", tid, "oferta", rotulo, disc.get("url"), texto, meta)
    return out


def montar_todos_embeddings(dados: dict) -> dict[str, list[dict]]:
    ix = Indices(dados)
    return {
        "emb_curso":      montar_emb_curso(dados, ix),
        "emb_disciplina": montar_emb_disciplina(dados, ix),
        "emb_servidor":   montar_emb_servidor(dados, ix),
        "emb_projeto":    montar_emb_projeto(dados, ix),
        "emb_turma":      montar_emb_turma(dados, ix),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Camada vetorial — geração e carga
# ─────────────────────────────────────────────────────────────────────────────

def carregar_embeddings(conn, registros: dict[str, list[dict]],
                        batch: int = EMBED_BATCH,
                        delay: float = EMBED_DELAY) -> dict[str, int]:
    """
    Gera embeddings em lote e insere nas tabelas emb_*.

    Falha de lote não aborta a carga: o lote é registrado e o processo segue,
    para que uma indisponibilidade momentânea da API não custe o crawl todo.
    """
    from providers import get_embeddings                      # noqa: E402

    embeddings = get_embeddings()
    resumo: dict[str, int] = {}

    for tabela, fk_col in EMB_TABLES.items():
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
            textos = [r["texto"] for r in lote]
            try:
                vetores = embeddings.embed_documents(textos)
            except Exception as exc:
                log.error("[%s] lote %d-%d falhou: %s",
                          tabela, i, i + len(lote), str(exc)[:120])
                continue

            valores = [
                (r[fk_col], r["escopo"], r["titulo"], r.get("url"),
                 r["texto"], _vetor_literal(v), Json(r.get("metadata") or {}))
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
            log.info("[%-14s] %d/%d vetorizados", tabela, inseridos, len(rows))
            if delay and i + batch < len(rows):
                time.sleep(delay)

        resumo[tabela] = inseridos
    return resumo


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Carrega o dataset de Computação no PostgreSQL "
                    "(tabelas relacionais + pgvector).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default="computacao.json", metavar="FILE",
                   help="JSON produzido por crawl_computacao.py")
    p.add_argument("--schema", action="store_true",
                   help="Aplica schema_computacao.sql antes da carga (DROP + CREATE)")
    p.add_argument("--no-truncate", action="store_true",
                   help="Não esvazia as tabelas antes de inserir (carga incremental)")
    p.add_argument("--skip-embeddings", action="store_true",
                   help="Carrega só a camada relacional (sem custo de API)")
    p.add_argument("--only-embeddings", action="store_true",
                   help="Recalcula apenas os embeddings, sem mexer no relacional")
    p.add_argument("--dry-run", action="store_true",
                   help="Não toca no banco: mostra contagens e amostras dos textos")
    p.add_argument("--batch-size", type=int, default=EMBED_BATCH, metavar="N",
                   help="Textos por chamada de embedding")
    p.add_argument("--delay", type=float, default=EMBED_DELAY, metavar="SECS",
                   help="Pausa entre lotes de embedding")
    return p


def _relatorio_dry_run(dados: dict, registros: dict[str, list[dict]]) -> None:
    print("=" * 70)
    print("  CAMADA RELACIONAL")
    print("=" * 70)
    for tabela in Dataset.ORDER:
        n = len(dados.get(tabela) or [])
        if n:
            print(f"    {tabela:<34} {n:>6}")

    print()
    print("=" * 70)
    print("  CAMADA VETORIAL")
    print("=" * 70)
    for tabela, rows in registros.items():
        escopos = defaultdict(int)
        for r in rows:
            escopos[re.sub(r"#\d+$", "", r["escopo"])] += 1
        print(f"    {tabela:<16} {len(rows):>6} vetores")
        for escopo, n in sorted(escopos.items(), key=lambda x: -x[1]):
            print(f"        {escopo:<34} {n:>5}")

    print()
    print("=" * 70)
    print("  AMOSTRA DOS TEXTOS VETORIZADOS")
    print("=" * 70)
    for tabela, rows in registros.items():
        if not rows:
            continue
        vistos: set[str] = set()
        for r in rows:
            escopo = re.sub(r"#\d+$", "", r["escopo"])
            if escopo in vistos:
                continue
            vistos.add(escopo)
            print(f"\n--- {tabela} / escopo={r['escopo']} ---")
            texto = r["texto"]
            print(texto[:600] + ("…" if len(texto) > 600 else ""))
            if len(vistos) >= 3:
                break


def main() -> None:
    args = _build_parser().parse_args()

    caminho = Path(args.input)
    if not caminho.exists():
        sys.exit(f"Arquivo não encontrado: {caminho}\n"
                 f"Rode primeiro: python crawl_computacao.py --output {caminho}")

    dados = json.loads(caminho.read_text(encoding="utf-8"))
    log.info("[Entrada] %s — %d tabelas com dados", caminho.name,
             sum(1 for t in Dataset.ORDER if dados.get(t)))

    registros = montar_todos_embeddings(dados)

    if args.dry_run:
        _relatorio_dry_run(dados, registros)
        return

    conn = _connect()
    try:
        if args.schema:
            aplicar_schema(conn)

        if not args.only_embeddings:
            resumo = carregar_relacional(conn, dados,
                                         truncate=not (args.no_truncate or args.schema))
            total = sum(resumo.values())
            log.info("[Relacional] %d linhas em %d tabelas", total,
                     sum(1 for v in resumo.values() if v))

        if args.skip_embeddings:
            log.info("[Vetorial] ignorado (--skip-embeddings)")
        else:
            resumo_emb = carregar_embeddings(conn, registros,
                                             batch=args.batch_size, delay=args.delay)
            log.info("[Vetorial] %d vetores em %d tabelas",
                     sum(resumo_emb.values()),
                     sum(1 for v in resumo_emb.values() if v))

        with conn.cursor() as cur:
            cur.execute("ANALYZE")
        conn.commit()
        log.info("[Concluído] ANALYZE executado")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
