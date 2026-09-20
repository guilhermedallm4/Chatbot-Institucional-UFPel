"""
Carga do PORTAL da Computação-UFPel no PostgreSQL (relacional + pgvector)
=============================================================================
Consome o JSON de `crawl_portal_computacao.py` e popula as duas camadas de
`schema_portal_computacao.sql`.

Uso
---
    # schema + carga completa
    python load_portal_computacao.py --input portal_computacao.json --schema

    # inspeção: contagens e os textos que seriam vetorizados (não toca o banco)
    python load_portal_computacao.py --input portal_computacao.json --dry-run

    # só o relacional (rápido, sem custo de API)
    python load_portal_computacao.py --input portal_computacao.json --skip-embeddings

    # só revetorizar (relacional intacto)
    python load_portal_computacao.py --input portal_computacao.json --only-embeddings

O que é vetorizado, e por quê
-----------------------------
  emb_port_post     uma notícia por embedding (fatiada se longa). O cabeçalho
                    traz título, DATA POR EXTENSO, seção e curso — é o que
                    permite responder "quando isso foi publicado?" sem um
                    JOIN extra e sem o modelo inventar a data.

  emb_port_pagina   uma linha por SEÇÃO da página, mais uma "ficha" por página
                    (título + caminho + resumo) para as perguntas de navegação
                    ("onde encontro o horário dos laboratórios?").

  emb_port_faq      um embedding por par pergunta/resposta.

O que NÃO é vetorizado, de propósito: `port_calendario_evento`,
`port_pessoa`, `port_grupo_pesquisa` e `port_documento`. São registros curtos
e factuais, sempre melhor servidos por SQL (`WHERE data BETWEEN …`,
`nome ILIKE …`) — vetorizá-los encheria o índice de trechos de uma linha que
competem com o conteúdo real.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_wp_common import (                              # noqa: E402
    aplicar_schema, carregar_embeddings, carregar_json, carregar_relacional,
    connect, emitir, juntar, mapear_ids, relatorio_dry_run, rotulo,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("load_portal")

SCHEMA_FILE = Path(__file__).with_name("schema_portal_computacao.sql")

#: Ordem de carga — espelha PortalDataset.ORDER e respeita as FKs.
ORDEM = (
    "port_crawl_meta", "port_secao", "port_autor", "port_categoria",
    "port_tag", "port_pagina", "port_menu_item", "port_pagina_secao",
    "port_post", "port_post_categoria", "port_post_tag", "port_link",
    "port_documento", "port_faq", "port_pessoa", "port_grupo_pesquisa",
    "port_calendario_evento",
)

EMB_TABLES = {
    "emb_port_post":   "post_id",
    "emb_port_pagina": "pagina_id",
    "emb_port_faq":    "faq_id",
}

#: Rótulo legível de `escopo_curso`, para entrar no texto vetorizado. "ambos"
#: não significa nada num embedding; "Ciência da Computação e Engenharia de
#: Computação" casa com quem pergunta pelo nome do curso.
_CURSO_POR_ESCOPO = {
    "ccomp": "Ciência da Computação",
    "ecomp": "Engenharia de Computação",
    "ambos": "Ciência da Computação e Engenharia de Computação",
    "geral": None,
}


class Indices:
    """Mapas de apoio para montar os textos sem varrer listas repetidamente."""

    def __init__(self, dados: dict) -> None:
        self.pagina = {p["pagina_id"]: p for p in dados.get("port_pagina", [])}
        self.secao = {s["secao_slug"]: s for s in dados.get("port_secao", [])}
        self.autor = {a["autor_id"]: a for a in dados.get("port_autor", [])}
        self.categoria = {c["categoria_id"]: c
                          for c in dados.get("port_categoria", [])}
        self.tag = {t["tag_id"]: t for t in dados.get("port_tag", [])}

        self.cats_do_post: dict[int, list[str]] = {}
        for pc in dados.get("port_post_categoria", []):
            cat = self.categoria.get(pc["categoria_id"])
            if cat:
                self.cats_do_post.setdefault(pc["post_id"], []).append(cat["nome"])
        self.tags_do_post: dict[int, list[str]] = {}
        for pt in dados.get("port_post_tag", []):
            tag = self.tag.get(pt["tag_id"])
            if tag:
                self.tags_do_post.setdefault(pt["post_id"], []).append(tag["nome"])

    def nome_secao(self, slug: Optional[str]) -> Optional[str]:
        s = self.secao.get(slug or "")
        return s["nome"] if s else None


def _caminho_legivel(caminho: Optional[str]) -> Optional[str]:
    """"sobre-a-computacao-ufpel/laboratorios" → "sobre a computacao ufpel › laboratorios"."""
    if not caminho:
        return None
    return " › ".join(p.replace("-", " ") for p in caminho.split("/"))


# ─────────────────────────────────────────────────────────────────────────────
# Construção dos textos vetorizados
# ─────────────────────────────────────────────────────────────────────────────

def montar_emb_post(dados: dict, ix: Indices) -> list[dict]:
    saida: list[dict] = []
    for post in dados.get("port_post", []):
        categorias = ix.cats_do_post.get(post["post_id"], [])
        tags = ix.tags_do_post.get(post["post_id"], [])
        autor = ix.autor.get(post.get("autor_id")) or {}
        curso = _CURSO_POR_ESCOPO.get(post.get("escopo_curso"))

        cabecalho = juntar(
            "Notícia do portal da Computação-UFPel",
            rotulo("Título", post["titulo"]),
            # a data por extenso é o que torna "o que saiu em julho de 2026?"
            # uma pergunta respondível por similaridade
            rotulo("Publicada em", post.get("data_por_extenso")),
            rotulo("Curso", curso),
            rotulo("Categorias", categorias),
            rotulo("Palavras-chave", tags),
            rotulo("Autor", autor.get("nome")),
        )
        corpo = post.get("texto") or post.get("resumo") or ""
        emitir(saida, "post_id", post["post_id"], "noticia", post["titulo"],
               post.get("url"), f"{cabecalho}\n\n{corpo}",
               {"tipo": "noticia", "acervo": "portal",
                "ano": post.get("ano"), "mes": post.get("mes"),
                "data": post.get("data_publicacao"),
                "secao": post.get("secao_slug"),
                "escopo_curso": post.get("escopo_curso"),
                "categorias": categorias})
    return saida


def montar_emb_pagina(dados: dict, ix: Indices) -> list[dict]:
    saida: list[dict] = []

    n_secoes: dict[int, int] = {}
    for sec in dados.get("port_pagina_secao", []):
        n_secoes[sec["pagina_id"]] = n_secoes.get(sec["pagina_id"], 0) + 1

    # 1) ficha da página — responde "onde fica X no site?"
    #
    # Só para páginas com MAIS DE UMA seção. Numa página de seção única, a
    # ficha e a seção seriam quase o mesmo texto e ocupariam duas vagas do
    # top-k com o mesmo conteúdo — foi o que a busca por "laboratórios"
    # devolveu (0,535 e 0,533, o mesmo destino duas vezes). O caminho no site,
    # que era o diferencial da ficha, passou a entrar no texto da seção.
    for pag in dados.get("port_pagina", []):
        if n_secoes.get(pag["pagina_id"], 0) <= 1:
            continue
        texto = juntar(
            "Página do portal da Computação-UFPel",
            rotulo("Título", pag["titulo"]),
            rotulo("Seção", ix.nome_secao(pag.get("secao_slug"))),
            rotulo("Caminho no site", _caminho_legivel(pag.get("caminho"))),
            rotulo("Idioma", "inglês" if pag.get("idioma") == "en" else None),
            rotulo("Atualizada em", pag.get("data_modificacao")),
            "",
            pag.get("resumo"),
        )
        emitir(saida, "pagina_id", pag["pagina_id"], "ficha", pag["titulo"],
               pag.get("url"), texto,
               {"tipo": "pagina_ficha", "acervo": "portal",
                "secao": pag.get("secao_slug"), "idioma": pag.get("idioma"),
                "caminho": pag.get("caminho")})

    # 2) uma linha por seção — a granularidade que faz a recuperação funcionar
    for sec in dados.get("port_pagina_secao", []):
        pag = ix.pagina.get(sec["pagina_id"])
        if not pag:
            continue
        titulo_completo = juntar(pag["titulo"], sec.get("titulo_secao"), sep=" — ")
        texto = juntar(
            f"Portal da Computação-UFPel — {ix.nome_secao(pag.get('secao_slug')) or 'site'}",
            rotulo("Página", pag["titulo"]),
            rotulo("Seção", sec.get("titulo_secao")),
            rotulo("Caminho no site", _caminho_legivel(pag.get("caminho"))),
            rotulo("Atualizada em", pag.get("data_modificacao")),
            "",
            sec.get("texto"),
        )
        emitir(saida, "pagina_id", sec["pagina_id"], f"secao#{sec['ordem']}",
               titulo_completo, sec.get("url"), texto,
               {"tipo": "pagina_secao", "acervo": "portal",
                "secao": pag.get("secao_slug"), "idioma": pag.get("idioma"),
                "titulo_secao": sec.get("titulo_secao"),
                "ancora": sec.get("ancora")})
    return saida


def montar_emb_faq(dados: dict, ix: Indices,
                   ids: Optional[dict[tuple, int]] = None) -> list[dict]:
    """
    Um embedding por par pergunta/resposta.

    `ids` vem de `mapear_ids(conn, "port_faq", "faq_id", ["pagina_id","ordem"])`
    depois da carga relacional — `faq_id` é BIGSERIAL e não existe no JSON.
    Sem conexão (`--dry-run`), cai para a posição na lista, que serve para
    inspecionar o texto mas não para gravar.
    """
    saida: list[dict] = []
    for i, faq in enumerate(dados.get("port_faq", []), start=1):
        chave = (faq["pagina_id"], faq["ordem"])
        faq_id = ids.get(chave) if ids else i
        if faq_id is None:
            continue
        pag = ix.pagina.get(faq["pagina_id"]) or {}
        texto = juntar(
            f"FAQ da Computação-UFPel — {faq.get('secao') or 'dúvidas gerais'}",
            f"Pergunta: {faq['pergunta']}",
            f"Resposta: {faq['resposta']}",
        )
        emitir(saida, "faq_id", faq_id, "qa", faq["pergunta"], faq.get("url"),
               texto,
               {"tipo": "faq", "acervo": "portal", "secao_faq": faq.get("secao"),
                "pagina": pag.get("titulo")})
    return saida


def montar_todos(dados: dict, conn=None) -> dict[str, list[dict]]:
    ix = Indices(dados)
    ids_faq = (mapear_ids(conn, "port_faq", "faq_id", ["pagina_id", "ordem"])
               if conn is not None else None)
    return {
        "emb_port_post":   montar_emb_post(dados, ix),
        "emb_port_pagina": montar_emb_pagina(dados, ix),
        "emb_port_faq":    montar_emb_faq(dados, ix, ids_faq),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Carga do portal da Computação-UFPel (relacional + pgvector)")
    ap.add_argument("--input", default="portal_computacao.json")
    ap.add_argument("--schema", action="store_true",
                    help="aplica schema_portal_computacao.sql antes da carga")
    ap.add_argument("--dry-run", action="store_true",
                    help="mostra contagens e textos vetorizáveis; não toca o banco")
    ap.add_argument("--skip-embeddings", action="store_true")
    ap.add_argument("--only-embeddings", action="store_true")
    ap.add_argument("--batch", type=int, default=100)
    args = ap.parse_args()

    dados = carregar_json(Path(args.input))

    if args.dry_run:
        relatorio_dry_run(dados, ORDEM, montar_todos(dados))
        return

    conn = connect()
    try:
        if args.schema:
            aplicar_schema(conn, SCHEMA_FILE)
        if not args.only_embeddings:
            carregar_relacional(conn, dados, ORDEM)
        if not args.skip_embeddings:
            # montado DEPOIS do relacional: os ids BIGSERIAL só existem lá
            registros = montar_todos(dados, conn)
            carregar_embeddings(conn, registros, EMB_TABLES, batch=args.batch)
    finally:
        conn.close()
    log.info("Carga do portal concluída")


if __name__ == "__main__":
    main()
