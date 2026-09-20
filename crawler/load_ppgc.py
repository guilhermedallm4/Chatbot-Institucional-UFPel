"""
Carga do acervo do PPGC no PostgreSQL (relacional + pgvector)
=============================================================================
Consome o JSON de `crawl_ppgc.py` e popula as duas camadas de
`schema_ppgc.sql`.

Uso
---
    python load_ppgc.py --input ppgc.json --schema      # schema + carga
    python load_ppgc.py --input ppgc.json --dry-run     # inspeção, sem banco
    python load_ppgc.py --input ppgc.json --skip-embeddings
    python load_ppgc.py --input ppgc.json --only-embeddings

O que é vetorizado
------------------
  emb_ppgc_edital     um edital por embedding, com tipo, nível, período,
                      número oficial E a lista dos PDFs anexos. É o chunk mais
                      importante do acervo: "qual o edital de mestrado de
                      2026?" precisa recuperar exatamente um registro e já
                      trazer o link do PDF do edital.
  emb_ppgc_normativo  resolução/portaria com a VIGÊNCIA escrita no texto.
  emb_ppgc_faq        um par pergunta/resposta por embedding.
  emb_ppgc_pagina     uma linha por seção da página, mais uma "ficha"
                      (título + assunto) nas páginas com várias seções. Páginas
                      cujo conteúdo já virou tabela própria (edital, norma,
                      FAQ, linha de pesquisa, disciplina) NÃO entram aqui —
                      ver `_CATEGORIAS_JA_NORMALIZADAS`.
  emb_ppgc_post       notícia do Programa, com data por extenso e assunto.
  emb_ppgc_linha_pesquisa / emb_ppgc_disciplina
                      o material de captação: linhas de pesquisa e ementas.
  emb_ppgc_documento  os chunks dos PDFs — lidos do BANCO, não do JSON.

O que NÃO é vetorizado: `ppgc_calendario_evento`, `ppgc_requisito`,
`ppgc_docente` e `ppgc_defesa`. São fatos curtos e datados, melhor servidos
por SQL — "até quando comprovo proficiência?" é `WHERE nivel = 'mestrado'`,
e transformar isso em vizinhança vetorial só adiciona chance de errar o nível.
Dessas, só `ppgc_requisito` tem cópia semântica: a tabela de requisitos e
prazos da página de FAQ entra em `emb_ppgc_pagina`, porque "quais os
requisitos do mestrado" é uma pergunta que se faz em linguagem natural.

Os PDFs dos editais
-------------------
`ppgc_documento` e `ppgc_documento_chunk` NÃO são truncadas. Os metadados dos
documentos entram por UPSERT e as colunas de extração (`texto_extraido`,
`n_paginas`, `sha256`, `extraido_em`) ficam intocadas. Um recrawl atualiza os
vínculos edital↔arquivo sem custar a reextração de centenas de PDFs.

Quando a ingestão de PDFs existir, o fluxo é:
    1. para cada linha de `ppgc_documento` com `texto_extraido IS NULL`,
       baixar o arquivo, extrair o texto, gravar `texto_extraido`/`n_paginas`/
       `sha256`/`extraido_em` e as linhas de `ppgc_documento_chunk`;
    2. `python load_ppgc.py --input ppgc.json --only-embeddings`
       → `emb_ppgc_documento` passa a existir, sem mudar nada no schema.
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
log = logging.getLogger("load_ppgc")

SCHEMA_FILE = Path(__file__).with_name("schema_ppgc.sql")

ORDEM = (
    "ppgc_crawl_meta", "ppgc_pagina", "ppgc_pagina_secao", "ppgc_post",
    "ppgc_post_tag", "ppgc_documento", "ppgc_edital", "ppgc_edital_documento",
    "ppgc_normativo", "ppgc_linha_pesquisa", "ppgc_docente", "ppgc_faq",
    "ppgc_requisito", "ppgc_disciplina", "ppgc_calendario_evento",
    "ppgc_defesa", "ppgc_link",
)

#: Tabelas que sobrevivem ao refresh. Ver docstring do módulo.
PRESERVAR = {
    "ppgc_documento": {
        "upsert_em": ["url"],
        "nao_sobrescrever": ["texto_extraido", "n_paginas", "sha256",
                             "extraido_em"],
    },
}

EMB_TABLES = {
    "emb_ppgc_edital":         "edital_id",
    "emb_ppgc_normativo":      "normativo_id",
    "emb_ppgc_faq":            "faq_id",
    "emb_ppgc_pagina":         "pagina_id",
    "emb_ppgc_post":           "post_id",
    "emb_ppgc_linha_pesquisa": "linha_slug",
    "emb_ppgc_disciplina":     "disciplina_id",
    "emb_ppgc_documento":      "documento_url",
}

#: Rótulos legíveis. O valor cru ('ingresso_regular') é bom para `WHERE` e
#: péssimo para embedding: o candidato escreve "seleção de mestrado", não
#: "ingresso_regular".
TIPO_EDITAL_NOME = {
    "ingresso_regular":    "seleção de estudante regular (mestrado/doutorado)",
    "ingresso_especial":   "seleção de estudante especial (disciplinas isoladas)",
    "bolsa_classificacao": "edital interno de classificação para bolsas",
    "bolsa_sanduiche":     "bolsa de doutorado sanduíche no exterior (PDSE/CAPES)",
    "bolsa_posdoc":        "bolsa de pós-doutorado",
    "professor_visitante": "seleção de professor visitante",
    "dinter":              "doutorado interinstitucional (DInter)",
    "resultado":           "resultado / homologação de processo seletivo",
    "outro":               "edital",
}

NIVEL_NOME = {
    "mestrado":  "Mestrado",
    "doutorado": "Doutorado",
    "ambos":     "Mestrado e Doutorado",
    "posdoc":    "Pós-doutorado",
    "docente":   "Docente",
}

TIPO_DOC_NOME = {
    "edital":      "Edital",
    "retificacao": "Retificação",
    "formulario":  "Formulário",
    "resultado":   "Resultado",
    "cronograma":  "Cronograma",
    "anexo":       "Anexo",
    "outro":       "Documento",
}

TIPO_NORMATIVO_NOME = {
    "resolucao":                "Resolução",
    "portaria":                 "Portaria",
    "regimento":                "Regimento",
    "planejamento_estrategico": "Planejamento Estratégico",
    "outro":                    "Norma",
}

ASSUNTO_NOME = {
    "edital":    "edital / processo seletivo",
    "defesa":    "defesa de dissertação ou tese",
    "bolsa":     "bolsas",
    "matricula": "matrícula e oferta de disciplinas",
    "avaliacao": "avaliação CAPES",
    "evento":    "evento acadêmico",
    "geral":     "geral",
}

PROGRAMA = ("PPGC — Programa de Pós-Graduação em Computação da UFPel")


class Indices:
    def __init__(self, dados: dict) -> None:
        self.pagina = {p["pagina_id"]: p for p in dados.get("ppgc_pagina", [])}
        self.docs_do_edital: dict[str, list[dict]] = {}
        for doc in dados.get("ppgc_edital_documento", []):
            self.docs_do_edital.setdefault(doc["edital_id"], []).append(doc)
        self.tags_do_post: dict[int, list[str]] = {}
        for pt in dados.get("ppgc_post_tag", []):
            self.tags_do_post.setdefault(pt["post_id"], []).append(pt["tag_slug"])


# ─────────────────────────────────────────────────────────────────────────────
# Construção dos textos vetorizados
# ─────────────────────────────────────────────────────────────────────────────

def montar_emb_edital(dados: dict, ix: Indices) -> list[dict]:
    """
    Um embedding por edital, já com a lista de anexos.

    A lista de documentos entra no TEXTO, não só na tabela, porque "saiu o
    resultado da seleção 2026/1?" é uma pergunta sobre a EXISTÊNCIA de um
    anexo. Com "Resultado Final do Processo Seletivo" dentro do chunk, o
    acerto vetorial já traz a evidência; sem ela, seria preciso um segundo
    passo relacional para descobrir que o resultado existe.
    """
    saida: list[dict] = []
    for ed in dados.get("ppgc_edital", []):
        docs = sorted(ix.docs_do_edital.get(ed["edital_id"], []),
                      key=lambda d: (d.get("ordem") or 0))
        lista_docs = [
            f"  - {TIPO_DOC_NOME.get(d.get('tipo_documento'), 'Documento')}: "
            f"{d.get('titulo') or d.get('url')}"
            for d in docs
        ]
        texto = juntar(
            f"Edital do {PROGRAMA}",
            rotulo("Título", ed["titulo"]),
            rotulo("Tipo", TIPO_EDITAL_NOME.get(ed.get("tipo"))),
            rotulo("Nível", NIVEL_NOME.get(ed.get("nivel"))),
            rotulo("Período letivo", ed.get("periodo_letivo")),
            rotulo("Número oficial do edital", ed.get("numero_oficial")),
            rotulo("Publicado em", ed.get("data_publicacao")),
            ("Documentos anexos:\n" + "\n".join(lista_docs)) if lista_docs else None,
            "",
            ed.get("texto") or ed.get("resumo"),
        )
        emitir(saida, "edital_id", ed["edital_id"], "edital", ed["titulo"],
               ed.get("url"), texto,
               {"tipo": "edital", "acervo": "ppgc",
                "edital_tipo": ed.get("tipo"), "nivel": ed.get("nivel"),
                "ano": ed.get("ano"), "semestre": ed.get("semestre"),
                "numero_oficial": ed.get("numero_oficial"),
                "n_documentos": len(docs)})
    return saida


def montar_emb_normativo(dados: dict) -> list[dict]:
    """
    Um embedding por norma, com a vigência ESCRITA no texto.

    O campo `vigente` na tabela resolve o `WHERE`, mas não protege a síntese:
    se o chunk recuperado não disser "REVOGADA", o modelo cita a norma como se
    valesse. Numa pergunta sobre regra de bolsa ou prazo de defesa, esse é o
    erro mais caro que este acervo pode cometer.
    """
    saida: list[dict] = []
    for nor in dados.get("ppgc_normativo", []):
        vigencia = ("VIGENTE" if nor.get("vigente") is True else
                    "REVOGADA — não use como regra atual"
                    if nor.get("vigente") is False else
                    "situação não confirmada no índice de normas do Programa")
        identificacao = juntar(
            TIPO_NORMATIVO_NOME.get(nor.get("tipo"), "Norma"),
            (f"{nor['numero']}/{nor['ano']}" if nor.get("numero") and nor.get("ano")
             else str(nor.get("ano") or "")),
            sep=" ",
        ).strip()
        texto = juntar(
            f"Norma do {PROGRAMA}",
            rotulo("Identificação", identificacao),
            rotulo("Situação", vigencia),
            rotulo("Ementa", nor.get("ementa") or nor.get("titulo")),
            "",
            nor.get("texto"),
        )
        emitir(saida, "normativo_id", nor["normativo_id"], "norma",
               identificacao or nor["normativo_id"],
               nor.get("url_pagina") or nor.get("url_pdf"), texto,
               {"tipo": "normativo", "acervo": "ppgc",
                "normativo_tipo": nor.get("tipo"), "ano": nor.get("ano"),
                "vigente": nor.get("vigente")})
    return saida


def montar_emb_faq(dados: dict, ix: Indices,
                   ids: Optional[dict[tuple, int]] = None) -> list[dict]:
    saida: list[dict] = []
    for i, faq in enumerate(dados.get("ppgc_faq", []), start=1):
        faq_id = ids.get((faq["pagina_id"], faq["ordem"])) if ids else i
        if faq_id is None:
            continue
        texto = juntar(
            f"FAQ de estudantes do {PROGRAMA}",
            rotulo("Contexto", faq.get("secao")),
            f"Pergunta: {faq['pergunta']}",
            f"Resposta: {faq['resposta']}",
        )
        emitir(saida, "faq_id", faq_id, "qa", faq["pergunta"], faq.get("url"),
               texto,
               {"tipo": "faq", "acervo": "ppgc", "secao_faq": faq.get("secao")})
    return saida


#: Categorias de página cujo conteúdo JÁ foi normalizado numa tabela própria,
#: com metadados melhores: `ppgc_edital` (tipo, nível, período, lista de PDFs),
#: `ppgc_normativo` (vigência), `ppgc_faq` (âncora por pergunta),
#: `ppgc_linha_pesquisa` e `ppgc_disciplina` (ementa).
#:
#: Vetorizar a página DE NOVO custa 304 dos 396 vetores (77%) e não acrescenta
#: nada: o índice fica com duas cópias quase idênticas do mesmo texto, que
#: ocupam as duas primeiras posições do top-k uma atrás da outra — foi
#: exatamente o que "o que a resolução diz sobre bolsas" devolveu. A cópia da
#: página é a pior das duas, porque não carrega tipo, vigência nem âncora.
_CATEGORIAS_JA_NORMALIZADAS = frozenset({
    "edital", "normativo", "linha_pesquisa", "disciplinas",
})


def montar_emb_pagina(dados: dict, ix: Indices) -> list[dict]:
    saida: list[dict] = []

    ignorar = {p["pagina_id"] for p in dados.get("ppgc_pagina", [])
               if p.get("categoria") in _CATEGORIAS_JA_NORMALIZADAS}

    # A página de FAQ é caso à parte: as seções que SÃO pergunta já estão em
    # `ppgc_faq`, mas as que não são — as tabelas "REQUISITOS PARA ALUNOS
    # REGULARES DE MESTRADO/DOUTORADO" — não estão em lugar nenhum vetorizado
    # (`ppgc_requisito` é servida por SQL). Excluir a página inteira apagaria
    # a única cópia semântica desses requisitos.
    paginas_faq = {p["pagina_id"] for p in dados.get("ppgc_pagina", [])
                   if p.get("categoria") == "faq"}

    n_secoes: dict[int, int] = {}
    for sec in dados.get("ppgc_pagina_secao", []):
        n_secoes[sec["pagina_id"]] = n_secoes.get(sec["pagina_id"], 0) + 1

    # ficha só para páginas com mais de uma seção — em página de seção única
    # ela seria uma quase-duplicata que ocupa duas vagas do top-k com o mesmo
    # conteúdo (ver o comentário equivalente em load_portal_computacao.py)
    for pag in dados.get("ppgc_pagina", []):
        if pag["pagina_id"] in ignorar or pag["pagina_id"] in paginas_faq:
            continue
        if n_secoes.get(pag["pagina_id"], 0) <= 1:
            continue
        texto = juntar(
            f"Página do {PROGRAMA}",
            rotulo("Título", pag["titulo"]),
            rotulo("Assunto", pag.get("categoria")),
            rotulo("Idioma", "inglês" if pag.get("idioma") == "en" else None),
            rotulo("Atualizada em", pag.get("data_modificacao")),
            "",
            pag.get("resumo"),
        )
        emitir(saida, "pagina_id", pag["pagina_id"], "ficha", pag["titulo"],
               pag.get("url"), texto,
               {"tipo": "pagina_ficha", "acervo": "ppgc",
                "categoria": pag.get("categoria"), "idioma": pag.get("idioma")})

    for sec in dados.get("ppgc_pagina_secao", []):
        pag = ix.pagina.get(sec["pagina_id"])
        if not pag or sec["pagina_id"] in ignorar:
            continue
        if (sec["pagina_id"] in paginas_faq
                and (sec.get("titulo_secao") or "").strip().endswith("?")):
            continue                    # já é um par pergunta/resposta
        titulo_completo = juntar(pag["titulo"], sec.get("titulo_secao"), sep=" — ")
        texto = juntar(
            f"{PROGRAMA}",
            rotulo("Página", pag["titulo"]),
            rotulo("Seção", sec.get("titulo_secao")),
            rotulo("Assunto", pag.get("categoria")),
            rotulo("Atualizada em", pag.get("data_modificacao")),
            "",
            sec.get("texto"),
        )
        emitir(saida, "pagina_id", sec["pagina_id"], f"secao#{sec['ordem']}",
               titulo_completo, sec.get("url"), texto,
               {"tipo": "pagina_secao", "acervo": "ppgc",
                "categoria": pag.get("categoria"),
                "titulo_secao": sec.get("titulo_secao"),
                "ancora": sec.get("ancora")})
    return saida


def montar_emb_post(dados: dict, ix: Indices) -> list[dict]:
    saida: list[dict] = []
    for post in dados.get("ppgc_post", []):
        texto = juntar(
            f"Notícia do {PROGRAMA}",
            rotulo("Título", post["titulo"]),
            rotulo("Publicada em", post.get("data_por_extenso")),
            rotulo("Assunto", ASSUNTO_NOME.get(post.get("assunto"))),
            rotulo("Palavras-chave", ix.tags_do_post.get(post["post_id"], [])),
            "",
            post.get("texto") or post.get("resumo"),
        )
        emitir(saida, "post_id", post["post_id"], "noticia", post["titulo"],
               post.get("url"), texto,
               {"tipo": "noticia", "acervo": "ppgc",
                "assunto": post.get("assunto"), "ano": post.get("ano"),
                "mes": post.get("mes"), "data": post.get("data_publicacao")})
    return saida


def montar_emb_linha(dados: dict) -> list[dict]:
    saida: list[dict] = []
    for linha in dados.get("ppgc_linha_pesquisa", []):
        texto = juntar(
            f"Linha de pesquisa do {PROGRAMA}",
            rotulo("Linha", linha["nome"]),
            "",
            linha.get("descricao"),
        )
        emitir(saida, "linha_slug", linha["slug"], "linha", linha["nome"],
               linha.get("url"), texto,
               {"tipo": "linha_pesquisa", "acervo": "ppgc",
                "linha": linha["nome"]})
    return saida


def montar_emb_disciplina(dados: dict,
                          ids: Optional[dict[tuple, int]] = None) -> list[dict]:
    saida: list[dict] = []
    for i, disc in enumerate(dados.get("ppgc_disciplina", []), start=1):
        chave = (disc["nome"], disc.get("ano"), disc.get("semestre"))
        disc_id = ids.get(chave) if ids else i
        if disc_id is None:
            continue
        especial = ("sim" if disc.get("aluno_especial") is True else
                    "não" if disc.get("aluno_especial") is False else None)
        texto = juntar(
            f"Disciplina do {PROGRAMA}",
            rotulo("Disciplina", disc["nome"]),
            rotulo("Créditos", disc.get("creditos")),
            rotulo("Oferta", (f"{disc['ano']}/{disc['semestre']}"
                              if disc.get("ano") and disc.get("semestre") else None)),
            rotulo("Responsável", disc.get("responsavel")),
            rotulo("Aberta a estudante especial", especial),
            rotulo("Horário", disc.get("horario")),
            rotulo("Modalidade", disc.get("modalidade")),
            "",
            rotulo("Ementa", disc.get("ementa")),
        )
        emitir(saida, "disciplina_id", disc_id, "disciplina", disc["nome"],
               disc.get("url"), texto,
               {"tipo": "disciplina", "acervo": "ppgc",
                "ano": disc.get("ano"), "semestre": disc.get("semestre"),
                "aluno_especial": disc.get("aluno_especial")})
    return saida


def montar_emb_documento(conn) -> list[dict]:
    """
    Chunks dos PDFs → embeddings. Lê do BANCO, não do JSON.

    O crawler não abre PDF nenhum; quem preenche `ppgc_documento_chunk` é a
    ingestão de documentos. Esta função existe desde já para que essa etapa
    seja só "gravar os chunks e rodar --only-embeddings".

    O cabeçalho do chunk identifica o edital de origem: um trecho de PDF sem
    dizer de que edital veio é inútil para citar — e perigoso, porque um
    critério de seleção de 2021 lido como se fosse de 2026 é uma resposta
    errada com aparência de certa.
    """
    if conn is None:
        return []
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.ppgc_documento_chunk')")
        if cur.fetchone()[0] is None:
            return []
        cur.execute("""
            SELECT c.url, c.ordem, c.pagina_pdf, c.texto,
                   d.titulo, d.data_upload,
                   ed.tipo_documento, e.titulo, e.periodo_letivo, e.numero_oficial
              FROM ppgc_documento_chunk c
              JOIN ppgc_documento d ON d.url = c.url
              LEFT JOIN LATERAL (
                    SELECT tipo_documento, edital_id
                      FROM ppgc_edital_documento x
                     WHERE x.url = c.url
                     ORDER BY x.ordem LIMIT 1
              ) ed ON TRUE
              LEFT JOIN ppgc_edital e ON e.edital_id = ed.edital_id
             ORDER BY c.url, c.ordem
        """)
        linhas = cur.fetchall()

    saida: list[dict] = []
    for (url, ordem, pagina_pdf, texto_chunk, titulo_doc, data_upload,
         tipo_doc, titulo_edital, periodo, numero) in linhas:
        texto = juntar(
            f"Documento do {PROGRAMA}",
            rotulo("Arquivo", titulo_doc),
            rotulo("Tipo", TIPO_DOC_NOME.get(tipo_doc)),
            rotulo("Edital", titulo_edital),
            rotulo("Período letivo", periodo),
            rotulo("Número oficial", numero),
            rotulo("Página do PDF", pagina_pdf),
            "",
            texto_chunk,
        )
        saida.append({
            "documento_url": url,
            "escopo": f"pdf#{ordem}",
            "titulo": titulo_doc or url,
            "url": url,
            "texto": texto,
            "metadata": {"tipo": "documento_pdf", "acervo": "ppgc",
                         "tipo_documento": tipo_doc, "edital": titulo_edital,
                         "periodo_letivo": periodo, "pagina_pdf": pagina_pdf,
                         "data_upload": str(data_upload) if data_upload else None},
        })
    return saida


def montar_todos(dados: dict, conn=None) -> dict[str, list[dict]]:
    ix = Indices(dados)
    ids_faq = (mapear_ids(conn, "ppgc_faq", "faq_id", ["pagina_id", "ordem"])
               if conn is not None else None)
    ids_disc = (mapear_ids(conn, "ppgc_disciplina", "disciplina_id",
                           ["nome", "ano", "semestre"])
                if conn is not None else None)
    return {
        "emb_ppgc_edital":         montar_emb_edital(dados, ix),
        "emb_ppgc_normativo":      montar_emb_normativo(dados),
        "emb_ppgc_faq":            montar_emb_faq(dados, ix, ids_faq),
        "emb_ppgc_pagina":         montar_emb_pagina(dados, ix),
        "emb_ppgc_post":           montar_emb_post(dados, ix),
        "emb_ppgc_linha_pesquisa": montar_emb_linha(dados),
        "emb_ppgc_disciplina":     montar_emb_disciplina(dados, ids_disc),
        "emb_ppgc_documento":      montar_emb_documento(conn),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Carga do acervo do PPGC (relacional + pgvector)")
    ap.add_argument("--input", default="ppgc.json")
    ap.add_argument("--schema", action="store_true",
                    help="aplica schema_ppgc.sql antes da carga")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-embeddings", action="store_true")
    ap.add_argument("--only-embeddings", action="store_true")
    ap.add_argument("--reset-documentos", action="store_true",
                    help="APAGA o texto já extraído dos PDFs e recarrega os "
                         "documentos do zero (por padrão a extração é preservada)")
    ap.add_argument("--batch", type=int, default=100)
    args = ap.parse_args()

    dados = carregar_json(Path(args.input))

    if args.dry_run:
        relatorio_dry_run(dados, ORDEM, montar_todos(dados))
        return

    preservar = {} if args.reset_documentos else PRESERVAR
    if args.reset_documentos:
        log.warning("--reset-documentos: o texto extraído dos PDFs será APAGADO")

    conn = connect()
    try:
        if args.schema:
            aplicar_schema(conn, SCHEMA_FILE)
        if not args.only_embeddings:
            carregar_relacional(conn, dados, ORDEM, preservar=preservar)
        if not args.skip_embeddings:
            registros = montar_todos(dados, conn)
            carregar_embeddings(conn, registros, EMB_TABLES, batch=args.batch)
    finally:
        conn.close()
    log.info("Carga do PPGC concluída")


if __name__ == "__main__":
    main()
