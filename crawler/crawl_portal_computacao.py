"""
Crawler do PORTAL da Computação-UFPel  (wp.ufpel.edu.br/computacao)
=============================================================================
Coleta o site institucional do curso — notícias e todo o conteúdo dos menus —
já no formato das tabelas de `schema_portal_computacao.sql`.

  Notícias                  todos os posts, com data, autor, categorias e tags
  Sobre a Computação-UFPel  história, servidores, identidade visual,
                            laboratórios, FAQ da Computação, localização
  Graduação                 CComp, EComp, Diretório Acadêmico, PET, Empresa
                            Júnior, Semana Acadêmica, prêmios SBC
  Pós-Graduação             apenas o índice (o conteúdo vai para crawl_ppgc.py)
  Pesquisa                  grupos de pesquisa
  Ensino / Extensão         ligas, projetos de extensão, IEEE
  Calendário Acadêmico      eventos do calendário Cobalto, dia a dia

O que NÃO entra aqui
--------------------
Tudo que `wp_common.pagina_e_ppgc` / `post_e_ppgc` classificam como PPGC. Esse
conteúdo é do `crawl_ppgc.py`, em tabelas próprias (`ppgc_*`). A partição é
exclusiva e feita pelas MESMAS funções nos dois crawlers, então nenhuma
página fica de fora nem entra duas vezes — `--relatorio-particao` imprime a
divisão para conferência.

Fonte: REST API do WordPress. O porquê e os cuidados com o WAF estão em
`wp_common.py`.

Uso
---
    python crawl_portal_computacao.py                       # tudo
    python crawl_portal_computacao.py --output portal.json
    python crawl_portal_computacao.py --cache-dir .cache_wp # reusa respostas
    python crawl_portal_computacao.py --relatorio-particao  # portal × PPGC

Requisitos: pip install -r requirements_crawler.txt
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wp_common import (                                        # noqa: E402
    API_URL, SITE_URL, Dataset, WPClient, anos_citados, ano_semestre_de,
    caminho_da_pagina, contar_palavras, data_por_extenso,
    dividir_secoes, e_subtitulo, extrair_links, html_para_texto, idioma_de,
    iso, limpar_texto, MES_POR_NOME_NORM, nome_arquivo_de, normalizar,
    pagina_e_ppgc, parse_data_wp, post_e_ppgc, resumo_de, semestre_de,
    titulo_wp,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("crawl_portal")

DEFAULT_OUTPUT = "portal_computacao.json"

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PortalDataset(Dataset):
    KEYS = {
        "port_crawl_meta":       ("chave",),
        "port_secao":            ("secao_slug",),
        "port_autor":            ("autor_id",),
        "port_categoria":        ("categoria_id",),
        "port_tag":              ("tag_id",),
        "port_pagina":           ("pagina_id",),
        "port_menu_item":        ("item_id",),
        "port_pagina_secao":     ("pagina_id", "ordem"),
        "port_post":             ("post_id",),
        "port_post_categoria":   ("post_id", "categoria_id"),
        "port_post_tag":         ("post_id", "tag_id"),
        "port_link":             ("origem_tipo", "origem_id", "ordem"),
        "port_documento":        ("url",),
        "port_faq":              ("pagina_id", "ordem"),
        "port_pessoa":           ("pagina_id", "nome"),
        "port_grupo_pesquisa":   ("nome",),
        "port_calendario_evento": ("pagina_id", "data", "descricao"),
    }
    ORDER = (
        "port_crawl_meta", "port_secao", "port_autor", "port_categoria",
        "port_tag", "port_pagina", "port_menu_item", "port_pagina_secao",
        "port_post", "port_post_categoria", "port_post_tag", "port_link",
        "port_documento", "port_faq", "port_pessoa", "port_grupo_pesquisa",
        "port_calendario_evento",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Menu do site — a estrutura que o enunciado chama de "seções"
# ─────────────────────────────────────────────────────────────────────────────

#: Seções de primeiro nível, na ordem do menu. `secao_slug` é a chave que
#: viaja para `port_pagina.secao_slug` e `port_post.secao_slug`, e é o filtro
#: mais barato que o roteador do RAG tem ("só o que está em Extensão").
SECOES_PADRAO: tuple[tuple[str, str], ...] = (
    ("noticias",   "Notícias"),
    ("sobre",      "Sobre a Computação-UFPel"),
    ("graduacao",  "Graduação"),
    ("pos-graduacao", "Pós-Graduação"),
    ("pesquisa",   "Pesquisa"),
    ("ensino",     "Ensino"),
    ("extensao",   "Extensão"),
    ("calendario", "Calendário Acadêmico"),
)

#: Título do menu → secao_slug. Casado por texto normalizado porque o menu é
#: HTML e não expõe id estável de seção.
_TITULO_PARA_SECAO = {
    "noticias": "noticias",
    "sobre a computacao-ufpel": "sobre",
    "graduacao": "graduacao",
    "pos-graduacao": "pos-graduacao",
    "pesquisa": "pesquisa",
    "ensino": "ensino",
    "extensao": "extensao",
    "calendario academico": "calendario",
}


def coletar_menu(cliente: WPClient, ds: PortalDataset,
                 pagina_por_url: dict[str, int],
                 ids_portal: set[int]) -> dict[int, str]:
    """
    Lê o menu do cabeçalho e devolve `{pagina_id: secao_slug}`.

    Por que o menu e não a árvore de páginas: as duas discordam, e o menu é a
    que o estudante enxerga. "Liga Acadêmica de Robótica" está sob Graduação
    *e* sob Ensino no menu; "Ciência da Computação" é filha de Graduação no
    menu mas é página raiz na árvore. A árvore de páginas continua guardada em
    `port_pagina.caminho` — as duas visões convivem, cada uma respondendo o
    que sabe responder.

    O endpoint `/wp/v2/menus` exige autenticação (401), então a fonte é o HTML
    do `<nav id="nav-header">`, com a marcação padrão de menu do WordPress
    (`li.menu-item-<ID>` + `ul.sub-menu`).
    """
    html = cliente.get_texto(SITE_URL + "/")
    soup = BeautifulSoup(html, "lxml")
    nav = soup.select_one("nav#nav-header") or soup.select_one("nav#nav-mobile")
    if nav is None:
        log.warning("[Menu] <nav> do cabeçalho não encontrado — seções ficam só "
                    "com o mapeamento por caminho")
        for ordem, (slug, nome) in enumerate(SECOES_PADRAO):
            ds.add("port_secao", {"secao_slug": slug, "nome": nome,
                                  "ordem": ordem, "url": None})
        return {}

    for ordem, (slug, nome) in enumerate(SECOES_PADRAO):
        ds.add("port_secao", {"secao_slug": slug, "nome": nome, "ordem": ordem,
                              "url": None})

    secao_por_pagina: dict[int, str] = {}
    raiz = nav.find("ul")
    if raiz is None:
        return secao_por_pagina

    def id_do_item(li: Tag) -> Optional[int]:
        for classe in li.get("class") or []:
            m = re.fullmatch(r"menu-item-(\d+)", classe)
            if m:
                return int(m.group(1))
        return None

    def caminhar(ul: Tag, secao_slug: Optional[str], pai: Optional[int],
                 nivel: int) -> None:
        for ordem, li in enumerate(ul.find_all("li", recursive=False)):
            a = li.find("a", href=True)
            if a is None:
                continue
            titulo = limpar_texto(a.get_text(" ", strip=True))
            url = (a["href"] or "").strip()
            if not titulo:
                continue
            item_id = id_do_item(li)
            if item_id is None:
                continue

            slug_atual = secao_slug
            if nivel == 0:
                slug_atual = _TITULO_PARA_SECAO.get(normalizar(titulo))
                if slug_atual:
                    ds.add("port_secao", {
                        "secao_slug": slug_atual,
                        "nome": titulo,
                        "ordem": next((i for i, (s, _) in enumerate(SECOES_PADRAO)
                                       if s == slug_atual), ordem),
                        "url": url,
                    })

            pagina_id = pagina_por_url.get(url.rstrip("/") + "/")
            if pagina_id and slug_atual and pagina_id not in secao_por_pagina:
                secao_por_pagina[pagina_id] = slug_atual

            interno = urlparse(url).netloc.endswith("wp.ufpel.edu.br")
            ds.add("port_menu_item", {
                "item_id": item_id,
                "secao_slug": slug_atual,
                "parent_item_id": pai,
                "ordem": ordem,
                "nivel": nivel,
                "titulo": titulo,
                "url": url,
                "alvo_tipo": ("pagina" if pagina_id else
                              "interno" if interno else "externo"),
                "pagina_id": pagina_id,
                # 6 itens do menu ("Pós-Graduação", "Mestrado", "Doutorado",
                # "DInter", "Calendário PPGC"…) apontam para páginas que vivem
                # no dataset do PPGC. O menu tem de continuar apontando para
                # elas — é o menu real do site —, então `pagina_id` NÃO tem FK
                # e esta coluna diz em qual dataset a página está.
                "pagina_dataset": (None if not pagina_id else
                                   "portal" if pagina_id in ids_portal
                                   else "ppgc"),
            })

            sub = li.find("ul", recursive=False)
            if sub is not None:
                caminhar(sub, slug_atual, item_id, nivel + 1)

    caminhar(raiz, None, None, 0)
    log.info("[Menu] %d itens, %d páginas com seção atribuída",
             len(ds.tables["port_menu_item"]), len(secao_por_pagina))
    return secao_por_pagina


#: Prefixo de caminho → seção, para as páginas que não estão no menu.
#: O menu cobre 40 das 191 páginas; o resto (subpáginas, editais antigos)
#: precisa de um fallback ou ficaria sem filtro de seção nenhum.
#: A segunda metade é a tradução em inglês (Polylang). O menu do site só
#: existe em português, então TODA página em inglês dependeria deste fallback;
#: sem ele, um terço das páginas do portal ficaria sem filtro de seção.
_CAMINHO_PARA_SECAO: tuple[tuple[str, str], ...] = (
    ("sobre-a-computacao-ufpel", "sobre"),
    ("graduacao", "graduacao"),
    ("ciencia-da-computacao", "graduacao"),
    ("engenharia-de-computacao", "graduacao"),
    ("pos-graduacao", "pos-graduacao"),
    ("ppgc", "pos-graduacao"),
    ("pesquisa", "pesquisa"),
    ("ensino", "ensino"),
    ("extensao", "extensao"),
    ("calendario", "calendario"),
    # inglês
    ("graduation", "graduacao"),
    ("computer-science", "graduacao"),
    ("computer-engineering", "graduacao"),
    ("research", "pesquisa"),
    ("education", "ensino"),
    ("extension", "extensao"),
    ("computing-course-history", "sobre"),
    ("faculty-and-technical-administrative", "sobre"),
    ("visual-identity", "sobre"),
    ("location-and-contacts", "sobre"),
    ("laboratory-schedule", "sobre"),
)


def secao_por_caminho(caminho: str) -> Optional[str]:
    """
    Seção a partir da raiz do caminho hierárquico.

    Compara a raiz INTEIRA, não por prefixo solto: "research-groups" não pode
    virar "pesquisa" por acidente de string — ele chega aqui como
    `research/research-groups`, cuja raiz já é "research".
    """
    raiz = normalizar(caminho).split("/", 1)[0]
    for prefixo, secao in _CAMINHO_PARA_SECAO:
        if raiz == prefixo:
            return secao
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Parsers específicos
# ─────────────────────────────────────────────────────────────────────────────

_RE_DIA = re.compile(r"^\s*(\d{1,2})\s+(\S+)")


def parsear_calendario(pagina: dict, texto_html: str) -> list[dict]:
    """
    Calendário acadêmico (`div.cobalto-calendario`) → uma linha por evento.

    O WordPress embute aqui o widget do Cobalto: um cabeçalho de mês
    ("Fevereiro / 2022") seguido de linhas com bolinha colorida (o `title` do
    span carrega o nome do calendário — "Feriados e pontos facultativos",
    "Calendário Acadêmico - Cursos semestrais"), dia + dia da semana, e a
    descrição.

    Vale normalizar em vez de deixar como texto corrido porque a pergunta
    típica é factual e datada: "quando começa o período de matrícula?",
    "quais os feriados de setembro?". Com data em coluna DATE isso é um
    `WHERE`, não uma busca vetorial.
    """
    soup = BeautifulSoup(texto_html, "lxml")
    linhas = soup.select("div.cobalto-calendario-linha")
    if not linhas:
        return []

    eventos: list[dict] = []
    mes_atual: Optional[int] = None
    ano_atual: Optional[int] = None
    ano_pagina, _ = ano_semestre_de(pagina["slug"], titulo_wp(pagina["title"]["rendered"]))

    for no in soup.select("div.cobalto-calendario-mes, div.cobalto-calendario-linha"):
        classes = no.get("class") or []
        if "cobalto-calendario-mes" in classes:
            rotulo = limpar_texto(no.get_text(" ", strip=True))
            m = re.match(r"([A-Za-zÀ-ÿ]+)\s*/\s*(\d{4})", rotulo)
            if m:
                mes_atual = MES_POR_NOME_NORM.get(normalizar(m.group(1)))
                ano_atual = int(m.group(2))
            continue

        dia_txt = limpar_texto(
            (no.select_one("span.cobalto-calendario-dia") or no).get_text(" ", strip=True))
        descricao = limpar_texto(
            (no.select_one("span.cobalto-calendario-descricao") or no).get_text(" ", strip=True))
        tipo_span = no.select_one("span.cobalto-calendario-tipo")
        tipo = limpar_texto(tipo_span.get("title")) if tipo_span else None

        m = _RE_DIA.match(dia_txt)
        if not m or mes_atual is None or not descricao:
            continue
        dia, dia_semana = int(m.group(1)), m.group(2)
        ano = ano_atual or ano_pagina
        try:
            data = datetime(ano, mes_atual, dia).date()
        except (TypeError, ValueError):
            continue

        eventos.append({
            "pagina_id": pagina["id"],
            "data": data.isoformat(),
            "ano": data.year,
            "mes": data.month,
            "dia_semana": dia_semana,
            "descricao": descricao,
            "tipo": tipo,
            "url": pagina["link"],
        })
    return eventos


_RE_PERGUNTA = re.compile(r"^\s*pergunta\s*:?\s*", re.I)
_RE_RESPOSTA = re.compile(r"^\s*resposta\s*:?\s*", re.I)


def parsear_faq(pagina: dict, html: str) -> list[dict]:
    """
    Página de FAQ → pares (pergunta, resposta), um por linha.

    Essa é a decisão de granularidade que mais rende no dataset inteiro. As
    duas páginas de FAQ somam ~86 mil caracteres. Vetorizadas inteiras, "como
    peço segunda chamada?" competiria com estágio, e-mail institucional e TCC
    dentro do mesmo vetor. Quebradas em Q&A, o acerto é cirúrgico — e a
    resposta já vem com a âncora para citar a fonte exata.

    Os dois FAQs do site têm marcações DIFERENTES, e por isso há duas
    estratégias:

      A. FAQ da Computação (graduação) — parágrafos planos "Pergunta: …" /
         "Resposta: …" sob títulos de assunto em negrito.
      B. FAQ de Alunos (PPGC) — `<a name="prim-mat">` seguido da pergunta em
         negrito ("COMO REALIZAR A PRIMEIRA MATRÍCULA?") e da resposta em
         lista; nenhum rótulo "Pergunta"/"Resposta" em lugar nenhum.

    A tentativa A roda primeiro e, se não achar nada, cai para B. Vale a pena
    manter as duas: aplicar só B ao FAQ da graduação juntaria pergunta e
    resposta num bloco só, e aplicar só A ao FAQ do PPGC devolveria zero.
    """
    faqs = _parsear_faq_pergunta_resposta(pagina, html)
    return faqs if faqs else _parsear_faq_por_ancora(pagina, html)


def _parsear_faq_por_ancora(pagina: dict, html: str) -> list[dict]:
    """
    Estratégia B: a pergunta É o título da seção, e termina em "?".

    Reaproveita `dividir_secoes`, que já sabe tratar `<a name>` como âncora e
    parágrafo curto em negrito como título. Títulos que não são pergunta
    ("Mestrado e Doutorado", "REQUISITOS PARA ALUNOS REGULARES DE MESTRADO")
    viram o assunto das perguntas seguintes.
    """
    faqs: list[dict] = []
    assunto: Optional[str] = None
    for secao in dividir_secoes(html):
        titulo = (secao.get("titulo") or "").strip()
        if not titulo:
            continue
        if not titulo.endswith("?"):
            assunto = titulo
            continue
        resposta = (secao.get("texto") or "").strip()
        if not resposta:
            continue
        faqs.append({
            "pagina_id": pagina["id"],
            "ordem": len(faqs),
            "secao": assunto,
            "ancora": secao.get("ancora"),
            "pergunta": titulo,
            "resposta": resposta,
            "url": (pagina["link"] +
                    (f"#{secao['ancora']}" if secao.get("ancora") else "")),
        })
    return faqs


def _parsear_faq_pergunta_resposta(pagina: dict, html: str) -> list[dict]:
    """Estratégia A: parágrafos rotulados "Pergunta:" / "Resposta:"."""
    soup = BeautifulSoup(html, "lxml")
    faqs: list[dict] = []
    secao_atual: Optional[str] = None
    ancora_atual: Optional[str] = None
    pergunta: Optional[str] = None
    resposta: list[str] = []

    def fechar() -> None:
        nonlocal pergunta, resposta
        if pergunta and resposta:
            faqs.append({
                "pagina_id": pagina["id"],
                "ordem": len(faqs),
                "secao": secao_atual,
                "ancora": ancora_atual,
                "pergunta": pergunta,
                "resposta": "\n".join(resposta).strip(),
                "url": (pagina["link"] +
                        (f"#{ancora_atual}" if ancora_atual else "")),
            })
        pergunta, resposta = None, []

    for no in (soup.body or soup).find_all(
            ["p", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6"],
            recursive=True):
        if no.find_parent(["li", "ul", "ol", "table"]):
            continue                                   # já consumido pelo pai
        texto = limpar_texto(no.get_text(" ", strip=True))
        if not texto:
            continue

        # Cabeçalho de assunto: <h_> ou parágrafo curto todo em negrito.
        # A condição não pode exigir `id`: só 4 dos 9 assuntos do FAQ da
        # Computação têm âncora, e sem esta regra as perguntas de "TCC" e
        # "ATIVIDADES COMPLEMENTARES" herdariam a seção "MATRÍCULA" —
        # metadado errado é pior que metadado ausente na hora de citar a fonte.
        if no.name.startswith("h") or (
                no.name == "p" and e_subtitulo(no)
                and not _RE_PERGUNTA.match(texto) and not _RE_RESPOSTA.match(texto)):
            fechar()
            secao_atual = texto.rstrip(":")
            ancora_atual = no.get("id")
            continue

        if _RE_PERGUNTA.match(texto):
            fechar()
            pergunta = _RE_PERGUNTA.sub("", texto).strip()
        elif _RE_RESPOSTA.match(texto):
            resposta.append(_RE_RESPOSTA.sub("", texto).strip())
        elif pergunta:
            resposta.append(html_para_texto(str(no)) if no.name in ("ul", "ol", "table")
                            else texto)

    fechar()
    return faqs


#: URL do portal institucional → id do servidor. É a MESMA chave da tabela
#: `servidor` de schema_computacao.sql: quem já carregou o dataset
#: institucional consegue juntar o nome que aparece aqui com titulação,
#: projetos e disciplinas ministradas de lá.
_RE_SERVIDOR_ID = re.compile(r"institucional\.ufpel\.edu\.br/servidores/id/(\d+)")
_RE_LATTES = re.compile(r"(lattes\.cnpq\.br|buscatextual\.cnpq\.br)", re.I)


def parsear_pessoas(pagina: dict, html: str) -> list[dict]:
    """
    Página de servidores/docentes → uma linha por pessoa.

    A categoria (docente / técnico-administrativo / aposentado) vem do
    parágrafo em negrito que abre cada bloco, e o setor vem do prefixo do
    próprio item ("Secretaria PPGC: Fulana"). Guardamos os dois separados
    porque "quem é a secretária do PPGC?" e "quais são os docentes?" são
    perguntas diferentes sobre a mesma lista.
    """
    soup = BeautifulSoup(html, "lxml")
    pessoas: list[dict] = []
    categoria: Optional[str] = None
    setor: Optional[str] = None

    def classificar(rotulo: str) -> Optional[str]:
        n = normalizar(rotulo)
        if "tecnico" in n:
            return "tecnico_administrativo"
        if "aposentad" in n:
            return "aposentado"
        if "docente" in n or "professor" in n:
            return "docente"
        return None

    for no in (soup.body or soup).descendants:
        if not isinstance(no, Tag):
            continue
        if no.name in ("strong", "b", "h1", "h2", "h3", "h4", "p"):
            if no.find("a", href=True):
                continue
            nova = classificar(limpar_texto(no.get_text(" ", strip=True)))
            if nova:
                categoria, setor = nova, None
            continue
        if no.name != "li":
            continue

        links = no.find_all("a", href=True)
        texto_li = limpar_texto(no.get_text(" ", strip=True))
        if not links:
            if ":" in texto_li and len(texto_li) < 120:
                setor = texto_li.split(":", 1)[0].strip()
            continue

        # "Secretaria PPGC: Fulana" — prefixo antes do link é o setor
        prefixo = texto_li.split(":", 1)[0] if ":" in texto_li else ""
        setor_item = (prefixo.strip()
                      if prefixo and prefixo.lower() not in
                      normalizar(links[0].get_text(" ", strip=True))
                      and len(prefixo) < 80 else setor)

        principal = links[0]
        nome = limpar_texto(principal.get_text(" ", strip=True))
        if not nome or len(nome) < 4:
            continue

        hrefs = [a["href"] for a in links if a.get("href")]
        servidor_id = next(
            (m.group(1) for h in hrefs if (m := _RE_SERVIDOR_ID.search(h))), None)
        lattes = next((h for h in hrefs if _RE_LATTES.search(h)), None)
        perfil = next((h for h in hrefs
                       if not _RE_LATTES.search(h)
                       and not _RE_SERVIDOR_ID.search(h)), None)

        pessoas.append({
            "pagina_id": pagina["id"],
            "nome": nome,
            "idioma": idioma_de(pagina.get("link"), pagina.get("slug")),
            "categoria": categoria or "docente",
            "setor": setor_item,
            "servidor_id": servidor_id,
            "lattes_url": lattes,
            "url_perfil": perfil,
            "url_origem": pagina["link"],
        })
    return pessoas


_RE_SIGLA = re.compile(r"\(([A-Za-zÀ-ÿ0-9\-]{2,12})\)\s*$")


def parsear_grupos(pagina: dict, html: str) -> list[dict]:
    """Grupos de pesquisa → nome, sigla (do parêntese final) e URL."""
    soup = BeautifulSoup(html, "lxml")
    grupos: list[dict] = []
    for li in soup.find_all("li"):
        a = li.find("a", href=True)
        if a is None:
            continue
        nome = limpar_texto(a.get_text(" ", strip=True))
        if len(nome) < 6:
            continue
        m = _RE_SIGLA.search(nome)
        grupos.append({
            "nome": nome,
            "idioma": idioma_de(pagina.get("link"), pagina.get("slug")),
            "sigla": m.group(1) if m else None,
            "url": a["href"],
            "pagina_id": pagina["id"],
            "url_origem": pagina["link"],
        })
    return grupos


# ─────────────────────────────────────────────────────────────────────────────
# Montagem das tabelas
# ─────────────────────────────────────────────────────────────────────────────

def registrar_links_e_documentos(ds: Dataset, origem_tipo: str, origem_id: int,
                                 origem_titulo: str, origem_url: str,
                                 html: str, media_por_url: dict[str, dict],
                                 tabela_link: str = "port_link",
                                 tabela_doc: Optional[str] = "port_documento") -> int:
    """
    Grava os links do conteúdo e promove os anexos a `*_documento`.

    Um "documento" é o link para PDF/DOC/planilha. Ele ganha tabela própria
    (e não só uma linha em `port_link`) por dois motivos: o texto do link é o
    melhor rótulo que existe para o arquivo ("Retificação 1 do Edital"), e é
    esta tabela que a ingestão futura de PDFs vai usar como fila de trabalho —
    `texto_extraido` nasce NULL e é preenchido depois.

    `tabela_doc=None` grava só os links: é o que o crawler do PPGC faz, porque
    lá os PDFs são registrados por `_registrar_edital`, que sabe a qual edital
    e a qual seção cada arquivo pertence — informação que se perderia numa
    tabela genérica de documentos.
    """
    n_docs = 0
    for link in extrair_links(html, origem_url or SITE_URL):
        ds.add(tabela_link, {
            "origem_tipo": origem_tipo,
            "origem_id": origem_id,
            "ordem": link["ordem"],
            "url": link["url"],
            "texto": link["texto"],
            "host": link["host"],
            "tipo": link["tipo"],
            "extensao": link["extensao"],
        })
        if tabela_doc is None or link["tipo"] not in ("documento", "planilha"):
            continue
        media = media_por_url.get(link["url"]) or {}
        data_media = parse_data_wp(media.get("date"))
        ds.add(tabela_doc, {
            "url": link["url"],
            "nome_arquivo": nome_arquivo_de(link["url"]),
            "titulo": link["texto"] or titulo_wp((media.get("title") or {}).get("rendered")),
            "extensao": link["extensao"],
            "mime": media.get("mime_type"),
            "media_id": media.get("id"),
            "data_upload": iso(data_media.date()) if data_media else None,
            "origem_tipo": origem_tipo,
            "origem_id": origem_id,
            "origem_titulo": origem_titulo,
            "origem_url": origem_url,
            "texto_link": link["texto"],
        })
        n_docs += 1
    return n_docs


def escopo_curso_de(slugs: Iterable[str]) -> str:
    """ccomp | ecomp | ambos | geral — a partir das categorias do post."""
    slugs = set(slugs)
    tem_cc, tem_ec = "ccomp" in slugs, "ecomp" in slugs
    if tem_cc and tem_ec:
        return "ambos"
    if tem_cc:
        return "ccomp"
    if tem_ec:
        return "ecomp"
    return "geral"


def montar(cliente: WPClient, *, relatorio_particao: bool = False) -> PortalDataset:
    ds = PortalDataset()

    log.info("[1/6] Taxonomias, autores e mídia")
    categorias = cliente.fetch_all("categories", hide_empty="false")
    tags = cliente.fetch_all("tags", hide_empty="false")
    autores = cliente.fetch_all("users")
    media = cliente.fetch_all("media")

    slug_por_categoria = {c["id"]: c["slug"] for c in categorias}
    for c in categorias:
        ds.add("port_categoria", {
            "categoria_id": c["id"], "slug": c["slug"],
            "nome": titulo_wp(c["name"]),
            "descricao": limpar_texto(c.get("description")) or None,
            "total_posts": c.get("count"), "parent_id": c.get("parent") or None,
        })
    for t in tags:
        ds.add("port_tag", {
            "tag_id": t["id"], "slug": t["slug"], "nome": titulo_wp(t["name"]),
            "total_posts": t.get("count"),
        })
    for u in autores:
        ds.add("port_autor", {
            "autor_id": u["id"], "nome": titulo_wp(u.get("name")),
            "slug": u.get("slug"),
            "descricao": limpar_texto(u.get("description")) or None,
            "url": u.get("link"),
        })

    media_por_url = {m["source_url"]: m for m in media if m.get("source_url")}

    log.info("[2/6] Páginas")
    paginas = cliente.fetch_all("pages", status="publish")
    por_id = {p["id"]: p for p in paginas}
    caminhos = {p["id"]: caminho_da_pagina(p, por_id) for p in paginas}

    paginas_portal = [p for p in paginas
                      if not pagina_e_ppgc(caminhos[p["id"]], p["slug"],
                                           titulo_wp(p["title"]["rendered"]))]
    ids_portal = {p["id"] for p in paginas_portal}
    log.info("      %d páginas no total → %d no portal, %d no PPGC",
             len(paginas), len(paginas_portal), len(paginas) - len(paginas_portal))

    log.info("[3/6] Menu do cabeçalho")
    pagina_por_url = {p["link"]: p["id"] for p in paginas}
    secao_por_pagina = coletar_menu(cliente, ds, pagina_por_url, ids_portal)

    log.info("[4/6] Conteúdo das páginas")
    n_faq = n_cal = n_pessoa = n_grupo = 0
    for p in paginas_portal:
        html = p["content"]["rendered"]
        texto = html_para_texto(html)
        caminho = caminhos[p["id"]]
        pai = p.get("parent") or None
        publicado = parse_data_wp(p.get("date"))
        modificado = parse_data_wp(p.get("modified"))
        titulo = titulo_wp(p["title"]["rendered"])

        ds.add("port_pagina", {
            "pagina_id": p["id"],
            "slug": p["slug"],
            "caminho": caminho,
            "titulo": titulo,
            "url": p["link"],
            "pagina_pai_id": pai if pai in ids_portal else None,
            "caminho_pai": caminho.rsplit("/", 1)[0] if "/" in caminho else None,
            "nivel": caminho.count("/"),
            "secao_slug": (secao_por_pagina.get(p["id"])
                           or secao_por_caminho(caminho)),
            "idioma": idioma_de(p["link"], p["slug"]),
            "ordem_menu": p.get("menu_order"),
            "data_publicacao": iso(publicado.date()) if publicado else None,
            "data_modificacao": iso(modificado.date()) if modificado else None,
            "resumo": resumo_de(texto),
            "texto": texto or None,
            "n_palavras": contar_palavras(texto),
            "template": p.get("template") or None,
        })

        for ordem, secao in enumerate(dividir_secoes(html)):
            if not (secao["texto"] or "").strip():
                continue
            ds.add("port_pagina_secao", {
                "pagina_id": p["id"],
                "ordem": ordem,
                "titulo_secao": secao["titulo"],
                "nivel_titulo": secao["nivel"] or None,
                "ancora": secao["ancora"],
                "texto": secao["texto"],
                "n_palavras": contar_palavras(secao["texto"]),
                "url": p["link"] + (f"#{secao['ancora']}" if secao["ancora"] else ""),
            })

        registrar_links_e_documentos(ds, "pagina", p["id"], titulo, p["link"],
                                     html, media_por_url)

        slug_norm = normalizar(p["slug"])
        if "faq" in slug_norm:
            for faq in parsear_faq(p, html):
                ds.add("port_faq", faq)
                n_faq += 1
        if "cobalto-calendario" in html:
            for evento in parsear_calendario(p, html):
                ds.add("port_calendario_evento", evento)
                n_cal += 1
        if slug_norm in ("servidores", "docentes", "faculty-and-technical-administrative"):
            for pessoa in parsear_pessoas(p, html):
                ds.add("port_pessoa", pessoa)
                n_pessoa += 1
        if slug_norm in ("grupos", "research-groups"):
            for grupo in parsear_grupos(p, html):
                ds.add("port_grupo_pesquisa", grupo)
                n_grupo += 1

    log.info("      FAQ %d · eventos de calendário %d · pessoas %d · grupos %d",
             n_faq, n_cal, n_pessoa, n_grupo)

    log.info("[5/6] Posts (notícias)")
    posts = cliente.fetch_all("posts", status="publish")
    posts_portal = [p for p in posts if not post_e_ppgc(p, slug_por_categoria)]
    log.info("      %d posts no total → %d no portal, %d no PPGC",
             len(posts), len(posts_portal), len(posts) - len(posts_portal))

    for p in posts_portal:
        html = p["content"]["rendered"]
        texto = html_para_texto(html)
        publicado = parse_data_wp(p.get("date"))
        modificado = parse_data_wp(p.get("modified"))
        data_pub = publicado.date() if publicado else None
        titulo = titulo_wp(p["title"]["rendered"])
        cat_slugs = [slug_por_categoria.get(c) for c in (p.get("categories") or [])]

        n_docs = registrar_links_e_documentos(
            ds, "post", p["id"], titulo, p["link"], html, media_por_url)

        ds.add("port_post", {
            "post_id": p["id"],
            "slug": p["slug"],
            "titulo": titulo,
            "url": p["link"],
            "secao_slug": "noticias",
            "data_publicacao": iso(data_pub),
            "publicado_em": iso(publicado),
            "ano": data_pub.year if data_pub else None,
            "mes": data_pub.month if data_pub else None,
            "semestre": semestre_de(data_pub),
            "data_por_extenso": data_por_extenso(data_pub),
            "data_modificacao": iso(modificado.date()) if modificado else None,
            "autor_id": p.get("author") or None,
            "escopo_curso": escopo_curso_de(s for s in cat_slugs if s),
            "resumo": (limpar_texto(html_para_texto(p["excerpt"]["rendered"]))
                       or resumo_de(texto)),
            "texto": texto or None,
            "n_palavras": contar_palavras(texto),
            "imagem_destaque_url": p.get("jetpack_featured_media_url") or None,
            "n_documentos": n_docs,
            "anos_citados": anos_citados(titulo, texto) or None,
            "url_curta": p.get("jetpack_shortlink") or None,
        })
        for cid in p.get("categories") or []:
            ds.add("port_post_categoria", {"post_id": p["id"], "categoria_id": cid})
        for tid in p.get("tags") or []:
            ds.add("port_post_tag", {"post_id": p["id"], "tag_id": tid})

    log.info("[6/6] Metadados do crawl")
    ds.add("port_crawl_meta", {"chave": "fonte", "valor": SITE_URL})
    ds.add("port_crawl_meta", {"chave": "api", "valor": API_URL})
    ds.add("port_crawl_meta", {"chave": "coletado_em",
                               "valor": datetime.now().astimezone().isoformat()})
    ds.add("port_crawl_meta", {"chave": "requisicoes",
                               "valor": str(cliente.n_requisicoes)})
    ds.add("port_crawl_meta", {"chave": "posts_totais", "valor": str(len(posts))})
    ds.add("port_crawl_meta", {"chave": "posts_portal", "valor": str(len(posts_portal))})
    ds.add("port_crawl_meta", {"chave": "paginas_totais", "valor": str(len(paginas))})
    ds.add("port_crawl_meta", {"chave": "paginas_portal",
                               "valor": str(len(paginas_portal))})

    if relatorio_particao:
        _relatorio_particao(paginas, caminhos, posts, slug_por_categoria)
    return ds


def _relatorio_particao(paginas: list[dict], caminhos: dict[int, str],
                        posts: list[dict], slug_por_categoria: dict[int, str]) -> None:
    """Imprime a divisão portal × PPGC para conferência manual."""
    print("\n── PÁGINAS ─────────────────────────────────────────────────────────")
    for p in sorted(paginas, key=lambda x: caminhos[x["id"]]):
        titulo = titulo_wp(p["title"]["rendered"])
        destino = "PPGC  " if pagina_e_ppgc(caminhos[p["id"]], p["slug"], titulo) \
            else "portal"
        print(f"  {destino}  {p['id']:>5}  {caminhos[p['id']]}")
    print("\n── POSTS por ano ───────────────────────────────────────────────────")
    from collections import Counter
    portal_c, ppgc_c = Counter(), Counter()
    for p in posts:
        ano = (p.get("date") or "")[:4]
        (ppgc_c if post_e_ppgc(p, slug_por_categoria) else portal_c)[ano] += 1
    for ano in sorted(set(portal_c) | set(ppgc_c)):
        print(f"  {ano}   portal {portal_c[ano]:>4}   PPGC {ppgc_c[ano]:>4}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Crawler do portal wp.ufpel.edu.br/computacao (sem o PPGC)")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="JSON de saída")
    ap.add_argument("--cache-dir", default=None,
                    help="reaproveita respostas da API já baixadas")
    ap.add_argument("--delay", type=float, default=0.3,
                    help="pausa entre requisições (s)")
    ap.add_argument("--relatorio-particao", action="store_true",
                    help="imprime a divisão portal × PPGC e sai")
    args = ap.parse_args()

    cliente = WPClient(delay=args.delay,
                       cache_dir=Path(args.cache_dir) if args.cache_dir else None)
    ds = montar(cliente, relatorio_particao=args.relatorio_particao)

    if args.relatorio_particao:
        return                       # relatório é conferência, não coleta

    destino = Path(args.output)
    ds.salvar(destino)
    log.info("Gravado %s (%.1f MB, %d requisições)", destino,
             destino.stat().st_size / 1e6, cliente.n_requisicoes)
    for tabela, n in ds.contagens().items():
        if n:
            log.info("  %-24s %6d", tabela, n)


if __name__ == "__main__":
    main()
