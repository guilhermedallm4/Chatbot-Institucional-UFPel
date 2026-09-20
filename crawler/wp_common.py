"""
Utilitários compartilhados dos crawlers do WordPress da Computação-UFPel
=============================================================================
Base comum de `crawl_portal_computacao.py` (portal) e `crawl_ppgc.py` (PPGC).

Por que REST API e não scraping de HTML
---------------------------------------
`https://wp.ufpel.edu.br/computacao/` é WordPress com a REST API aberta:

    /wp-json/wp/v2/posts       746 posts   (2011 → hoje)
    /wp-json/wp/v2/pages       191 páginas
    /wp-json/wp/v2/categories    8 categorias
    /wp-json/wp/v2/tags         16 tags
    /wp-json/wp/v2/media       445 arquivos (349 PDFs)
    /wp-json/wp/v2/users         3 autores

Isso muda a natureza do problema. Raspar o HTML renderizado obrigaria a
adivinhar a data no texto ("Publicado em 15 de julho de 2026"), a paginar
`/page/2/`, `/page/3/`… e a perder posts que não aparecem em nenhuma listagem.
A API entrega `date`, `date_gmt`, `modified`, `author`, `categories`, `tags` e
`link` como CAMPOS — que é exatamente o que o enunciado pede ("capture sempre
a data dos posts"). São ~25 requisições para o site inteiro, contra ~1.000 de
um crawl de HTML.

O WAF na frente do site
-----------------------
O domínio está atrás de um SafeLine WAF que devolve **403 com uma página de
bloqueio HTML** para requisições sem cara de navegador (curl padrão, urllib,
qualquer coisa sem `Accept-Language`). O 403 NÃO é rate limit e não adianta
esperar: é o conjunto de cabeçalhos. `HEADERS` abaixo é o mínimo que passa —
mexer nele quebra o crawl inteiro de uma vez, com um 403 idêntico em todas as
requisições. `WPClient.get` detecta essa página e levanta erro explícito em
vez de tentar parsear o HTML do bloqueio como JSON.

O que este módulo entrega
-------------------------
  * `WPClient`      — sessão HTTP + paginação da REST API + cache em disco
  * `html_para_texto` — HTML do WordPress → texto limpo, legível e vetorizável
  * `dividir_secoes`  — quebra a página nos títulos, para 1 embedding por seção
  * `extrair_links` / `classificar_link` — links e anexos (PDF/DOC) de cada
    conteúdo, que é como os editais chegam
  * helpers de data, ano/semestre e slug
"""

from __future__ import annotations

import hashlib
import html as html_mod
import json
import logging
import re
import time
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

log = logging.getLogger("wp_common")

# ─────────────────────────────────────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────────────────────────────────────

SITE_URL = "https://wp.ufpel.edu.br/computacao"
API_URL = f"{SITE_URL}/wp-json/wp/v2"

#: Cabeçalhos que passam pelo SafeLine. Ver docstring do módulo antes de mudar.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,*/*;q=0.8"),
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}

MAX_RETRY = 4
RETRY_BASE = 2.0
DEFAULT_DELAY = 0.3
PER_PAGE = 100


class WAFBloqueado(RuntimeError):
    """O WAF devolveu a página de bloqueio em vez do recurso pedido."""


class WPClient:
    """
    Cliente da REST API do WordPress, com retry, cache opcional e paginação.

    O cache em disco (`--cache-dir`) guarda a resposta JSON de cada requisição
    pela URL completa. Serve para iterar no parser sem rebaixar o site a cada
    execução — o crawl inteiro sai do cache em ~2 s.
    """

    def __init__(self, api_url: str = API_URL, *, delay: float = DEFAULT_DELAY,
                 cache_dir: Optional[Path] = None, timeout: int = 60) -> None:
        self.api_url = api_url.rstrip("/")
        self.delay = delay
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.n_requisicoes = 0

    # ── infra ────────────────────────────────────────────────────────────────

    def _cache_path(self, url: str, params: dict) -> Optional[Path]:
        if not self.cache_dir:
            return None
        chave = url + "?" + json.dumps(params, sort_keys=True)
        return self.cache_dir / (hashlib.sha1(chave.encode()).hexdigest() + ".json")

    def get_url(self, url: str, params: Optional[dict] = None) -> tuple[Any, dict]:
        """
        GET em uma URL absoluta. Devolve `(json, headers)`.

        Os headers importam: `X-WP-TotalPages` é o que encerra a paginação.
        """
        params = params or {}
        cache = self._cache_path(url, params)
        if cache and cache.exists():
            guardado = json.loads(cache.read_text(encoding="utf-8"))
            return guardado["body"], guardado["headers"]

        ultimo_erro: Optional[Exception] = None
        for tentativa in range(MAX_RETRY):
            try:
                self.n_requisicoes += 1
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 403 and "slg-box" in r.text:
                    raise WAFBloqueado(
                        f"WAF bloqueou {url} — verifique HEADERS em wp_common.py")
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                corpo = r.json()
                headers = {k: v for k, v in r.headers.items()
                           if k.lower().startswith("x-wp-")}
                if cache:
                    cache.write_text(
                        json.dumps({"body": corpo, "headers": headers},
                                   ensure_ascii=False),
                        encoding="utf-8")
                if self.delay:
                    time.sleep(self.delay)
                return corpo, headers
            except WAFBloqueado:
                raise
            except Exception as exc:                      # noqa: BLE001
                ultimo_erro = exc
                espera = RETRY_BASE ** tentativa
                log.warning("  retry %d/%d em %s (%s) — aguardando %.1fs",
                            tentativa + 1, MAX_RETRY, url, str(exc)[:80], espera)
                time.sleep(espera)
        raise RuntimeError(f"falha ao buscar {url}: {ultimo_erro}")

    # ── REST API ─────────────────────────────────────────────────────────────

    def get(self, endpoint: str, **params) -> Any:
        corpo, _ = self.get_url(f"{self.api_url}/{endpoint.lstrip('/')}", params)
        return corpo

    def fetch_all(self, endpoint: str, **params) -> list[dict]:
        """
        Percorre todas as páginas de uma coleção da REST API.

        A API limita `per_page` a 100 e informa o total de páginas em
        `X-WP-TotalPages`. Paginar por "resposta vazia" falharia em silêncio se
        uma página no meio voltasse vazia por erro transitório.
        """
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        itens: list[dict] = []
        pagina = 1
        while True:
            corpo, headers = self.get_url(
                url, {"per_page": PER_PAGE, "page": pagina, **params})
            if not isinstance(corpo, list) or not corpo:
                break
            itens.extend(corpo)
            total_paginas = int(headers.get("X-WP-TotalPages") or 1)
            if pagina >= total_paginas:
                break
            pagina += 1
        log.info("  %-14s %5d itens", endpoint, len(itens))
        return itens

    def get_texto(self, url: str) -> str:
        """GET cru (não-JSON) — usado para o .ics do Google Calendar."""
        for tentativa in range(MAX_RETRY):
            try:
                self.n_requisicoes += 1
                r = self.session.get(url, timeout=self.timeout)
                r.raise_for_status()
                r.encoding = r.encoding or "utf-8"
                return r.text
            except Exception as exc:                      # noqa: BLE001
                if tentativa == MAX_RETRY - 1:
                    raise
                log.warning("  retry %d em %s (%s)", tentativa + 1, url, str(exc)[:80])
                time.sleep(RETRY_BASE ** tentativa)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# HTML → texto
# ─────────────────────────────────────────────────────────────────────────────

#: Blocos que viram parágrafo próprio no texto extraído.
_BLOCOS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote",
           "figcaption", "dd", "dt", "pre", "div"}

#: Tags cujo conteúdo nunca vai para o texto.
_IGNORAR = {"script", "style", "noscript", "iframe", "svg", "form", "button"}


#: Marcador temporário para a quebra de linha vinda de `<br>`.
#:
#: Precisa ser um caractere NÃO-branco. `get_text(" ", strip=True)` do
#: BeautifulSoup aplica `.strip()` a cada nó de texto, e um "\n" puro
#: seria descartado exatamente onde queremos preservá-lo. Um caractere de
#: uso privado (U+E000) atravessa o strip intacto e vira "\n" aqui.
_MARCA_QUEBRA = "\ue000"


def limpar_texto(texto: Optional[str]) -> str:
    """Desescapa entidades, normaliza NBSP/quebras e colapsa espaços."""
    if not texto:
        return ""
    t = html_mod.unescape(texto)
    t = t.replace(" ", " ").replace("​", "")
    t = t.replace(_MARCA_QUEBRA, "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"[ \t]*\n[ \t]*", "\n", t)
    # "…um <a>link</a>." vira "…um link ." ao juntar os nós com espaço.
    # Cola a pontuação de volta: o texto vai para o embedding e para a
    # resposta ao usuário, e espaço antes de vírgula/ponto aparece nas duas.
    t = re.sub(r" +([,.;:!?%)\]])", r"\1", t)
    t = re.sub(r"([(\[]) +", r"\1", t)
    return t.strip()


def titulo_wp(rendered: Optional[str]) -> str:
    """Título do WordPress (`title.rendered`) → texto puro."""
    if not rendered:
        return ""
    return limpar_texto(BeautifulSoup(rendered, "lxml").get_text(" ", strip=True))


def _texto_de_tabela(tabela: Tag) -> str:
    """
    Tabela → linhas "célula | célula".

    O formato importa para o embedding: a tabela de requisitos do PPGC
    ("mínimo de 20 créditos | até antes de marcar a defesa") só responde
    "qual o prazo para X" se requisito e prazo ficarem na MESMA linha. Achatar
    célula a célula em parágrafos separados destrói esse pareamento.
    """
    linhas: list[str] = []
    for tr in tabela.find_all("tr"):
        celulas = [limpar_texto(td.get_text(" ", strip=True))
                   for td in tr.find_all(["th", "td"])]
        celulas = [c for c in celulas if c]
        if celulas:
            linhas.append(" | ".join(celulas))
    return "\n".join(linhas)


def html_para_texto(html: Optional[str]) -> str:
    """
    HTML do editor de blocos do WordPress → texto limpo para leitura e embedding.

    Preserva a estrutura que carrega significado e descarta a que não carrega:
      * `<li>` vira "- item"  (listas de editais, requisitos, documentos)
      * `<table>` vira linhas "a | b"  (ver `_texto_de_tabela`)
      * títulos viram linha própria seguida de linha em branco
      * `<script>`, `<style>`, `<iframe>` somem

    Não usamos `soup.get_text("\\n")` direto porque ele quebra em TODO nó de
    texto: "Resposta<em>:</em> O aluno deve…" viraria três linhas, e o chunker
    passaria a cortar no meio de frases.
    """
    if not html or not html.strip():
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(_IGNORAR):
        tag.decompose()
    # <br> é separador de CAMPO em várias páginas — a lista de disciplinas do
    # PPGC põe nome, responsável, ementa e horário num <li> só, separados por
    # <br>. `get_text` ignora <br>, o que colaria tudo numa linha e tornaria
    # impossível recuperar cada campo depois.
    for br in soup.find_all("br"):
        br.replace_with(_MARCA_QUEBRA)

    partes: list[str] = []

    def emitir(txt: str) -> None:
        txt = limpar_texto(txt)
        if txt:
            partes.append(txt)

    def caminhar(no: Tag) -> None:
        for filho in no.children:
            if isinstance(filho, NavigableString):
                txt = limpar_texto(str(filho))
                if txt:
                    partes.append(txt)
                continue
            if not isinstance(filho, Tag):
                continue
            nome = filho.name
            if nome in _IGNORAR:
                continue
            if nome == "table":
                emitir(_texto_de_tabela(filho))
            elif nome == "li":
                emitir("- " + filho.get_text(" ", strip=True))
            elif nome in _BLOCOS:
                # div/figure podem conter outros blocos: só achata quando é folha
                tem_bloco_dentro = filho.find(
                    ["p", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol"])
                if tem_bloco_dentro:
                    caminhar(filho)
                else:
                    emitir(filho.get_text(" ", strip=True))
            elif nome in {"ul", "ol", "figure", "section", "article", "tbody",
                          "thead", "main", "header", "footer", "span", "a",
                          "strong", "em", "b", "i", "mark", "s", "u", "small",
                          "sup", "sub", "code"}:
                if filho.find(["p", "li", "table", "div", "h1", "h2", "h3",
                               "h4", "h5", "h6"]):
                    caminhar(filho)
                else:
                    emitir(filho.get_text(" ", strip=True))
            else:
                caminhar(filho)

    caminhar(soup.body or soup)

    texto = "\n".join(partes)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    return texto.strip()


def resumo_de(texto: str, limite: int = 400) -> Optional[str]:
    """Primeiras frases do texto, até `limite` caracteres, sem cortar palavra."""
    texto = (texto or "").strip()
    if not texto:
        return None
    if len(texto) <= limite:
        return texto
    corte = texto[:limite]
    ponto = max(corte.rfind(". "), corte.rfind("\n"))
    if ponto > limite * 0.5:
        return corte[:ponto + 1].strip()
    return corte.rsplit(" ", 1)[0].strip() + "…"


def contar_palavras(texto: Optional[str]) -> int:
    return len(re.findall(r"\w+", texto or "", flags=re.UNICODE))


# ─────────────────────────────────────────────────────────────────────────────
# Seções (granularidade do embedding)
# ─────────────────────────────────────────────────────────────────────────────

#: Um parágrafo com <= 90 chars inteiramente em <strong> e sem ponto final é,
#: no editor de blocos deste site, um subtítulo — o autor não usou <h_>.
#: É assim que "Regimento Interno", "Aluno regular" e "Resoluções em vigor"
#: aparecem em /ppgc/regimento-e-resolucoes e /ppgc/editais-de-ingresso.
_MAX_PSEUDO_TITULO = 90

#: Rótulos que são negrito estrutural, não subtítulo. Nas páginas de FAQ o
#: par "Pergunta:"/"Resposta:" está inteiro em <strong>; sem esta lista, cada
#: resposta órfã abriria uma seção chamada "Resposta".
_NAO_SAO_TITULOS = {"pergunta", "resposta", "atencao", "obs", "observacao",
                    "importante", "nota", "aviso", "fonte", "link", "links"}


def e_subtitulo(tag: Tag) -> bool:
    """
    `True` se este `<p>` é, na prática, um subtítulo: texto curto, inteiramente
    em negrito e sem ponto final. Ver `_MAX_PSEUDO_TITULO`.
    """
    if tag.name != "p":
        return False
    texto = limpar_texto(tag.get_text(" ", strip=True))
    if not texto or len(texto) > _MAX_PSEUDO_TITULO or texto.endswith("."):
        return False
    if normalizar(texto).rstrip(":").strip() in _NAO_SAO_TITULOS:
        return False
    fortes = tag.find_all(["strong", "b"])
    if not fortes:
        return False
    texto_forte = limpar_texto(" ".join(f.get_text(" ", strip=True) for f in fortes))
    return texto_forte.rstrip(":") == texto.rstrip(":")


def dividir_secoes(html: Optional[str]) -> list[dict]:
    """
    Divide o conteúdo nos títulos: `[{titulo, nivel, ancora, texto, html}]`.

    O `html` de cada seção volta junto porque o TÍTULO DA SEÇÃO é o melhor
    rótulo que existe para os links que ela contém. Numa página de edital do
    PPGC, o mesmo PDF é "Edital", "Retificação" ou "Resultado" conforme o
    bloco em que está — e o texto do link raramente diz isso sozinho.

    Por que seção e não a página inteira: a granularidade do embedding É a
    qualidade da recuperação. "Quais são os requisitos de proficiência em
    inglês?" precisa recuperar o trecho da proficiência — não a página de 42 mil
    caracteres do FAQ, onde o sinal some. Uma página sem título nenhum devolve
    uma única seção com `titulo=None`, e o chunker do loader cuida do resto.

    A âncora (`<a name="prim-mat">`) vira o fragmento da URL, o que permite
    citar `…/faq-alunos/#prim-mat` na resposta em vez de mandar o usuário
    procurar na página.
    """
    if not html or not html.strip():
        return []
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(_IGNORAR):
        tag.decompose()
    raiz = soup.body or soup

    secoes: list[dict] = []
    atual = {"titulo": None, "nivel": 0, "ancora": None, "html": []}
    ancora_pendente: Optional[str] = None

    def fechar() -> None:
        html_secao = "".join(atual["html"])
        texto = html_para_texto(html_secao)
        if texto or atual["titulo"]:
            secoes.append({"titulo": atual["titulo"], "nivel": atual["nivel"],
                           "ancora": atual["ancora"], "texto": texto,
                           "html": html_secao})

    for filho in list(raiz.children):
        if isinstance(filho, NavigableString):
            if limpar_texto(str(filho)):
                atual["html"].append(str(filho))
            continue
        if not isinstance(filho, Tag):
            continue

        # <p><a name="x"></a></p> — a âncora vale para o título que vem depois
        ancoras = [a.get("name") or a.get("id") for a in filho.find_all("a")
                   if (a.get("name") or a.get("id")) and not a.get("href")]
        if ancoras and not limpar_texto(filho.get_text(" ", strip=True)):
            ancora_pendente = ancoras[0]
            continue
        if filho.get("id"):
            ancora_pendente = filho.get("id")

        titulo_novo = None
        nivel = 0
        if re.fullmatch(r"h[1-6]", filho.name or ""):
            titulo_novo = limpar_texto(filho.get_text(" ", strip=True))
            nivel = int(filho.name[1])
        elif e_subtitulo(filho):
            titulo_novo = limpar_texto(filho.get_text(" ", strip=True)).rstrip(":")
            nivel = 4

        if titulo_novo:
            fechar()
            atual = {"titulo": titulo_novo, "nivel": nivel,
                     "ancora": ancoras[0] if ancoras else ancora_pendente,
                     "html": []}
            ancora_pendente = None
        else:
            atual["html"].append(str(filho))

    fechar()
    return [s for s in secoes if s["texto"] or s["titulo"]]


# ─────────────────────────────────────────────────────────────────────────────
# Links e anexos
# ─────────────────────────────────────────────────────────────────────────────

_EXT_DOCUMENTO = {"pdf", "doc", "docx", "odt", "rtf"}
_EXT_PLANILHA = {"xls", "xlsx", "ods", "csv"}
_EXT_IMAGEM = {"jpg", "jpeg", "png", "gif", "webp", "svg"}


def extensao_de(url: str) -> Optional[str]:
    caminho = urlparse(unquote(url)).path
    m = re.search(r"\.([A-Za-z0-9]{1,5})$", caminho)
    return m.group(1).lower() if m else None


def classificar_link(url: str) -> str:
    """documento | planilha | imagem | email | interno | externo"""
    if url.lower().startswith("mailto:"):
        return "email"
    ext = extensao_de(url)
    if ext in _EXT_DOCUMENTO:
        return "documento"
    if ext in _EXT_PLANILHA:
        return "planilha"
    if ext in _EXT_IMAGEM:
        return "imagem"
    host = (urlparse(url).netloc or "").lower()
    if not host or host.endswith("wp.ufpel.edu.br"):
        return "interno"
    return "externo"


def extrair_links(html: Optional[str], base: str = SITE_URL) -> list[dict]:
    """
    Todos os `<a href>` do conteúdo, na ordem, com texto e classificação.

    É por aqui que os editais entram no dataset: no WordPress da Computação o
    PDF do edital não é um campo — é um link dentro do corpo da página, com o
    rótulo carregando o tipo do documento ("Edital", "Retificação 1",
    "Resultado final"). Guardar o texto do link é o que permite tipar o
    documento depois, sem abrir o PDF.
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    links: list[dict] = []
    vistos: set[str] = set()
    for ordem, a in enumerate(soup.find_all("a", href=True)):
        href = (a["href"] or "").strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        url = urljoin(base + "/", href)
        texto = limpar_texto(a.get_text(" ", strip=True))
        chave = (url, texto)
        if chave in vistos:
            continue
        vistos.add(chave)
        links.append({
            "ordem": len(links),
            "url": url,
            "texto": texto or None,
            "host": (urlparse(url).netloc or None),
            "tipo": classificar_link(url),
            "extensao": extensao_de(url),
        })
    return links


def nome_arquivo_de(url: str) -> Optional[str]:
    caminho = urlparse(unquote(url)).path
    nome = caminho.rsplit("/", 1)[-1]
    return nome or None


# ─────────────────────────────────────────────────────────────────────────────
# Datas, ano/semestre, slugs
# ─────────────────────────────────────────────────────────────────────────────

MESES_PT = {1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril", 5: "maio",
            6: "junho", 7: "julho", 8: "agosto", 9: "setembro", 10: "outubro",
            11: "novembro", 12: "dezembro"}

MES_POR_NOME = {v: k for k, v in MESES_PT.items()}

#: Mesma coisa sem acento e em minúsculas — o cabeçalho do widget de
#: calendário vem "Março / 2022", "Fevereiro / 2022", e casar por
#: `normalizar()` evita depender da acentuação exata do HTML.
MES_POR_NOME_NORM: dict[str, int] = {}          # preenchido abaixo de normalizar()


def parse_data_wp(valor: Optional[str]) -> Optional[datetime]:
    """`2026-07-15T14:33:46` (formato da REST API) → datetime."""
    if not valor:
        return None
    try:
        return datetime.fromisoformat(valor.replace("Z", "+00:00"))
    except ValueError:
        return None


def data_por_extenso(d: Optional[date]) -> Optional[str]:
    """
    `date(2026, 7, 15)` → "15 de julho de 2026".

    Vai no CABEÇALHO do texto vetorizado, não só na coluna. Uma pergunta como
    "o que saiu sobre matrícula em julho de 2026?" só casa semanticamente se o
    mês estiver escrito no chunk; "2026-07-15" não ativa nada no embedding.
    """
    if not d:
        return None
    return f"{d.day} de {MESES_PT[d.month]} de {d.year}"


def semestre_de(d: Optional[date]) -> Optional[int]:
    if not d:
        return None
    return 1 if d.month <= 6 else 2


_RE_ANO_SEM = re.compile(r"\b(20\d{2})\s*[/\-.]\s*([12])\b")
_RE_ANO_SEM_SLUG = re.compile(r"\b(20\d{2})-([12])\b")
_RE_ANO = re.compile(r"\b(20\d{2})\b")


def ano_semestre_de(*textos: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    """
    Extrai (ano, semestre) de títulos/slugs como "…2025/1", "…-2024-2".

    Percorre os textos na ordem dada e devolve o PRIMEIRO casamento — por isso
    quem chama passa o slug antes do corpo: "selecao-aluno-regular-2026-1" é
    uma afirmação sobre o edital, enquanto um "2019" no meio do texto é só uma
    citação. Ano sem semestre é aceito (`(2026, None)`); nada encontrado
    devolve `(None, None)`.
    """
    for texto in textos:
        if not texto:
            continue
        m = _RE_ANO_SEM.search(texto) or _RE_ANO_SEM_SLUG.search(texto)
        if m:
            return int(m.group(1)), int(m.group(2))
    for texto in textos:
        if not texto:
            continue
        m = _RE_ANO.search(texto)
        if m:
            return int(m.group(1)), None
    return None, None


def anos_citados(*textos: Optional[str]) -> list[int]:
    """Todos os anos 20xx citados, sem repetição e em ordem."""
    achados: list[int] = []
    for texto in textos:
        for m in _RE_ANO.finditer(texto or ""):
            ano = int(m.group(1))
            if ano not in achados:
                achados.append(ano)
    return sorted(achados)


def normalizar(texto: Optional[str]) -> str:
    """Minúsculas sem acento — para casar slug/título com heurística."""
    if not texto:
        return ""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


MES_POR_NOME_NORM.update({normalizar(nome): num for nome, num in MES_POR_NOME.items()})


def caminho_da_pagina(pagina: dict, por_id: dict[int, dict]) -> str:
    """
    Caminho hierárquico da página ("ppgc/editais-de-ingresso/selecao-2025").

    O `link` da REST API não serve para isso: o WordPress reescreve permalinks
    quando a página muda de pai, e várias páginas de edital do PPGC ficaram
    com `link` na raiz (`/computacao/selecao-de-aluno-regular-2026-1/`) mesmo
    tendo pai. O caminho reconstruído por `parent` é a hierarquia real.
    """
    partes: list[str] = []
    atual: Optional[dict] = pagina
    visitados: set[int] = set()
    while atual is not None and atual.get("id") not in visitados:
        visitados.add(atual.get("id"))
        partes.append(atual.get("slug") or str(atual.get("id")))
        pai = atual.get("parent") or 0
        atual = por_id.get(pai)
    return "/".join(reversed(partes))


# ─────────────────────────────────────────────────────────────────────────────
# Partição portal × PPGC
# ─────────────────────────────────────────────────────────────────────────────
# Os dois crawlers chamam ESTAS funções. Se a regra morasse em cada um, a
# menor divergência criaria conteúdo duplicado (nos dois datasets) ou órfão
# (em nenhum) — e nada no processo acusaria o erro.

#: Slug da categoria do WordPress que marca conteúdo de pós-graduação.
CATEGORIA_PPGC = "ppgc"

#: Páginas do PPGC que o WordPress deixou na raiz do site (sem `parent`).
#: A hierarquia do menu diz que são do Programa, mas a árvore de páginas não —
#: vários editais foram publicados soltos. Sem esta regra eles cairiam no
#: portal e o dataset do PPGC perderia justamente os editais mais recentes
#: (seleção de aluno regular 2026/1 e 2026/2, por exemplo).
_SLUG_PPGC = re.compile(
    r"^(selecao-|selecao-de-|edital-|editais-|resultado-homologacao"
    r"|lista-de-disciplinas|primeira-matricula-no-dinter"
    r"|programacao-de-ofertas-dinter|dinter-|ppgc-disciplines"
    r"|graduate-program-in-computing|postgraduate-course)")

#: Caminhos (hierarquia real de páginas) cuja subárvore INTEIRA é do PPGC.
#: As duas últimas são a tradução do site em inglês (Polylang). Elas precisam
#: estar aqui, e não só em `_SLUG_PPGC`: a regra de slug vê o último segmento,
#: então `graduate-program-in-computing/research-lines` tem slug
#: "research-lines" e escaparia para o portal, levando junto as linhas de
#: pesquisa, o corpo docente e o mestrado/doutorado em inglês.
_CAMINHO_PPGC = ("ppgc", "pos-graduacao",
                 "graduate-program-in-computing", "postgraduate-course")


def pagina_e_ppgc(caminho: str, slug: str, titulo: str) -> bool:
    """
    Decide se uma PÁGINA pertence ao dataset do PPGC.

    Ordem das regras: subárvore → slug → título. A checagem por título é o
    último recurso e exige a sigla ou o nome do Programa; "mestrado" sozinho
    não basta, porque páginas de graduação citam mestrado o tempo todo.
    """
    caminho_norm = normalizar(caminho)
    if any(caminho_norm == raiz or caminho_norm.startswith(raiz + "/")
           for raiz in _CAMINHO_PPGC):
        return True
    if _SLUG_PPGC.match(normalizar(slug)):
        return True
    titulo_norm = normalizar(titulo)
    return ("ppgc" in titulo_norm
            or "pos-graduacao em computacao" in titulo_norm
            or "postgraduate program" in titulo_norm)


#: Categorias que NÃO dizem nada sobre o assunto do post: a categoria legada
#: (`noticias`, rotulada "Legado" no painel), o balde `todos`, `informe` e a
#: categoria vazia da tradução em inglês.
_CATEGORIAS_INDECISAS = {"noticias", "todos", "informe", "sem-categoria-en"}

#: Marcadores de pós-graduação no título/resumo. Só entram em jogo quando o
#: post não tem NENHUMA categoria informativa (ver `post_e_ppgc`).
_RE_MARCADOR_PPGC = re.compile(
    r"\bppgc\b|\bdinter\b|pos-?gradua|\bmestrado\b|\bdoutorado\b"
    r"|aluno[/\s-]*especial|aluno[/\s-]*regular|estudante[/\s-]*especial",
    re.I)


def post_e_ppgc(post: dict, slug_por_categoria: dict[int, str]) -> bool:
    """
    Decide se um POST pertence ao dataset do PPGC.

    A categoria `ppgc` do WordPress é a fonte primária e resolve 287 dos 746
    posts. Ela sozinha não basta, e o buraco é grande: **os 8 posts de 2026
    inteiros** — nota 6 da CAPES, editais de seleção 2026/1 e 2026/2, oferta
    de disciplinas — foram publicados na categoria legada `noticias`, sem
    `ppgc`. Confiar só na categoria entregaria ao dataset do PPGC um acervo
    que para em 2025, o que é o pior resultado possível: quem pergunta sobre
    edital quer o edital vigente.

    Por isso o desempate em três passos:

      1. tem a categoria `ppgc`                       → PPGC
      2. tem qualquer categoria informativa
         (`ccomp`, `ecomp`, `noticia`)                → portal, sem discussão
      3. só tem categoria genérica/legada             → decide pelo título

    O passo 2 é o que segura a heurística: um post marcado como Ciência da
    Computação continua no portal mesmo citando "mestrado" de passagem. O
    passo 3 só age onde a fonte não disse nada.
    """
    slugs = {slug_por_categoria.get(cid)
             for cid in (post.get("categories") or [])}
    slugs.discard(None)
    if CATEGORIA_PPGC in slugs:
        return True
    if slugs - _CATEGORIAS_INDECISAS:
        return False
    alvo = " ".join([
        titulo_wp((post.get("title") or {}).get("rendered")),
        titulo_wp((post.get("excerpt") or {}).get("rendered")),
        post.get("slug") or "",
    ])
    return bool(_RE_MARCADOR_PPGC.search(normalizar(alvo)))


def idioma_de(link: Optional[str], slug: Optional[str] = None) -> str:
    """
    'pt' | 'en' — o site usa Polylang com as páginas em inglês em /en/.

    Nem toda página em inglês está sob /en/ (algumas ficaram na raiz), então o
    fallback olha o slug: 'computer-science', 'research-lines'…
    """
    if link and "/computacao/en/" in link:
        return "en"
    slug_norm = normalizar(slug)
    marcadores_en = ("computer-", "graduate-program", "research", "graduation",
                     "education", "extension", "postgraduate", "faculty-and",
                     "visual-identity", "location-and", "laboratory-schedule",
                     "computing-course", "ppgc-disciplines", "admission")
    if slug_norm and slug_norm.startswith(marcadores_en):
        return "en"
    return "pt"


# ─────────────────────────────────────────────────────────────────────────────
# Coletor de linhas
# ─────────────────────────────────────────────────────────────────────────────

class Dataset:
    """
    Acumula as linhas já no formato das tabelas do schema.

    Mesmo contrato do `Dataset` de `crawl_computacao.py`: `KEYS` declara a
    chave lógica de cada tabela e `add` deduplica por ela, mantendo a linha
    mais preenchida. `ORDER` é a ordem de carga (respeita as FKs) e também a
    ordem em que o JSON é serializado.
    """

    KEYS: dict[str, tuple[str, ...]] = {}
    ORDER: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.tables: dict[str, list[dict]] = {t: [] for t in self.ORDER}
        self._index: dict[str, dict[tuple, int]] = {t: {} for t in self.ORDER}

    @staticmethod
    def _riqueza(row: dict) -> int:
        return sum(1 for v in row.values() if v not in (None, "", [], {}))

    def add(self, tabela: str, row: dict, *, merge: bool = False) -> None:
        """
        Insere a linha, deduplicando por `KEYS[tabela]`.

        `merge=False` (padrão): em colisão vence a linha com mais campos
        preenchidos, inteira. Serve para quando duas leituras da mesma fonte
        divergem em qualidade.

        `merge=True`: as duas linhas se COMPLEMENTAM campo a campo — cada
        valor nulo é preenchido pela outra. É o que os normativos do PPGC
        exigem: a página da resolução tem a íntegra do texto mas não sabe se
        ela ainda vale, e o índice "Regimento e Resoluções" sabe a vigência
        (está sob "Resoluções em vigor" ou "revogadas") mas não tem o texto.
        Com a regra de riqueza, a página venceria por ter mais campos e
        `vigente` voltaria a NULL — apagando justamente o dado que impede o
        RAG de responder com uma resolução revogada.
        """
        if tabela not in self.tables:
            raise KeyError(f"tabela desconhecida: {tabela}")
        chave = tuple(row.get(c) for c in self.KEYS[tabela])
        pos = self._index[tabela].get(chave)
        if pos is None:
            self._index[tabela][chave] = len(self.tables[tabela])
            self.tables[tabela].append(row)
            return
        atual = self.tables[tabela][pos]
        if merge:
            combinado = dict(atual)
            for campo, valor in row.items():
                if combinado.get(campo) in (None, "", [], {}):
                    combinado[campo] = valor
            self.tables[tabela][pos] = combinado
        elif self._riqueza(row) > self._riqueza(atual):
            self.tables[tabela][pos] = row

    def contagens(self) -> dict[str, int]:
        return {t: len(self.tables[t]) for t in self.ORDER}

    def to_json(self) -> dict[str, list[dict]]:
        return {t: self.tables[t] for t in self.ORDER}

    def salvar(self, destino: Path) -> None:
        destino = Path(destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")


def iso(valor: Any) -> Optional[str]:
    """datetime/date → string ISO; qualquer outra coisa passa direto."""
    if isinstance(valor, (datetime, date)):
        return valor.isoformat()
    return valor
