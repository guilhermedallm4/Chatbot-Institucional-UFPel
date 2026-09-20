"""
Crawler dos Cursos de COMPUTAÇÃO da UFPel — saída relacional + vetorial
=============================================================================
Coleta restrita aos 5 cursos de Computação e a tudo que eles referenciam.

  ┌────────┬─────────────────────────────────────────────────┬───────────────────┐
  │ Código │ Curso                                           │ Nível / Grau      │
  ├────────┼─────────────────────────────────────────────────┼───────────────────┤
  │ 3900   │ Ciência da Computação                           │ Grad./Bacharelado │
  │ 3910   │ Engenharia de Computação                        │ Grad./Bacharelado │
  │ 7057   │ Computação                                      │ Pós/Mestrado Acad.│
  │ 8102   │ Computação                                      │ Pós/Doutorado     │
  │ 9130   │ Especialização em Computação na Educação Básica │ Pós/Especialização│
  └────────┴─────────────────────────────────────────────────┴───────────────────┘

Diferença em relação a crawl_ufpel.py
-------------------------------------
crawl_ufpel.py produz DOCUMENTOS (um JSON por página, com dados_completos
JSONB) — ótimo para busca vetorial, ruim para SQL: qualquer agregação exige
navegar JSONB e a mesma turma aparece duplicada por versão de currículo.

Este módulo produz LINHAS DE TABELA já normalizadas, no formato exato do
schema_computacao.sql. A saída é um dict {nome_da_tabela: [linhas]}, o que
torna a carga (load_computacao.py) genérica e idempotente.

Fases
-----
  1. CURSOS       /cursos/cod/<cod>
                  ficha-dados (nível, códigos, coordenador, vagas por cota),
                  conceitos (Enade/CPC), aba Informações (uma linha por seção),
                  Matriz Curricular (+ pré-requisitos), Professores,
                  Turmas Ofertadas (por VERSÃO de currículo), rodapé (legenda
                  das siglas de cota)
  2. DISCIPLINAS  /disciplinas/cod/<cod>   (todas as citadas pelos 5 cursos)
                  ficha, Ementa/Objetivos/Conteúdo/Bibliografia, equivalentes.
                  A aba "Turmas Ofertadas" da disciplina é IGNORADA de
                  propósito: a oferta é capturada na fase 1, já com curso,
                  versão de currículo e semestre — informação que a página da
                  disciplina não tem.
  3. PROFESSORES  /servidores/id/<id>      (apenas vínculo ativo)
                  ficha do vínculo ativo, função/gratificação, contatos,
                  currículo Lattes (resumo, formação, áreas de atuação),
                  projetos vigentes e disciplinas ministradas
  4. PROJETOS     /projetos/id/<id>        (apenas vigentes)
                  ficha, aba Informações, equipe

Ao final, um passo de reconciliação liga turma → servidor_id cruzando
(disciplina, ano, semestre, código da turma) entre a aba de turmas do curso
(que publica só o nome do professor) e a aba "Disciplinas ministradas" da
página do professor (que tem o id).

Armadilhas do portal já tratadas aqui
-------------------------------------
  * Vínculos encerrados do servidor ficam no MESMO div.ficha-dados, dentro de
    div.vinculo.oculta-exibe-conteudo. Ler a ficha achatada faz o vínculo
    encerrado sobrescrever o ativo (cargo/titulação errados).
  * A aba de projetos do servidor usa UMA tabela com linhas-separadoras
    (th.tabela-quebra) para Ensino/Extensão/Pesquisa. Pegar o primeiro <th>
    da tabela atribui a mesma ênfase a todos os projetos.
  * A mesma turma aparece sob várias versões de currículo → turma e
    turma_curriculo são tabelas separadas.
  * O link do Lattes está em ul.contatos, fora da div#lattes.
  * Pré-requisitos vêm em span.tabela-detalhe-info dentro da célula da
    disciplina, na matriz — e não na página da disciplina.

Uso
---
    python crawl_computacao.py                            # todos os cursos
    python crawl_computacao.py --cursos 3900 3910         # subconjunto
    python crawl_computacao.py --output computacao.json
    python crawl_computacao.py --concurrency 5 --delay 1.0
    python crawl_computacao.py --cache-dir .cache_html    # reaproveita HTML baixado

Requisitos: pip install -r requirements_crawler.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL = "https://institucional.ufpel.edu.br"

TARGET_CURSOS: dict[str, dict[str, str]] = {
    "3900": {"nome": "Ciência da Computação",                           "grau": "Bacharelado"},
    "3910": {"nome": "Engenharia de Computação",                        "grau": "Bacharelado"},
    "7057": {"nome": "Computação",                                      "grau": "Mestrado Acadêmico"},
    "8102": {"nome": "Computação",                                      "grau": "Doutorado"},
    "9130": {"nome": "Especialização em Computação na Educação Básica", "grau": "Especialização"},
}

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept-Language": "pt-BR,pt;q=0.9",
    "Accept": "text/html,application/xhtml+xml",
}

SKIP_HTTP_ERRORS = frozenset({400, 401, 403, 404, 410, 500, 502, 503})
MAX_RETRY = 4
RETRY_BASE = 2.0
DEFAULT_CONCURRENCY = 5
DEFAULT_DELAY = 1.0
DEFAULT_OUTPUT = "computacao.json"

NOISE_TAGS = ["script", "style", "noscript", "iframe", "nav", "header", "footer", "form", "button"]

# Situações funcionais que indicam vínculo encerrado
INACTIVE_SITUACOES = frozenset({
    "aposentad", "falecid", "exonerad", "demitid", "excluíd", "excluid", "exclusão",
    "exclusao", "redistribuíd", "redistribuid", "cedid", "contrato encerrado",
    "rescindid", "desligad", "vacância", "vacancia",
})

# Versão sintética usada quando o curso não publica versões de currículo
# (acontece em cursos novos, sem turmas ofertadas ainda — ex.: 9130).
VERSAO_UNICA = "unica"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("crawl_computacao")


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários de texto
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str | None) -> Optional[str]:
    """
    Normaliza espaços/quebras. Retorna None (não sentinela) se vazio.

    Runs de quebra de linha colapsam para UMA. O HTML do portal já traz
    newlines nos nós de texto e usa <br> no mesmo ponto, então sem esse
    colapso toda ementa sairia com linhas em branco alternadas.
    """
    if text is None:
        return None
    t = unicodedata.normalize("NFC", str(text))
    t = t.replace("\xa0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"[ \t]*\n[ \t\n]*", "\n", t).strip()
    return t or None


def _unaccent(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def _slug(text: str) -> str:
    """'Perfil do Egresso' → 'perfil_egresso' (chave estável para escopo/seção)."""
    s = _unaccent(text or "").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "sem_titulo"


def _norm_nome(nome: str | None) -> str:
    """Chave de comparação de nomes de pessoa (sem acento, sem pontuação)."""
    if not nome:
        return ""
    return re.sub(r"[^a-z0-9 ]", "", _unaccent(nome).lower()).strip()


def _block_text(el: Any) -> Optional[str]:
    """
    Texto de um bloco preservando as quebras estruturais.

    O portal usa <br> como separador de item em ementa, conteúdo programático
    e formação acadêmica. get_text(" ") colaria tudo numa linha e destruiria
    a estrutura que faz o texto ser legível (e recuperável) depois.
    """
    if el is None:
        return None
    for br in el.find_all("br"):
        br.replace_with("\n")
    for block in el.find_all(["p", "li", "div", "tr"]):
        block.append("\n")
    return _clean(el.get_text(" ", strip=False))


def _txt(el: Any) -> Optional[str]:
    return _clean(el.get_text(" ", strip=True)) if el is not None else None


def _to_int(text: str | None) -> Optional[int]:
    """'60 horas' → 60; '4' → 4; '' → None."""
    if not text:
        return None
    m = re.search(r"-?\d+", str(text).replace(".", ""))
    return int(m.group()) if m else None


def _to_date_iso(text: str | None) -> Optional[str]:
    """'21/07/2025' → '2025-07-21' (ISO, pronto p/ cast em DATE)."""
    m = re.search(r"(\d{2})/(\d{2})/(\d{4})", text or "")
    if not m:
        return None
    try:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1))).isoformat()
    except ValueError:
        return None


def _to_time(text: str | None) -> Optional[str]:
    m = re.match(r"\s*(\d{1,2}):(\d{2})", text or "")
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    return f"{h:02d}:{mi:02d}" if 0 <= h <= 23 and 0 <= mi <= 59 else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_url(url: str) -> str:
    return urlparse(url)._replace(fragment="").geturl()


def _id_from_href(href: str | None, kind: str) -> Optional[str]:
    """
    Extrai o identificador do portal de um href.

      /servidores/id/132456      → '132456'
      /projetos/id/u9876         → 'u9876'
      /cursos/cod/3900           → '3900'
      /disciplinas/cod/22000294  → '22000294'
    """
    if not href:
        return None
    m = re.search(rf"/{kind}/(?:id|cod)/([A-Za-z0-9_-]+)", href)
    return m.group(1) if m else None


def _dia_norm(dia: str | None) -> Optional[str]:
    if not dia:
        return None
    d = _unaccent(dia).upper().strip()[:3]
    return d if d in ("SEG", "TER", "QUA", "QUI", "SEX", "SAB", "DOM") else None


def _split_ano_semestre(text: str | None) -> tuple[Optional[int], Optional[int]]:
    """'2026 / 2' ou '2026/2' → (2026, 2)."""
    m = re.search(r"(\d{4})\s*/\s*(\d)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _semestre_num(rotulo: str | None) -> Optional[int]:
    """'3º Semestre' → 3; 'Optativas' → None."""
    m = re.match(r"\s*(\d{1,2})\s*[°ºo]?\s*sem", _unaccent(rotulo or ""), re.IGNORECASE)
    return int(m.group(1)) if m else None


def _bloco_from_rotulo(rotulo: str | None) -> str:
    """Classifica o bloco da matriz a partir do rótulo do accordion."""
    r = _unaccent(rotulo or "").lower()
    if "optativ" in r:
        return "optativas"
    if "complementar" in r:
        return "complementares"
    return "semestre"


def _turma_key(disciplina: str, ano: int | None, semestre: int | None, codigo: str) -> str:
    """Chave determinística da turma: 22000294-2026-2-M11."""
    return f"{disciplina}-{ano or 0}-{semestre or 0}-{codigo}"


# ─────────────────────────────────────────────────────────────────────────────
# Extratores genéricos de HTML do portal
# ─────────────────────────────────────────────────────────────────────────────

def _strip_noise(soup: BeautifulSoup) -> None:
    for tag in soup(NOISE_TAGS):
        tag.decompose()


#: Classes que marcam conteúdo colapsado na página do servidor.
#  O vínculo ATIVO fica num div.vinculo simples; os ENCERRADOS ficam em
#  div.vinculo.oculta-exibe.oculta-exibe-conteudo. Portanto o discriminante é
#  a classe de ocultação — filtrar por "vinculo" descartaria o vínculo ativo
#  junto com os antigos (e a ficha sairia inteira nula).
_CLASSES_OCULTAS = ("oculta-exibe-conteudo", "oculta-exibe")


def _in_hidden_vinculo(el: Any, root: Any) -> bool:
    """
    True se `el` está dentro de um bloco colapsado de vínculo encerrado.

    O portal empilha os vínculos antigos do servidor no MESMO div.ficha-dados.
    Sem esse filtro, os rótulos repetidos (Cargo, Titulação, Lotação) do
    vínculo encerrado sobrescrevem os do vínculo ativo.
    """
    node = el
    while node is not None and node is not root:
        classes = node.get("class") or [] if hasattr(node, "get") else []
        if any(c in classes for c in _CLASSES_OCULTAS):
            return True
        node = node.parent
    return False


def _ficha_pairs(root: Any, skip_hidden: bool = True) -> list[tuple[str, Any]]:
    """
    Pares (rótulo, elemento-valor) da div.ficha-dados, na ordem da página.

    O valor normalmente é o div.ficha-campo irmão; no bloco de vagas o
    rótulo está aninhado num ficha-campo e seguido de <table>, então o
    fallback devolve o próximo irmão útil.
    """
    ficha = root if (hasattr(root, "get") and "ficha-dados" in (root.get("class") or [])) \
        else (root.find(class_="ficha-dados") if root else None)
    if not ficha:
        return []

    pairs: list[tuple[str, Any]] = []
    for label_el in ficha.find_all(class_="ficha-label"):
        if skip_hidden and _in_hidden_vinculo(label_el, ficha):
            continue
        label = _clean(label_el.get_text(" ", strip=True))
        if not label:
            continue
        label = label.rstrip(":")
        value = label_el.find_next_sibling(class_="ficha-campo")
        if value is None:
            value = label_el.find_next_sibling(["table", "ul", "div", "p"])
        pairs.append((label, value))
    return pairs


def _ficha_dict(root: Any, skip_hidden: bool = True) -> dict[str, Optional[str]]:
    """
    Dict {rótulo normalizado: texto}. PRIMEIRA ocorrência vence — o vínculo
    ativo do servidor vem antes dos encerrados na ordem do documento.
    """
    out: dict[str, Optional[str]] = {}
    for label, value in _ficha_pairs(root, skip_hidden):
        key = _ficha_key(label)
        if key not in out:
            out[key] = _block_text(value)
    return out


def _ficha_key(label: str) -> str:
    """'Código e-MEC' → 'codigo_e_mec'; remove anotações '(*)'/'(**)'."""
    return _slug(re.sub(r"\(\*+\)", "", label))


def _fget(ficha: dict[str, Optional[str]], *keys: str) -> Optional[str]:
    """Primeiro valor não vazio entre as chaves candidatas."""
    for k in keys:
        v = ficha.get(_slug(k))
        if v:
            return v
    return None


def _ficha_link(root: Any, label_re: str, kind: str) -> tuple[Optional[str], Optional[str]]:
    """
    (id, texto) do primeiro <a> de um campo da ficha cujo rótulo casa label_re.
    Ex.: (`443`, 'Centro de Desenvolvimento Tecnológico') para Unidade.
    """
    for label, value in _ficha_pairs(root):
        if re.search(label_re, _unaccent(label), re.IGNORECASE) and value is not None:
            a = value.find("a", href=True)
            if a:
                return _id_from_href(a["href"], kind), _clean(a.get_text(" ", strip=True))
            return None, _block_text(value)
    return None, None


def _direct_rows(table: Any) -> list:
    """<tr> diretas da tabela — ignora as de tabelas aninhadas (horários)."""
    if table is None:
        return []
    return [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]


def _accordion_sections(container: Any) -> list[tuple[str, str]]:
    """
    Seções de accordion de um container, em ordem: [(rótulo, texto), ...].

    Aplica-se à aba Informações do curso, ao conteúdo da disciplina e à aba
    Informações do projeto — o markup é o mesmo nos três casos.
    """
    if container is None:
        return []
    out: list[tuple[str, str]] = []
    for acc in container.find_all("div", class_="accordion"):
        heading = acc.find(["h3", "h4"], class_="cor-fundo") or acc.find(["h3", "h4"])
        content = acc.find(attrs={"data-content": True})
        if content is not None:
            inner = content.find(class_="accordion-content")
            content = inner if inner is not None else content
        rotulo = _clean(heading.get_text(" ", strip=True)) if heading else None
        texto = _block_text(content)
        if rotulo and texto:
            out.append((rotulo, texto))
    return out


def _parse_grade_horarios(cell: Any) -> list[dict]:
    """
    Horários da tabela aninhada .grade-horarios.

    Layout: colunas = período (Manhã/Tarde/Noite); dentro da célula,
    <span class="grade-horarios-dia">SEG</span> marca o dia e os intervalos
    seguintes herdam o último dia visto (o portal só repete o span quando o
    dia muda, deixando-o vazio para as continuações).
    """
    out: list[dict] = []
    if cell is None:
        return out
    for tab in cell.find_all("table", class_=re.compile(r"grade-horarios")):
        periodos = [_clean(th.get_text(strip=True)) for th in tab.find_all("th")]
        for tr in _direct_rows(tab):
            tds = tr.find_all("td", recursive=False)
            for idx, td in enumerate(tds):
                periodo = periodos[idx] if idx < len(periodos) else None
                dia: Optional[str] = None
                for child in td.children:
                    if getattr(child, "name", None) == "span":
                        marcado = _dia_norm(child.get_text(strip=True))
                        if marcado:
                            dia = marcado
                    elif isinstance(child, str):
                        m = re.search(r"(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})", child)
                        if m:
                            out.append({
                                "periodo":     periodo,
                                "dia_semana":  dia,
                                "hora_inicio": _to_time(m.group(1)),
                                "hora_fim":    _to_time(m.group(2)),
                            })
    return out


def _is_situacao_ativa(situacao: str | None) -> bool:
    if not situacao:
        return True          # portal sem situação publicada → decide pela data de saída
    s = _unaccent(situacao).lower()
    return not any(t in s for t in (_unaccent(x) for x in INACTIVE_SITUACOES))


def _vinculo_ativo(situacao: str | None, data_saida_iso: str | None,
                   hoje: date | None = None) -> bool:
    """
    O vínculo lido está corrente na data do crawl?

    Duas evidências independentes, porque nenhuma aparece sempre:

      * `Situação` — presente nos vínculos permanentes ("Ativo Permanente",
        "Contr. Prof. Substituto"). Valores como "Aposentado"/"Exonerado"
        marcam encerramento.
      * `Data de saída do Cargo` — quando existe E já passou, o vínculo
        acabou. É o ÚNICO sinal em fichas que não publicam `Situação`.

    Checar só `Situação` deixa passar quem tem um vínculo único já expirado
    (a página não recebe o rótulo "(VÍNCULO ENCERRADO)" porque não há outro
    vínculo para colapsar).
    """
    if not _is_situacao_ativa(situacao):
        return False
    if data_saida_iso:
        try:
            return date.fromisoformat(data_saida_iso) >= (hoje or date.today())
        except ValueError:
            return True
    return True


def _is_vigente(data_fim_iso: str | None, hoje: date | None = None) -> bool:
    """Vigente = sem data final OU data final >= hoje."""
    if not data_fim_iso:
        return True
    try:
        return date.fromisoformat(data_fim_iso) >= (hoje or date.today())
    except ValueError:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — acumulador de linhas por tabela, com deduplicação por chave
# ─────────────────────────────────────────────────────────────────────────────

class Dataset:
    """
    Coletor das linhas normalizadas.

    `add` deduplica pela chave primária lógica da tabela. Em colisão, mantém
    a linha com MAIS campos preenchidos — o portal repete registros por
    vínculo/versão e as repetições costumam ser versões pobres da mesma linha.
    """

    #: Chave primária lógica de cada tabela (espelha schema_computacao.sql).
    KEYS: dict[str, tuple[str, ...]] = {
        "curso":                 ("codigo_ufpel",),
        "curso_conceito":        ("curso_codigo", "indicador", "ano"),
        "curso_info_secao":      ("curso_codigo", "secao_slug"),
        "curso_forma_ingresso":  ("curso_codigo", "sigla"),
        "curso_vaga":            ("curso_codigo", "processo", "ano", "semestre", "cota"),
        "curriculo_versao":      ("curso_codigo", "versao"),
        "disciplina":            ("codigo",),
        "disciplina_conteudo":   ("disciplina_codigo", "secao_slug"),
        "disciplina_bibliografia": ("disciplina_codigo", "tipo", "ordem"),
        "disciplina_equivalencia": ("disciplina_codigo", "equivalente_nome", "curso_codigo"),
        "curso_matriz":          ("curso_codigo", "versao", "disciplina_codigo"),
        "matriz_prerequisito":   ("curso_codigo", "versao", "disciplina_codigo", "prereq_codigo"),
        "servidor":              ("servidor_id",),
        "servidor_funcao":       ("servidor_id", "funcao", "unidade"),
        "servidor_formacao":     ("servidor_id", "ordem"),
        "servidor_area_atuacao": ("servidor_id", "ordem"),
        "servidor_curso":        ("servidor_id", "curso_codigo", "origem"),
        "projeto":               ("projeto_id",),
        "projeto_info_secao":    ("projeto_id", "secao_slug"),
        "projeto_equipe":        ("projeto_id", "nome"),
        "servidor_projeto":      ("servidor_id", "projeto_id"),
        "turma":                 ("turma_id",),
        "turma_curriculo":       ("turma_id", "curso_codigo", "versao"),
        "turma_professor":       ("turma_id", "nome"),
        "turma_horario":         ("turma_id", "ordem"),
        "servidor_disciplina_ministrada":
            ("servidor_id", "disciplina_codigo", "ano", "semestre", "codigo_turma"),
    }

    #: Ordem de carga — respeita as dependências de FK.
    ORDER: tuple[str, ...] = (
        "curso", "curso_conceito", "curso_info_secao", "curso_forma_ingresso",
        "curso_vaga", "curriculo_versao",
        "disciplina", "disciplina_conteudo", "disciplina_bibliografia",
        "disciplina_equivalencia", "curso_matriz", "matriz_prerequisito",
        "servidor", "servidor_funcao", "servidor_formacao",
        "servidor_area_atuacao", "servidor_curso",
        "projeto", "projeto_info_secao", "projeto_equipe", "servidor_projeto",
        "turma", "turma_curriculo", "turma_professor", "turma_horario",
        "servidor_disciplina_ministrada",
    )

    def __init__(self) -> None:
        self.tables: dict[str, list[dict]] = {t: [] for t in self.ORDER}
        self._index: dict[str, dict[tuple, int]] = {t: {} for t in self.ORDER}
        self._lock = asyncio.Lock()

    @staticmethod
    def _richness(row: dict) -> int:
        return sum(1 for v in row.values() if v not in (None, "", [], {}))

    def add(self, table: str, row: dict) -> None:
        key = tuple(row.get(c) for c in self.KEYS[table])
        pos = self._index[table].get(key)
        if pos is None:
            self._index[table][key] = len(self.tables[table])
            self.tables[table].append(row)
        elif self._richness(row) > self._richness(self.tables[table][pos]):
            self.tables[table][pos] = row

    def add_many(self, table: str, rows: Iterable[dict]) -> None:
        for r in rows:
            self.add(table, r)

    def get(self, table: str, key: tuple) -> Optional[dict]:
        pos = self._index[table].get(key)
        return self.tables[table][pos] if pos is not None else None

    def has(self, table: str, key: tuple) -> bool:
        return key in self._index[table]

    def counts(self) -> dict[str, int]:
        return {t: len(rows) for t, rows in self.tables.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Cliente HTTP assíncrono
# ─────────────────────────────────────────────────────────────────────────────

class RateLimitedClient:
    """Sessão aiohttp com semáforo, delay polido, retry exponencial e cache."""

    def __init__(self, concurrency: int = DEFAULT_CONCURRENCY,
                 delay: float = DEFAULT_DELAY,
                 cache_dir: str | None = None):
        self._sem = asyncio.Semaphore(concurrency)
        self._delay = delay
        self._cache = Path(cache_dir) if cache_dir else None
        if self._cache:
            self._cache.mkdir(parents=True, exist_ok=True)
        self._session: aiohttp.ClientSession | None = None
        self.stats: dict[str, int] = {"ok": 0, "cache": 0, "skip": 0, "error": 0}

    async def __aenter__(self) -> "RateLimitedClient":
        self._session = aiohttp.ClientSession(
            headers=HEADERS,
            timeout=aiohttp.ClientTimeout(total=45, connect=10),
            connector=aiohttp.TCPConnector(limit=0, ssl=False),
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._session:
            await self._session.close()

    def _cache_path(self, url: str) -> Optional[Path]:
        if not self._cache:
            return None
        return self._cache / (re.sub(r"[^A-Za-z0-9]+", "_", url.replace(BASE_URL, "")) + ".html")

    async def fetch(self, url: str) -> Optional[BeautifulSoup]:
        cached = self._cache_path(url)
        if cached and cached.exists():
            self.stats["cache"] += 1
            return BeautifulSoup(cached.read_text(encoding="utf-8"), "html.parser")

        for attempt in range(1, MAX_RETRY + 1):
            async with self._sem:
                try:
                    assert self._session is not None
                    async with self._session.get(url) as resp:
                        if resp.status in SKIP_HTTP_ERRORS:
                            log.warning("HTTP %s — ignorado: %s", resp.status, url)
                            self.stats["skip"] += 1
                            return None
                        resp.raise_for_status()
                        html = await resp.text(encoding="utf-8", errors="replace")
                    self.stats["ok"] += 1
                    if cached:
                        cached.write_text(html, encoding="utf-8")
                    await asyncio.sleep(self._delay)
                    return BeautifulSoup(html, "html.parser")
                except Exception as exc:
                    if attempt == MAX_RETRY:
                        log.error("Falha definitiva em %s: %s", url, exc)
                        self.stats["error"] += 1
                        return None
                    espera = RETRY_BASE ** attempt + random.uniform(0, 1)
                    log.debug("Retry %d/%d em %.1fs (%s): %s",
                              attempt, MAX_RETRY, espera, type(exc).__name__, url)
                    await asyncio.sleep(espera)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1 — CURSO
# ─────────────────────────────────────────────────────────────────────────────

def _extract_conceitos(soup: BeautifulSoup, curso_codigo: str) -> list[dict]:
    """span.curso-conceito → 'Enade (2021)' + nota '4'."""
    rows: list[dict] = []
    for span in soup.find_all("span", class_="curso-conceito"):
        nome = _txt(span.find(class_="curso-conceito-nome"))
        nota = _txt(span.find(class_="curso-conceito-nota"))
        if not nome or not nota:
            continue
        m = re.match(r"\s*(.+?)\s*\((\d{4})\)\s*$", nome)
        rows.append({
            "curso_codigo": curso_codigo,
            "indicador": _clean(m.group(1)) if m else nome,
            "ano": int(m.group(2)) if m else None,
            "nota": nota,
        })
    return rows


def _extract_vagas(soup: BeautifulSoup, curso_codigo: str) -> list[dict]:
    """
    Matriz (processo × cota) da ficha → formato longo.

    Cada table.tabela-fixed tem cabeçalho [vazio, AC, LB_EP, ..., Total] e uma
    linha de dados cuja primeira célula é o processo ('SISU 2026/1').
    """
    rows: list[dict] = []
    ficha = soup.find(class_="ficha-dados")
    if not ficha:
        return rows

    for label_el in ficha.find_all(class_="ficha-label"):
        if "vagas" not in _unaccent(label_el.get_text(strip=True)).lower():
            continue
        campo = label_el.parent
        for table in campo.find_all("table", class_="tabela-fixed"):
            header = [_clean(th.get_text(strip=True)) for th in table.find_all("th")]
            for tr in _direct_rows(table):
                cells = tr.find_all("td", recursive=False)
                if not cells:
                    continue
                processo_raw = _clean(cells[0].get_text(" ", strip=True))
                if not processo_raw:
                    continue
                ano, semestre = _split_ano_semestre(processo_raw)
                processo = _clean(re.sub(r"\d{4}\s*/\s*\d", "", processo_raw)) or processo_raw
                # cells[0] cobre 2 colunas (colspan=2) → alinha a partir do 3º th
                offset = len(header) - (len(cells) - 1)
                for i, cell in enumerate(cells[1:]):
                    idx = i + offset
                    cota = header[idx] if 0 <= idx < len(header) else None
                    vagas = _to_int(cell.get_text(strip=True))
                    if not cota or vagas is None:
                        continue
                    rows.append({
                        "curso_codigo": curso_codigo,
                        "processo": processo,
                        "ano": ano,
                        "semestre": semestre,
                        "cota": "TOTAL" if _unaccent(cota).lower() == "total" else cota,
                        "vagas": vagas,
                    })
        break
    return rows


def _extract_formas_ingresso(soup: BeautifulSoup, curso_codigo: str) -> list[dict]:
    """Legenda das siglas de cota no rodapé (#notas, bloco '(**)')."""
    rows: list[dict] = []
    rodape = soup.find(class_="conteudo-rodape")
    if not rodape:
        return rows

    coletando = False
    for el in rodape.find_all(["p", "ul", "div", "table"], recursive=False):
        if el.name == "p":
            coletando = "ingresso" in _unaccent(el.get_text(strip=True)).lower()
            continue
        if not coletando:
            continue
        table = el if el.name == "table" else el.find("table")
        if not table:
            continue
        for tr in _direct_rows(table):
            tds = tr.find_all("td", recursive=False)
            if len(tds) < 2:
                continue
            sigla = _clean(tds[0].get_text(strip=True))
            descricao = _clean(tds[1].get_text(" ", strip=True))
            if sigla and descricao:
                rows.append({"curso_codigo": curso_codigo,
                             "sigla": sigla, "descricao": descricao})
    return rows


def _extract_matriz(soup: BeautifulSoup, url: str, curso_codigo: str,
                    versao: str) -> tuple[list[dict], list[dict], set[str]]:
    """
    Aba Matriz Curricular (#curriculo) → (linhas de curso_matriz,
    linhas de matriz_prerequisito, URLs de disciplinas encontradas).

    Um accordion por bloco ('1º Semestre', 'Optativas', 'Complementares').
    Pré-requisitos vêm em span.tabela-detalhe-info dentro da célula da
    disciplina, no formato '22000294 - ALGORITMOS E PROGRAMAÇÃO'.
    """
    matriz: list[dict] = []
    prereqs: list[dict] = []
    disc_urls: set[str] = set()

    secao = soup.find(id="curriculo")
    if not secao:
        return matriz, prereqs, disc_urls

    ordem_global = 0
    for heading in secao.find_all(["h3", "h4"], class_="cor-fundo"):
        rotulo = _clean(heading.get_text(" ", strip=True))
        content = heading.find_next_sibling(attrs={"data-content": True}) \
            or heading.find_next_sibling("div")
        table = content.find("table") if content else None
        if not table or not rotulo:
            continue

        header = [_clean(th.get_text(strip=True)) for th in table.find_all("th")]
        bloco = _bloco_from_rotulo(rotulo)
        for tr in _direct_rows(table):
            cells = tr.find_all("td", recursive=False)
            if not cells:
                continue
            col = {header[i]: cells[i] for i in range(min(len(header), len(cells)))}

            disc_cell = next((c for c in cells
                              if c.find("a", href=re.compile(r"/disciplinas/"))), None)
            if disc_cell is None:
                continue
            links = disc_cell.find_all("a", href=re.compile(r"/disciplinas/"))
            codigo = _id_from_href(links[0]["href"], "disciplinas")
            if not codigo:
                continue
            disc_urls.add(_norm_url(urljoin(url, links[0]["href"])))

            def _cel(*nomes: str) -> Optional[str]:
                for n in nomes:
                    for k, v in col.items():
                        if k and _unaccent(k).lower().startswith(_unaccent(n).lower()):
                            return _clean(v.get_text(" ", strip=True))
                return None

            ordem_global += 1
            matriz.append({
                "curso_codigo": curso_codigo,
                "versao": versao,
                "disciplina_codigo": codigo,
                "bloco": bloco,
                "semestre_rotulo": rotulo,
                "semestre_num": _semestre_num(rotulo),
                "carater": _cel("Caráter", "Carater"),
                "creditos": _to_int(_cel("Cr.", "Créditos", "Creditos")),
                "horas": _to_int(_cel("Horas", "CH")),
                "ordem": ordem_global,
            })

            # Pré-requisitos: '22000294 - ALGORITMOS E PROGRAMAÇÃO' (um por <br>)
            #
            # O span fica na célula "Disciplina / Pré-requisitos", que NÃO é a
            # primeira célula com link de disciplina (a coluna "Código" também
            # linka). Por isso varremos todas as células da linha.
            spans = [sp for c in cells
                     for sp in c.find_all("span", class_="tabela-detalhe-info")]
            for span in spans:
                for linha in (_block_text(span) or "").split("\n"):
                    m = re.match(r"\s*(\d{4,})\s*[-–]\s*(.+)", linha)
                    if not m:
                        continue
                    prereqs.append({
                        "curso_codigo": curso_codigo,
                        "versao": versao,
                        "disciplina_codigo": codigo,
                        "prereq_codigo": m.group(1),
                        "prereq_nome": _clean(m.group(2)),
                    })

    return matriz, prereqs, disc_urls


def _extract_professores_curso(soup: BeautifulSoup, url: str,
                               curso_codigo: str) -> tuple[list[dict], dict[str, str]]:
    """
    Aba Professores (#professores) → (linhas de servidor_curso, {url: id}).

    O portal lista quem ministrou disciplinas no curso nos últimos três
    semestres — é este vínculo que define "professor da computação".
    """
    rows: list[dict] = []
    urls: dict[str, str] = {}
    secao = soup.find(id="professores")
    if not secao:
        return rows, urls

    for td in secao.find_all("td", class_="h-card"):
        link = td.find("a", class_="u-url", href=True) or td.find("a", href=re.compile(r"/servidores/"))
        if not link:
            continue
        servidor_id = _id_from_href(link["href"], "servidores")
        if not servidor_id:
            continue
        urls[_norm_url(urljoin(url, link["href"]))] = servidor_id
        rows.append({"servidor_id": servidor_id, "curso_codigo": curso_codigo,
                     "origem": "aba_professores"})
    return rows, urls


def _extract_turmas_curso(soup: BeautifulSoup, url: str, curso_codigo: str) -> dict:
    """
    Aba Turmas Ofertadas (#turmas).

    Estrutura: h3 com o período letivo → div.versoes → div.versao (h4 'Versão
    do Currículo: 1028 (ATUAL)') → accordion por semestre → tabela de turmas.

    A MESMA turma aparece sob várias versões. Retornamos turmas únicas e, à
    parte, os vínculos (turma, curso, versão, semestre) — é o que evita
    triplicar a turma no banco.
    """
    resultado: dict[str, Any] = {
        "ano": None, "semestre": None, "versoes": [], "turmas": {},
        "turma_curriculo": [], "turma_professor": [], "turma_horario": [],
        "disciplina_urls": set(),
    }
    secao = soup.find(id="turmas")
    if not secao:
        return resultado

    cab = secao.find(["h3", "h2"])
    ano, semestre = _split_ano_semestre(_txt(cab))
    resultado["ano"], resultado["semestre"] = ano, semestre

    for versao_div in secao.find_all(class_="versao"):
        h4 = versao_div.find(["h4", "h3"])
        rotulo_versao = _txt(h4) or ""
        m = re.search(r"(\d+)", rotulo_versao)
        versao = m.group(1) if m else VERSAO_UNICA
        is_atual = "atual" in _unaccent(rotulo_versao).lower()
        resultado["versoes"].append({"curso_codigo": curso_codigo,
                                     "versao": versao, "is_atual": is_atual})

        for heading in versao_div.find_all(["h3", "h4"], class_="cor-fundo"):
            rotulo = _clean(heading.get_text(" ", strip=True))
            content = heading.find_next_sibling(attrs={"data-content": True})
            if not content or not rotulo:
                continue
            for table in content.find_all("table", class_=re.compile(r"tabela-dados")):
                if table.find_parent("table") is not None:
                    continue
                for tr in _direct_rows(table):
                    cells = tr.find_all("td", recursive=False)
                    if len(cells) < 4:
                        continue
                    disc_cell = cells[0]
                    link = disc_cell.find("a", href=re.compile(r"/disciplinas/"))
                    if not link:
                        continue
                    disciplina_codigo = _id_from_href(link["href"], "disciplinas")
                    if not disciplina_codigo:
                        continue
                    resultado["disciplina_urls"].add(_norm_url(urljoin(url, link["href"])))

                    codigo_turma = _clean(cells[-3].get_text(strip=True))
                    if not codigo_turma:
                        continue
                    turma_id = _turma_key(disciplina_codigo, ano, semestre, codigo_turma)

                    # ano/semestre compõem a chave natural da turma e são
                    # NOT NULL no schema; 0 marca "período não identificado"
                    # (o cabeçalho da aba sempre traz o período na prática).
                    resultado["turmas"][turma_id] = {
                        "turma_id": turma_id,
                        "disciplina_codigo": disciplina_codigo,
                        "ano": ano or 0,
                        "semestre": semestre or 0,
                        "codigo_turma": codigo_turma,
                        "vagas": _to_int(cells[-2].get_text(strip=True)),
                        "matriculados": _to_int(cells[-1].get_text(strip=True)),
                    }
                    resultado["turma_curriculo"].append({
                        "turma_id": turma_id,
                        "curso_codigo": curso_codigo,
                        "versao": versao,
                        "semestre_rotulo": rotulo,
                        "semestre_num": _semestre_num(rotulo),
                    })

                    # Professores: o portal publica apenas o nome nesta aba.
                    for span in disc_cell.find_all("span", class_="tabela-detalhe-info"):
                        bruto = _block_text(span) or ""
                        for linha in bruto.split("\n"):
                            papel = ("responsavel" if "respons" in _unaccent(linha).lower()
                                     else "regente" if "regente" in _unaccent(linha).lower()
                                     else None)
                            nome = _clean(re.sub(
                                r"(?i)professor(?:\s+respons[áa]vel\s+pela\s+turma|"
                                r"\s+regente)?\s*:\s*", "", linha))
                            if nome and len(nome) > 3:
                                resultado["turma_professor"].append({
                                    "turma_id": turma_id, "nome": nome,
                                    "servidor_id": None, "papel": papel,
                                })

                    for i, h in enumerate(_parse_grade_horarios(disc_cell), start=1):
                        resultado["turma_horario"].append({"turma_id": turma_id, "ordem": i, **h})

    return resultado


def extract_curso(soup: BeautifulSoup, url: str, codigo: str,
                  ds: Dataset) -> tuple[set[str], dict[str, str]]:
    """
    Extrai um curso inteiro para o Dataset.
    Retorna (URLs de disciplinas, {url do professor: servidor_id}).
    """
    _strip_noise(soup)
    ficha = _ficha_dict(soup)
    alvo = TARGET_CURSOS.get(codigo, {})

    nivel_grau = _fget(ficha, "Nível / Grau", "Nível")
    nivel, grau = (None, None)
    if nivel_grau:
        partes = [p.strip() for p in nivel_grau.split("/", 1)]
        nivel = partes[0] or None
        grau = partes[1].title() if len(partes) > 1 and partes[1] else alvo.get("grau")

    unidade_id, unidade_nome = _ficha_link(soup, r"^unidade", "unidades")
    coord_id, coord_nome = _ficha_link(soup, r"^coordenador", "servidores")

    ds.add("curso", {
        "codigo_ufpel": codigo,
        "nome": _fget(ficha, "Nome do Curso / Conceitos", "Nome do Curso", "Nome")
                or alvo.get("nome"),
        "nivel": nivel,
        "grau": grau or alvo.get("grau"),
        "modalidade": _fget(ficha, "Modalidade"),
        "turno": _fget(ficha, "Turno"),
        "codigo_emec": _fget(ficha, "Código e-MEC"),
        "codigo_capes": _fget(ficha, "Código CAPES"),
        "unidade_nome": unidade_nome,
        "unidade_id": unidade_id,
        "programa": _fget(ficha, "Programa"),
        "coordenador_nome": coord_nome,
        "coordenador_id": coord_id,
        "criacao_reconhecimento": _fget(ficha, "Criação e Reconhecimento"),
        "url": url,
        "crawled_at": _now(),
    })

    ds.add_many("curso_conceito", _extract_conceitos(soup, codigo))
    ds.add_many("curso_vaga", _extract_vagas(soup, codigo))
    ds.add_many("curso_forma_ingresso", _extract_formas_ingresso(soup, codigo))

    # Aba Informações — uma linha por seção (vira um embedding cada)
    for i, (rotulo, texto) in enumerate(_accordion_sections(soup.find(id="informacoes")), start=1):
        ds.add("curso_info_secao", {
            "curso_codigo": codigo, "secao": rotulo,
            "secao_slug": _slug(rotulo), "ordem": i, "texto": texto,
        })

    # Turmas: registram as versões de currículo antes da matriz
    turmas = _extract_turmas_curso(soup, url, codigo)
    ds.add_many("curriculo_versao", turmas["versoes"])

    # Versão da matriz publicada em #curriculo: a marcada ATUAL; se o curso
    # não tem turmas (logo, nenhuma versão), usa a versão sintética.
    versoes = turmas["versoes"]
    atuais = [v["versao"] for v in versoes if v["is_atual"]]
    if atuais:
        versao_matriz = atuais[0]
    elif versoes:
        versao_matriz = max(versoes, key=lambda v: _to_int(v["versao"]) or 0)["versao"]
    else:
        versao_matriz = VERSAO_UNICA
        ds.add("curriculo_versao", {"curso_codigo": codigo,
                                    "versao": VERSAO_UNICA, "is_atual": True})

    matriz, prereqs, disc_urls_matriz = _extract_matriz(soup, url, codigo, versao_matriz)
    ds.add_many("curso_matriz", matriz)
    ds.add_many("matriz_prerequisito", prereqs)

    prof_rows, prof_urls = _extract_professores_curso(soup, url, codigo)
    ds.add_many("servidor_curso", prof_rows)

    ds.add_many("turma", list(turmas["turmas"].values()))
    ds.add_many("turma_curriculo", turmas["turma_curriculo"])
    ds.add_many("turma_professor", turmas["turma_professor"])
    ds.add_many("turma_horario", turmas["turma_horario"])

    log.info("[CURSO %s] %s — %d na matriz (v%s), %d versões, %d professores, %d turmas",
             codigo, _fget(ficha, "Nome do Curso / Conceitos") or alvo.get("nome"),
             len(matriz), versao_matriz, len(versoes) or 1,
             len(prof_rows), len(turmas["turmas"]))

    return disc_urls_matriz | turmas["disciplina_urls"], prof_urls


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2 — DISCIPLINA
# ─────────────────────────────────────────────────────────────────────────────

_BIBLIO_TIPOS = (("basica", r"b[áa]sica"), ("complementar", r"complementar"))


def _parse_bibliografia(texto_el: Any) -> list[tuple[str, str]]:
    """
    Bibliografia → [(tipo, referência), ...].

    O portal marca os grupos com <p><strong>Bibliografia Básica:</strong></p>
    seguidos de <ul><li>. Sem <ul>, cai para quebra por linha.
    """
    if texto_el is None:
        return []
    out: list[tuple[str, str]] = []
    tipo_atual = "nao_classificada"

    for el in texto_el.find_all(["p", "strong", "ul", "ol"]):
        rotulo = _unaccent(el.get_text(" ", strip=True) or "").lower()
        if el.name in ("p", "strong"):
            for tipo, padrao in _BIBLIO_TIPOS:
                if re.search(padrao, rotulo):
                    tipo_atual = tipo
                    break
        elif el.name in ("ul", "ol"):
            for li in el.find_all("li"):
                ref = _clean(li.get_text(" ", strip=True))
                if ref and len(ref) > 3:
                    out.append((tipo_atual, ref))

    if not out:
        for linha in (_block_text(texto_el) or "").split("\n"):
            ref = _clean(linha)
            if ref and len(ref) > 10 and not re.match(r"(?i)bibliografia", ref):
                out.append(("nao_classificada", ref))
    return out


def _extract_equivalentes(soup: BeautifulSoup, url: str,
                          disciplina_codigo: str) -> list[dict]:
    """
    Aba 'Disciplinas Equivalentes' → tabela Disciplina | Curso.

    Útil para o estudante ("posso aproveitar o que fiz em outro curso?"). O
    portal referencia a equivalente por id interno (/disciplinas/id/N), não
    por código — guardamos o id como referência opaca.
    """
    rows: list[dict] = []
    heading = next((h for h in soup.find_all(["h2", "h3"])
                    if "equivalent" in _unaccent(h.get_text(strip=True)).lower()), None)
    if not heading:
        return rows

    table = heading.find_next("table")
    if not table:
        return rows

    for tr in _direct_rows(table):
        cells = tr.find_all("td", recursive=False)
        if len(cells) < 2:
            continue
        disc_link = cells[0].find("a", href=True)
        curso_link = cells[1].find("a", href=True)
        nome = _clean((disc_link or cells[0]).get_text(" ", strip=True))
        if not nome:
            continue
        rows.append({
            "disciplina_codigo": disciplina_codigo,
            "equivalente_nome": nome,
            "equivalente_ref": _id_from_href(disc_link["href"], "disciplinas") if disc_link else None,
            "equivalente_url": _norm_url(urljoin(url, disc_link["href"])) if disc_link else None,
            "curso_codigo": _id_from_href(curso_link["href"], "cursos") if curso_link else None,
            "curso_nome": _clean((curso_link or cells[1]).get_text(" ", strip=True)),
        })
    return rows


def extract_disciplina(soup: BeautifulSoup, url: str, ds: Dataset) -> Optional[str]:
    """Extrai uma disciplina para o Dataset. Retorna o código, ou None."""
    _strip_noise(soup)
    ficha = _ficha_dict(soup)

    codigo = _fget(ficha, "CÓDIGO", "Código", "Código da Disciplina") \
        or _id_from_href(url, "disciplinas")
    if not codigo:
        return None

    unidade_id, unidade_nome = _ficha_link(soup, r"unidade", "unidades")

    ds.add("disciplina", {
        "codigo": codigo,
        "nome": _fget(ficha, "Nome da Atividade", "Nome", "Disciplina") or _txt(soup.find("h1")),
        "tipo_atividade": _fget(ficha, "Tipo de Atividade"),
        "periodicidade": _fget(ficha, "Periodicidade"),
        "creditos": _to_int(_fget(ficha, "CRÉDITOS", "Créditos")),
        "carga_horaria": _to_int(_fget(ficha, "Carga Horária")),
        "ch_teorica": _to_int(_fget(ficha, "CARGA HORÁRIA TEÓRICA")),
        "ch_pratica": _to_int(_fget(ficha, "CARGA HORÁRIA PRÁTICA")),
        "ch_obrigatoria": _to_int(_fget(ficha, "CARGA HORÁRIA OBRIGATÓRIA")),
        "freq_aprovacao": _fget(ficha, "FREQUÊNCIA APROVAÇÃO"),
        "unidade_nome": unidade_nome or _fget(ficha, "Unidade responsável", "Unidade"),
        "unidade_id": unidade_id,
        "url": url,
        "crawled_at": _now(),
    })

    # Ementa / Objetivos / Conteúdo Programático / Bibliografia
    informacoes = soup.find(id="informacoes")
    for i, (rotulo, texto) in enumerate(_accordion_sections(informacoes), start=1):
        ds.add("disciplina_conteudo", {
            "disciplina_codigo": codigo, "secao": rotulo,
            "secao_slug": _slug(rotulo), "ordem": i, "texto": texto,
        })

    # Bibliografia também em linhas individuais (dado enumerável)
    if informacoes is not None:
        for acc in informacoes.find_all("div", class_="accordion"):
            heading = acc.find(["h3", "h4"], class_="cor-fundo") or acc.find(["h3", "h4"])
            if not heading or "bibliografia" not in _unaccent(heading.get_text(strip=True)).lower():
                continue
            content = acc.find(attrs={"data-content": True})
            inner = content.find(class_="accordion-content") if content else None
            por_tipo: dict[str, int] = defaultdict(int)
            for tipo, ref in _parse_bibliografia(inner or content):
                por_tipo[tipo] += 1
                ds.add("disciplina_bibliografia", {
                    "disciplina_codigo": codigo, "tipo": tipo,
                    "ordem": por_tipo[tipo], "referencia": ref,
                })

    ds.add_many("disciplina_equivalencia", _extract_equivalentes(soup, url, codigo))
    return codigo


# ─────────────────────────────────────────────────────────────────────────────
# FASE 3 — PROFESSOR (SERVIDOR)
# ─────────────────────────────────────────────────────────────────────────────

_FORMACAO_RE = re.compile(
    r"^\s*(?P<nivel>[^()]+?)\s+em\s+(?P<area>[^()]+?)\s*"
    r"\((?P<inst>.+?),\s*(?P<ano>\d{4})\)\s*$"
)


def _parse_formacao(linha: str) -> dict:
    """'Doutorado em Oceanografia (Univ. Federal do Rio Grande, 2019)' → campos."""
    m = _FORMACAO_RE.match(linha)
    if not m:
        return {"nivel": None, "area": None, "instituicao": None, "ano": None,
                "texto_original": linha}
    return {
        "nivel": _clean(m.group("nivel")),
        "area": _clean(m.group("area")),
        "instituicao": _clean(m.group("inst")),
        "ano": int(m.group("ano")),
        "texto_original": linha,
    }


def _extract_lattes(soup: BeautifulSoup) -> dict[str, Any]:
    """
    Currículo (#lattes): Resumo, Formação acadêmica, Áreas de atuação.
    Cada seção é um <h3> seguido de <p> com itens separados por <br>.
    """
    out: dict[str, Any] = {"resumo": None, "formacao": [], "areas": []}
    secao = soup.find(id="lattes")
    if not secao:
        return out

    for h3 in secao.find_all(["h3", "h4"]):
        rotulo = _unaccent(h3.get_text(strip=True)).lower()
        partes: list[str] = []
        for sib in h3.next_siblings:
            nome_tag = getattr(sib, "name", None)
            if nome_tag in ("h3", "h4"):
                break
            if nome_tag is None:                      # nó de texto solto
                if isinstance(sib, str) and sib.strip():
                    partes.append(_clean(sib) or "")
                continue
            if "ficha-label" in (sib.get("class") or []):
                continue              # rodapé "Informações extraídas do Lattes"
            t = _block_text(sib)
            if t:
                partes.append(t)

        texto = _clean("\n".join(p for p in partes if p))
        if not texto:
            continue
        itens = [i for i in (_clean(l) for l in texto.split("\n")) if i]
        if "resumo" in rotulo:
            out["resumo"] = texto.replace("\n", " ")
        elif "formacao" in rotulo:
            out["formacao"] = itens
        elif "area" in rotulo or "atuacao" in rotulo:
            out["areas"] = itens
    return out


def _extract_contatos(soup: BeautifulSoup) -> dict[str, Optional[str]]:
    """
    Links de contato (ul.contatos) — fora da div#lattes.

    É AQUI que fica o link do Lattes; procurá-lo dentro de #lattes devolve
    sempre vazio.
    """
    out: dict[str, Optional[str]] = {"lattes_url": None, "email": None}
    for ul in soup.find_all("ul", class_="contatos"):
        for a in ul.find_all("a", href=True):
            href = a["href"]
            if "lattes.cnpq" in href and not out["lattes_url"]:
                out["lattes_url"] = href
            elif href.startswith("mailto:") and not out["email"]:
                out["email"] = href[len("mailto:"):].strip() or None
    if not out["lattes_url"]:
        a = soup.find("a", href=re.compile(r"lattes\.cnpq", re.I))
        if a:
            out["lattes_url"] = a["href"]
    return out


def _extract_servidor_projetos(soup: BeautifulSoup, url: str,
                               servidor_id: str) -> tuple[list[dict], dict[str, str]]:
    """
    Aba Projetos (#projetos) → (linhas de servidor_projeto, {url: projeto_id}).

    A tabela é ÚNICA e usa th.tabela-quebra como separador de ênfase
    (Ensino / Extensão / Pesquisa); a ênfase deve ser rastreada linha a linha.
    Projetos encerrados vêm marcados com tr.finalizado — sinal mais confiável
    que só comparar datas.
    """
    rows: list[dict] = []
    urls: dict[str, str] = {}
    secao = soup.find(id="projetos")
    if not secao:
        return rows, urls

    for table in secao.find_all("table"):
        if table.find_parent("table") is not None:
            continue
        enfase: Optional[str] = None
        for tr in _direct_rows(table):
            quebra = tr.find("th", class_="tabela-quebra")
            if quebra is not None:
                enfase = _clean(quebra.get_text(strip=True))
                continue
            if tr.find("th") is not None and not tr.find("td"):
                continue                                   # linha de cabeçalho
            if "finalizado" in (tr.get("class") or []):
                continue                                   # projeto encerrado
            cells = tr.find_all("td", recursive=False)
            if not cells:
                continue
            link = cells[0].find("a", href=re.compile(r"/projetos/"))
            if not link:
                continue
            projeto_id = _id_from_href(link["href"], "projetos")
            if not projeto_id:
                continue

            data_inicio = _to_date_iso(_txt(cells[1])) if len(cells) > 1 else None
            data_fim = _to_date_iso(_txt(cells[2])) if len(cells) > 2 else None
            if not _is_vigente(data_fim):
                continue

            papel_span = cells[0].find("span", class_="tabela-detalhe-info")
            urls[_norm_url(urljoin(url, link["href"]))] = projeto_id
            rows.append({
                "servidor_id": servidor_id,
                "projeto_id": projeto_id,
                "papel": _txt(papel_span),
                "enfase": enfase,
                "ch_semanal": _to_int(_txt(cells[3])) if len(cells) > 3 else None,
                "data_inicio": data_inicio,
                "data_fim": data_fim,
            })
    return rows, urls


def _extract_servidor_disciplinas(soup: BeautifulSoup, servidor_id: str) -> list[dict]:
    """
    Aba 'Disciplinas ministradas nos três últimos semestres' (#disciplinas).
    Colunas: Ano | Turma | Disciplina | CH | Curso.

    Esta é a única fonte que liga (disciplina, período, turma) a um
    servidor_id — a aba de turmas do curso publica apenas o nome.
    """
    rows: list[dict] = []
    secao = soup.find(id="disciplinas")
    if not secao:
        return rows

    for table in secao.find_all("table"):
        if table.find_parent("table") is not None:
            continue
        for tr in _direct_rows(table):
            cells = tr.find_all("td", recursive=False)
            if len(cells) < 3:
                continue
            disc_cell = next((c for c in cells
                              if c.find("a", href=re.compile(r"/disciplinas/"))), None)
            if disc_cell is None:
                continue
            disc_link = disc_cell.find("a", href=re.compile(r"/disciplinas/"))
            disciplina_codigo = _id_from_href(disc_link["href"], "disciplinas")
            if not disciplina_codigo:
                continue

            curso_cell = next((c for c in cells
                               if c.find("a", href=re.compile(r"/cursos/"))), None)
            curso_link = curso_cell.find("a", href=re.compile(r"/cursos/")) if curso_cell else None
            ano, semestre = _split_ano_semestre(_txt(cells[0]))
            codigo_turma = _clean(cells[1].get_text(strip=True)) if len(cells) > 1 else None

            rows.append({
                "servidor_id": servidor_id,
                "disciplina_codigo": disciplina_codigo,
                "ano": ano,
                "semestre": semestre,
                "codigo_turma": codigo_turma,
                "carga_horaria": _clean(cells[-2].get_text(strip=True)) if len(cells) >= 2 else None,
                "curso_codigo": _id_from_href(curso_link["href"], "cursos") if curso_link else None,
                "curso_nome": _txt(curso_link) if curso_link else None,
                "turma_id": (_turma_key(disciplina_codigo, ano, semestre, codigo_turma)
                             if codigo_turma else None),
            })
    return rows


def extract_servidor(soup: BeautifulSoup, url: str, servidor_id: str,
                     ds: Dataset) -> dict[str, str]:
    """
    Extrai um professor para o Dataset. Retorna {url do projeto: projeto_id}.
    Servidores com vínculo encerrado são descartados.
    """
    _strip_noise(soup)
    ficha = _ficha_dict(soup)              # só o vínculo corrente (não colapsado)

    nome = _fget(ficha, "Nome do Servidor", "Nome") or _txt(soup.find("h1"))
    if not nome:
        return {}

    situacao = _fget(ficha, "Situação")
    data_saida = _to_date_iso(_fget(ficha, "Data de saída do Cargo"))
    ativo = _vinculo_ativo(situacao, data_saida)
    if not ativo:
        # Mantido no dataset, com vinculo_ativo=False: o professor pode ter
        # ministrado turmas nos últimos semestres, e descartá-lo quebraria o
        # histórico (turma_professor, disciplinas ministradas). Quem quer só
        # o corpo docente atual filtra por vinculo_ativo.
        log.debug("Vínculo não corrente (situação=%s, saída=%s): %s",
                  situacao, data_saida, nome)

    lattes = _extract_lattes(soup)
    contatos = _extract_contatos(soup)
    lotacao_id, lotacao_nome = _ficha_link(soup, r"^lotacao", "unidades")

    ds.add("servidor", {
        "servidor_id": servidor_id,
        "nome": nome,
        "matricula_siape": _fget(ficha, "Matrícula SIAPE", "Matrícula"),
        "categoria": _fget(ficha, "Categoria"),
        "cargo": _fget(ficha, "Cargo", "Função"),
        "classe_nivel": _fget(ficha, "Classe / Nível"),
        "titulacao": _fget(ficha, "Titulação"),
        "lotacao_nome": lotacao_nome or _fget(ficha, "Lotação"),
        "lotacao_id": lotacao_id,
        "regime_jornada": _fget(ficha, "Regime / Jornada de Trabalho", "Regime de Trabalho"),
        "situacao": situacao,
        "vinculo_ativo": ativo,
        "data_ingresso_servico": _to_date_iso(_fget(ficha, "Data de ingresso no serviço público")),
        "data_ingresso_ufpel": _to_date_iso(_fget(ficha, "Data de ingresso na UFPel")),
        "data_ingresso_cargo": _to_date_iso(_fget(ficha, "Data de ingresso no cargo")),
        "data_saida_cargo": data_saida,
        "email": contatos["email"] or _fget(ficha, "E-mail"),
        "lattes_url": contatos["lattes_url"],
        "curriculo_resumo": lattes["resumo"],
        "url": url,
        "crawled_at": _now(),
    })

    # Função / gratificação — ex.: "Coordenador de Curso de Graduação /
    # Colegiado do Curso de Ciência da Computação"
    funcao_unidade = _fget(ficha, "Função / Unidade")
    if funcao_unidade:
        partes = [p.strip() for p in funcao_unidade.split("/", 1)]
        grat = _fget(ficha, "Gratificação / Data inicial") or ""
        grat_partes = [p.strip() for p in grat.split("/", 1)] if grat else []
        ds.add("servidor_funcao", {
            "servidor_id": servidor_id,
            "funcao": partes[0],
            "unidade": partes[1] if len(partes) > 1 else None,
            "gratificacao": grat_partes[0] if grat_partes else None,
            "data_inicio": _to_date_iso(grat_partes[1]) if len(grat_partes) > 1 else None,
        })

    for i, linha in enumerate(lattes["formacao"], start=1):
        ds.add("servidor_formacao", {"servidor_id": servidor_id, "ordem": i,
                                     **_parse_formacao(linha)})

    for i, area in enumerate(lattes["areas"], start=1):
        partes = [p.strip() for p in re.split(r"\s+[-–]\s+", area, maxsplit=1)]
        ds.add("servidor_area_atuacao", {
            "servidor_id": servidor_id, "ordem": i, "area": area,
            "area_geral": partes[0] or None,
            "subarea": partes[1] if len(partes) > 1 else None,
        })

    proj_rows, proj_urls = _extract_servidor_projetos(soup, url, servidor_id)
    ds.add_many("servidor_projeto", proj_rows)
    ds.add_many("servidor_disciplina_ministrada",
                _extract_servidor_disciplinas(soup, servidor_id))
    return proj_urls


# ─────────────────────────────────────────────────────────────────────────────
# FASE 4 — PROJETO
# ─────────────────────────────────────────────────────────────────────────────

def _extract_projeto_equipe(soup: BeautifulSoup, projeto_id: str) -> list[dict]:
    """
    Aba Equipe → Nome | CH Semanal | Data inicial | Data final.
    Integrantes sem link são discentes/externos (servidor_id NULL).
    """
    rows: list[dict] = []
    secao = soup.find(id="equipe")
    if not secao:
        return rows

    for table in secao.find_all("table"):
        if table.find_parent("table") is not None:
            continue
        for tr in _direct_rows(table):
            cells = tr.find_all("td", recursive=False)
            if not cells:
                continue
            link = cells[0].find("a", href=re.compile(r"/servidores/"))
            nome = _clean(cells[0].get_text(" ", strip=True))
            if not nome:
                continue
            data_fim = _to_date_iso(_txt(cells[3])) if len(cells) > 3 else None
            if not _is_vigente(data_fim):
                continue
            servidor_id = _id_from_href(link["href"], "servidores") if link else None
            rows.append({
                "projeto_id": projeto_id,
                "nome": nome,
                "servidor_id": servidor_id,
                "is_servidor": servidor_id is not None,
                "ch_semanal": _to_int(_txt(cells[1])) if len(cells) > 1 else None,
                "data_inicio": _to_date_iso(_txt(cells[2])) if len(cells) > 2 else None,
                "data_fim": data_fim,
            })
    return rows


def extract_projeto(soup: BeautifulSoup, url: str, projeto_id: str,
                    ds: Dataset) -> bool:
    """Extrai um projeto vigente. Retorna False se estiver encerrado."""
    _strip_noise(soup)
    ficha = _ficha_dict(soup)

    periodo = _fget(ficha, "Data inicial - Data final") or ""
    datas = re.findall(r"\d{2}/\d{2}/\d{4}", periodo)
    data_inicio = _to_date_iso(datas[0]) if datas else _to_date_iso(_fget(ficha, "Data de Início"))
    data_fim = _to_date_iso(datas[1]) if len(datas) > 1 else _to_date_iso(_fget(ficha, "Data de Término"))

    if not _is_vigente(data_fim):
        log.debug("Projeto encerrado (fim=%s) — ignorado: %s", data_fim, projeto_id)
        return False

    titulo = _fget(ficha, "Nome do Projeto", "Título", "Nome") or _txt(soup.find("h1"))
    if not titulo:
        return False

    coord_id, coord_nome = _ficha_link(soup, r"coordenador", "servidores")

    ds.add("projeto", {
        "projeto_id": projeto_id,
        "titulo": titulo,
        "enfase": _fget(ficha, "Ênfase"),
        "resumo": _fget(ficha, "Resumo"),
        "coordenador_nome": coord_nome or _fget(ficha, "Coordenador Atual", "Coordenador"),
        "coordenador_id": coord_id,
        "unidade_origem": _fget(ficha, "Unidade de Origem", "Unidade"),
        "area_cnpq": _fget(ficha, "Área CNPq", "Área"),
        "eixo_tematico": _fget(ficha, "Eixo Temático (Principal - Afim)", "Eixo Temático"),
        "linha_extensao": _fget(ficha, "Linha de Extensão"),
        "data_inicio": data_inicio,
        "data_fim": data_fim,
        "url": url,
        "crawled_at": _now(),
    })

    for i, (rotulo, texto) in enumerate(_accordion_sections(soup.find(id="informacoes")), start=1):
        ds.add("projeto_info_secao", {
            "projeto_id": projeto_id, "secao": rotulo,
            "secao_slug": _slug(rotulo), "ordem": i, "texto": texto,
        })

    ds.add_many("projeto_equipe", _extract_projeto_equipe(soup, projeto_id))
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Reconciliação — liga turma → servidor_id e limpa referências órfãs
# ─────────────────────────────────────────────────────────────────────────────

def reconciliar(ds: Dataset) -> dict[str, int]:
    """
    Passo final, depois de todas as fases:

      1. turma_professor.servidor_id — a aba de turmas do curso publica só o
         nome. Resolve-se pelo par (turma_id) vindo de
         servidor_disciplina_ministrada e, como reforço, por nome normalizado.
      2. servidor_curso ganha origem='turma' para professores que ministram
         turmas dos cursos mas não aparecem na aba Professores.
      3. Remove linhas cujas FKs não existem (disciplinas/turmas/projetos que
         ficaram fora do escopo do crawl) — a carga no Postgres falharia.
    """
    stats = {"prof_por_turma": 0, "prof_por_nome": 0,
             "servidor_curso_extra": 0, "orfas_removidas": 0}

    # ── 1a. turma_id vindo da página do professor (fonte com servidor_id) ───
    turma_para_servidor: dict[str, str] = {}
    for row in ds.tables["servidor_disciplina_ministrada"]:
        if row.get("turma_id"):
            turma_para_servidor[row["turma_id"]] = row["servidor_id"]

    nome_para_servidor = {_norm_nome(s["nome"]): s["servidor_id"]
                          for s in ds.tables["servidor"] if s.get("nome")}

    for row in ds.tables["turma_professor"]:
        if row.get("servidor_id"):
            continue
        sid = turma_para_servidor.get(row["turma_id"])
        if sid:
            row["servidor_id"] = sid
            stats["prof_por_turma"] += 1
            continue
        sid = nome_para_servidor.get(_norm_nome(row.get("nome")))
        if sid:
            row["servidor_id"] = sid
            stats["prof_por_nome"] += 1

    # ── 2. professores alcançados apenas via turma ──────────────────────────
    turma_curso = defaultdict(set)
    for tc in ds.tables["turma_curriculo"]:
        turma_curso[tc["turma_id"]].add(tc["curso_codigo"])

    for row in ds.tables["turma_professor"]:
        sid = row.get("servidor_id")
        if not sid or not ds.has("servidor", (sid,)):
            continue
        for curso_codigo in turma_curso.get(row["turma_id"], ()):
            key = (sid, curso_codigo, "turma")
            if not ds.has("servidor_curso", key):
                ds.add("servidor_curso", {"servidor_id": sid,
                                          "curso_codigo": curso_codigo,
                                          "origem": "turma"})
                stats["servidor_curso_extra"] += 1

    # ── 3. poda de linhas órfãs (FK inexistente) ───────────────────────────
    existe = {
        "curso": {r["codigo_ufpel"] for r in ds.tables["curso"]},
        "disciplina": {r["codigo"] for r in ds.tables["disciplina"]},
        "servidor": {r["servidor_id"] for r in ds.tables["servidor"]},
        "projeto": {r["projeto_id"] for r in ds.tables["projeto"]},
        "turma": {r["turma_id"] for r in ds.tables["turma"]},
        "versao": {(r["curso_codigo"], r["versao"]) for r in ds.tables["curriculo_versao"]},
    }

    def _podar(tabela: str, teste) -> None:
        antes = len(ds.tables[tabela])
        ds.tables[tabela] = [r for r in ds.tables[tabela] if teste(r)]
        stats["orfas_removidas"] += antes - len(ds.tables[tabela])

    _podar("turma", lambda r: r["disciplina_codigo"] in existe["disciplina"])
    existe["turma"] &= {r["turma_id"] for r in ds.tables["turma"]}

    _podar("curso_matriz", lambda r: r["disciplina_codigo"] in existe["disciplina"]
           and (r["curso_codigo"], r["versao"]) in existe["versao"])
    matriz_ok = {(r["curso_codigo"], r["versao"], r["disciplina_codigo"])
                 for r in ds.tables["curso_matriz"]}
    _podar("matriz_prerequisito",
           lambda r: (r["curso_codigo"], r["versao"], r["disciplina_codigo"]) in matriz_ok)

    _podar("disciplina_conteudo", lambda r: r["disciplina_codigo"] in existe["disciplina"])
    _podar("disciplina_bibliografia", lambda r: r["disciplina_codigo"] in existe["disciplina"])
    _podar("disciplina_equivalencia", lambda r: r["disciplina_codigo"] in existe["disciplina"])
    _podar("servidor_curso", lambda r: r["servidor_id"] in existe["servidor"]
           and r["curso_codigo"] in existe["curso"])
    _podar("servidor_projeto", lambda r: r["servidor_id"] in existe["servidor"]
           and r["projeto_id"] in existe["projeto"])
    _podar("projeto_info_secao", lambda r: r["projeto_id"] in existe["projeto"])
    _podar("projeto_equipe", lambda r: r["projeto_id"] in existe["projeto"])
    _podar("turma_curriculo", lambda r: r["turma_id"] in existe["turma"]
           and (r["curso_codigo"], r["versao"]) in existe["versao"])
    _podar("turma_professor", lambda r: r["turma_id"] in existe["turma"])
    _podar("turma_horario", lambda r: r["turma_id"] in existe["turma"])
    _podar("servidor_disciplina_ministrada", lambda r: r["servidor_id"] in existe["servidor"])

    # servidor_id/turma_id que não existem viram NULL (colunas nullable)
    for row in ds.tables["turma_professor"]:
        if row.get("servidor_id") not in existe["servidor"]:
            row["servidor_id"] = None
    for row in ds.tables["projeto_equipe"]:
        if row.get("servidor_id") not in existe["servidor"]:
            row["servidor_id"] = None
            row["is_servidor"] = False
    for row in ds.tables["servidor_disciplina_ministrada"]:
        if row.get("turma_id") not in existe["turma"]:
            row["turma_id"] = None

    log.info("[Reconciliação] professores por turma=%d, por nome=%d, "
             "vínculos extra curso=%d, linhas órfãs removidas=%d",
             stats["prof_por_turma"], stats["prof_por_nome"],
             stats["servidor_curso_extra"], stats["orfas_removidas"])
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Orquestração
# ─────────────────────────────────────────────────────────────────────────────

class ComputacaoCrawler:
    """Deep-crawl dos cursos de Computação em 4 fases + reconciliação."""

    def __init__(self, cursos: list[str] | None = None,
                 concurrency: int = DEFAULT_CONCURRENCY,
                 delay: float = DEFAULT_DELAY,
                 cache_dir: str | None = None):
        self.cursos = cursos or list(TARGET_CURSOS)
        self.concurrency = concurrency
        self.delay = delay
        self.cache_dir = cache_dir
        self.ds = Dataset()

    async def _fase_cursos(self, client: RateLimitedClient) -> tuple[set[str], dict[str, str]]:
        disc_urls: set[str] = set()
        prof_urls: dict[str, str] = {}

        async def _one(codigo: str) -> None:
            url = f"{BASE_URL}/cursos/cod/{codigo}"
            soup = await client.fetch(url)
            if soup is None:
                log.error("[CURSO %s] indisponível", codigo)
                return
            try:
                d, p = extract_curso(soup, url, codigo, self.ds)
                disc_urls.update(d)
                prof_urls.update(p)
            except Exception as exc:
                log.exception("[CURSO %s] erro na extração: %s", codigo, exc)

        await asyncio.gather(*(_one(c) for c in self.cursos))
        return disc_urls, prof_urls

    async def _fase_disciplinas(self, client: RateLimitedClient,
                                urls: set[str]) -> None:
        async def _one(url: str) -> None:
            soup = await client.fetch(url)
            if soup is None:
                return
            try:
                extract_disciplina(soup, url, self.ds)
            except Exception as exc:
                log.exception("[DISCIPLINA] erro em %s: %s", url, exc)

        await asyncio.gather(*(_one(u) for u in sorted(urls)))

    async def _fase_professores(self, client: RateLimitedClient,
                                prof_urls: dict[str, str]) -> dict[str, str]:
        proj_urls: dict[str, str] = {}

        async def _one(url: str, servidor_id: str) -> None:
            soup = await client.fetch(url)
            if soup is None:
                return
            try:
                proj_urls.update(extract_servidor(soup, url, servidor_id, self.ds))
            except Exception as exc:
                log.exception("[PROFESSOR] erro em %s: %s", url, exc)

        await asyncio.gather(*(_one(u, i) for u, i in sorted(prof_urls.items())))
        return proj_urls

    async def _fase_projetos(self, client: RateLimitedClient,
                             proj_urls: dict[str, str]) -> None:
        async def _one(url: str, projeto_id: str) -> None:
            soup = await client.fetch(url)
            if soup is None:
                return
            try:
                extract_projeto(soup, url, projeto_id, self.ds)
            except Exception as exc:
                log.exception("[PROJETO] erro em %s: %s", url, exc)

        await asyncio.gather(*(_one(u, i) for u, i in sorted(proj_urls.items())))

    async def crawl(self) -> dict[str, Any]:
        log.info("[Crawler] Cursos de Computação: %s",
                 ", ".join(f"{c} ({TARGET_CURSOS[c]['nome']})"
                           for c in self.cursos if c in TARGET_CURSOS))

        async with RateLimitedClient(self.concurrency, self.delay, self.cache_dir) as client:
            disc_urls, prof_urls = await self._fase_cursos(client)
            log.info("[Fase 1] %d cursos | %d disciplinas | %d professores a visitar",
                     len(self.ds.tables["curso"]), len(disc_urls), len(prof_urls))

            await self._fase_disciplinas(client, disc_urls)
            log.info("[Fase 2] %d disciplinas capturadas", len(self.ds.tables["disciplina"]))

            proj_urls = await self._fase_professores(client, prof_urls)
            servidores = self.ds.tables["servidor"]
            n_ativos = sum(1 for s in servidores if s.get("vinculo_ativo"))
            log.info("[Fase 3] %d professores (%d com vínculo corrente, %d sem) | "
                     "%d projetos vigentes a visitar",
                     len(servidores), n_ativos, len(servidores) - n_ativos, len(proj_urls))

            await self._fase_projetos(client, proj_urls)
            log.info("[Fase 4] %d projetos capturados", len(self.ds.tables["projeto"]))
            log.info("[HTTP] %s", client.stats)

        reconciliar(self.ds)

        return {
            "_meta": {
                "gerado_em": _now(),
                "fonte": BASE_URL,
                "cursos_alvo": {c: TARGET_CURSOS[c] for c in self.cursos if c in TARGET_CURSOS},
                "schema": "schema_computacao.sql",
                "ordem_carga": list(Dataset.ORDER),
                "contagens": self.ds.counts(),
            },
            **self.ds.tables,
        }


def save_json(dados: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(dados, fh, ensure_ascii=False, indent=2)
    tamanho = Path(output_path).stat().st_size / 1024
    log.info("[Saída] %s (%.1f KB)", output_path, tamanho)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Crawler dos cursos de Computação da UFPel (saída relacional).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cursos", nargs="+", choices=list(TARGET_CURSOS), default=None,
                   metavar="COD", help="Subconjunto dos cursos alvo")
    p.add_argument("--output", default=DEFAULT_OUTPUT, metavar="FILE",
                   help="JSON de saída (tabelas normalizadas)")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, metavar="N",
                   help="Requisições simultâneas")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY, metavar="SECS",
                   help="Delay entre requisições por worker (crawling polido)")
    p.add_argument("--cache-dir", default=None, metavar="DIR",
                   help="Diretório de cache do HTML (evita re-baixar em testes)")
    p.add_argument("--verbose", action="store_true", help="Log em nível DEBUG")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.verbose:
        log.setLevel(logging.DEBUG)

    crawler = ComputacaoCrawler(cursos=args.cursos, concurrency=args.concurrency,
                                delay=args.delay, cache_dir=args.cache_dir)
    dados = asyncio.run(crawler.crawl())
    save_json(dados, args.output)

    print()
    print("=" * 66)
    print("  Contagem por tabela")
    print("=" * 66)
    for tabela in Dataset.ORDER:
        print(f"    {tabela:<34} {len(dados[tabela]):>6}")
    print("=" * 66)
    print(f"  Próximo passo: python load_computacao.py --input {args.output} --schema")
    print(f"  (inspeção sem banco: python load_computacao.py --input {args.output} --dry-run)")

    if not dados["curso"]:
        sys.exit("Nenhum curso capturado — verifique a conectividade com o portal.")


if __name__ == "__main__":
    main()
