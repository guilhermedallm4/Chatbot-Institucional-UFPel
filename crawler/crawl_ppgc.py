"""
Crawler do PPGC — Programa de Pós-Graduação em Computação da UFPel
=============================================================================
Dataset SEPARADO do portal (`crawl_portal_computacao.py`) e do institucional
(`crawl_computacao.py`). Tabelas `ppgc_*`, schema `schema_ppgc.sql`.

Por que separado
----------------
O enunciado é explícito: o acervo do PPGC é acionado quando o roteador conclui
que a resposta NÃO está no portal. Separar em tabelas próprias transforma essa
decisão num filtro de tabela em vez de um `WHERE` sobre um índice misturado —
e o índice vetorial do PPGC fica com sinal denso de pós-graduação, sem 459
notícias de graduação competindo por vizinhança.

O que é coletado
----------------
  ppgc_pagina / _secao      126 páginas do Programa (subárvore /ppgc/, editais
                            soltos na raiz e a tradução em inglês)
  ppgc_post                 notícias com categoria `ppgc` + as que a categoria
                            legada esqueceu (ver `wp_common.post_e_ppgc`)
  ppgc_edital               editais normalizados: tipo, nível, ano, semestre,
                            número oficial ("253/2025")
  ppgc_edital_documento     os PDFs de cada edital, tipados pela seção da
                            página. `texto_extraido` nasce NULL — é o gancho
                            para a ingestão futura dos PDFs
  ppgc_normativo            regimento, resoluções e portarias (PDF + a íntegra
                            das que têm página própria)
  ppgc_linha_pesquisa       as 5 linhas, com a descrição que o candidato lê
  ppgc_docente              corpo docente, com Lattes e id do institucional
  ppgc_faq                  FAQ de alunos, pergunta a pergunta, com âncora
  ppgc_requisito            requisitos e prazos de mestrado/doutorado
  ppgc_disciplina           disciplinas ofertadas, com ementa e responsável
  ppgc_calendario_evento    430 eventos do Google Calendar público do PPGC
  ppgc_defesa               defesas de dissertação e tese anunciadas
  ppgc_link                 todos os links, para rastreabilidade

A ingestão futura dos PDFs
--------------------------
`ppgc_edital_documento` é a fila de trabalho: uma linha por PDF, já com URL,
tipo, edital de origem e data. Quando os PDFs entrarem, basta preencher
`texto_extraido` e gravar `ppgc_documento_chunk` — `load_ppgc.py` vetoriza os
chunks em `emb_ppgc_documento` sem que nada mais mude. O `sha256` existe para
que a reingestão pule arquivo que não mudou.

Uso
---
    python crawl_ppgc.py
    python crawl_ppgc.py --output ppgc.json --cache-dir .cache_wp
    python crawl_ppgc.py --sem-calendario        # pula o .ics do Google

Requisitos: pip install -r requirements_crawler.txt
"""

from __future__ import annotations

import argparse
import base64
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qs, unquote, urlparse

from bs4 import BeautifulSoup, Tag

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crawl_portal_computacao import (                          # noqa: E402
    parsear_faq, parsear_pessoas, registrar_links_e_documentos,
)
from wp_common import (                                        # noqa: E402
    API_URL, SITE_URL, Dataset, WPClient, ano_semestre_de, anos_citados,
    caminho_da_pagina, contar_palavras, data_por_extenso, dividir_secoes,
    extrair_links, html_para_texto, idioma_de, iso, limpar_texto,
    nome_arquivo_de, normalizar, pagina_e_ppgc, parse_data_wp, post_e_ppgc,
    resumo_de, semestre_de, titulo_wp,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("crawl_ppgc")

DEFAULT_OUTPUT = "ppgc.json"


class PPGCDataset(Dataset):
    KEYS = {
        "ppgc_crawl_meta":        ("chave",),
        "ppgc_pagina":            ("pagina_id",),
        "ppgc_pagina_secao":      ("pagina_id", "ordem"),
        "ppgc_post":              ("post_id",),
        "ppgc_post_tag":          ("post_id", "tag_slug"),
        "ppgc_edital":            ("edital_id",),
        "ppgc_edital_documento":  ("edital_id", "url"),
        "ppgc_documento":         ("url",),
        "ppgc_normativo":         ("normativo_id",),
        "ppgc_linha_pesquisa":    ("slug",),
        "ppgc_docente":           ("nome",),
        "ppgc_faq":               ("pagina_id", "ordem"),
        "ppgc_requisito":         ("nivel", "requisito"),
        "ppgc_disciplina":        ("nome", "ano", "semestre"),
        "ppgc_calendario_evento": ("uid",),
        "ppgc_defesa":            ("tipo", "discente", "data"),
        "ppgc_link":              ("origem_tipo", "origem_id", "ordem"),
    }
    ORDER = (
        "ppgc_crawl_meta", "ppgc_pagina", "ppgc_pagina_secao", "ppgc_post",
        "ppgc_post_tag", "ppgc_documento", "ppgc_edital",
        "ppgc_edital_documento", "ppgc_normativo", "ppgc_linha_pesquisa",
        "ppgc_docente", "ppgc_faq", "ppgc_requisito", "ppgc_disciplina",
        "ppgc_calendario_evento", "ppgc_defesa", "ppgc_link",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Editais
# ─────────────────────────────────────────────────────────────────────────────

#: Tipo do edital, na ORDEM em que as regras são testadas — a ordem é a regra.
#: "Edital de bolsa de doutorado sanduíche" precisa casar `bolsa_sanduiche`
#: ANTES de `ingresso_regular` chegar a ver a palavra "doutorado"; e
#: "classificação para alocação de bolsas" precisa casar antes de qualquer
#: coisa que olhe "mestrado e doutorado".
_TIPOS_EDITAL: tuple[tuple[str, str], ...] = (
    ("bolsa_sanduiche",     r"sanduiche|\bpdse\b"),
    ("bolsa_posdoc",        r"pos-?doutorado|\bpnpd\b|\bpipd\b"),
    ("bolsa_classificacao", r"classificacao.*bolsa|bolsa.*classificacao"
                            r"|alocacao.*bolsa|edital interno"),
    ("professor_visitante", r"professor[a]?\s*visitante"),
    ("dinter",              r"\bdinter\b"),
    ("ingresso_especial",   r"(aluno|estudante|discente)[\s/-]*especial"),
    ("ingresso_regular",    r"(aluno|estudante|discente)[\s/-]*regular"
                            r"|selecao.*(mestrado|doutorado)"),
    ("resultado",           r"resultado|homologacao"),
)

_NIVEL_EDITAL: tuple[tuple[str, str], ...] = (
    ("posdoc",    r"pos-?doutorado|\bpnpd\b|\bpipd\b"),
    ("docente",   r"professor[a]?\s*visitante|credenciamento"),
    ("ambos",     r"mestrado\s*(e|/|,)\s*doutorado|doutorado\s*(e|/|,)\s*mestrado"),
    ("doutorado", r"\bdoutorado\b|\btese\b"),
    ("mestrado",  r"\bmestrado\b|\bdissertacao\b"),
)

#: Título da seção da página → tipo do documento. É o sinal mais forte que a
#: página oferece: numa página de seleção, o mesmo tipo de PDF aparece sob
#: "Edital", "Outros Formulários" e "Processo Seletivo", e só o bloco
#: distingue o edital do formulário de autodeclaração.
_TIPOS_DOC_SECAO: tuple[tuple[str, str], ...] = (
    ("edital",        r"^edital"),
    ("retificacao",   r"retifica|errata"),
    ("formulario",    r"formulario|inscricao"),
    ("resultado",     r"resultado|processo seletivo|homologa|classificacao"
                      r"|selecionad|aprovad"),
    ("cronograma",    r"cronograma|calendario|horario"),
    ("anexo",         r"anexo|documento|orienta|instrucao"),
)

#: Fallback pelo texto do próprio link, quando a seção não decide.
_TIPOS_DOC_TEXTO: tuple[tuple[str, str], ...] = (
    ("retificacao",   r"retifica|errata"),
    ("resultado",     r"resultado|homologa|classifica|selecionad|aprovad"
                      r"|lista (preliminar|final)"),
    ("formulario",    r"formulario|autodeclara|requerimento|ficha"),
    ("cronograma",    r"cronograma|horario|calendario"),
    ("edital",        r"^edital|edital n"),
    ("anexo",         r"anexo"),
)

#: Número oficial do edital, como sai da PROAP/SEI: "Edital Nº 253/2025".
_RE_NUMERO_EDITAL = re.compile(
    r"edital\s*(?:interno\s*)?(?:n[º°o.]?\s*)?(\d{1,4}\s*/\s*\d{4})", re.I)


def _primeiro(regras: Iterable[tuple[str, str]], alvo: str,
              padrao: Optional[str] = None) -> Optional[str]:
    for rotulo, padrao_re in regras:
        if re.search(padrao_re, alvo):
            return rotulo
    return padrao


def tipo_do_edital(titulo: str, slug: str) -> str:
    return _primeiro(_TIPOS_EDITAL, normalizar(f"{titulo} {slug}"), "outro")


#: Editais destes tipos valem para mestrado E doutorado por definição, mesmo
#: quando o texto só cita um dos dois — "Seleção de aluno regular 2026/2"
#: mencionava doutorado primeiro (por causa do calendário de entrevistas) e
#: saía marcado só como doutorado, escondendo o edital de quem busca mestrado.
_TIPOS_AMBOS_NIVEIS = {"ingresso_regular", "ingresso_especial",
                       "bolsa_classificacao"}


def nivel_do_edital(titulo: str, slug: str, texto: str, tipo: str) -> Optional[str]:
    """
    Nível do edital, a partir do sinal mais confiável disponível.

    A ordem importa e foi corrigida por um caso real. O edital de seleção
    regular 2026/2 vale para mestrado E doutorado, mas a página não escreve
    "mestrado" em lugar nenhum — a única ocorrência de nível no corpo é
    "entrevistas para os candidatos ao doutorado". Deduzir o nível do corpo
    marcava o edital como exclusivo de doutorado, e quem perguntasse pelo
    edital de mestrado de 2026/2 não o encontraria.

    A regra passou a ser: AUSÊNCIA DE MENÇÃO NÃO É EVIDÊNCIA DE EXCLUSÃO.
      1. título ou slug dizem o nível → é afirmação deliberada, vale;
      2. edital de seleção/bolsa sem qualificação → 'ambos', que é como o
         Programa de fato os publica;
      3. só então o corpo decide — o que serve aos tipos cujo nível é
         intrínseco (sanduíche, pós-doc, professor visitante).
    """
    pelo_titulo = _primeiro(_NIVEL_EDITAL, normalizar(f"{titulo} {slug}"))
    if pelo_titulo:
        return pelo_titulo
    if tipo in _TIPOS_AMBOS_NIVEIS:
        return "ambos"
    return _primeiro(_NIVEL_EDITAL, normalizar(texto))


#: Rótulos inequívocos no texto do link, testados ANTES da seção. Dentro do
#: bloco "Processo Seletivo" convivem resultados e o calendário de entrevistas;
#: a seção diria "resultado" para os dois.
_TIPOS_DOC_TEXTO_FORTE: tuple[tuple[str, str], ...] = (
    ("retificacao", r"retifica|errata"),
    ("cronograma",  r"cronograma|calendario|horario|entrevista"),
)


def tipo_do_documento(titulo_secao: Optional[str], texto_link: Optional[str]) -> str:
    texto = normalizar(texto_link or "")
    forte = _primeiro(_TIPOS_DOC_TEXTO_FORTE, texto)
    if forte:
        return forte
    por_secao = _primeiro(_TIPOS_DOC_SECAO, normalizar(titulo_secao or ""))
    if por_secao:
        return por_secao
    return _primeiro(_TIPOS_DOC_TEXTO, texto, "outro")


#: Uma página é um edital quando o slug/título diz que é, ou quando ela é
#: filha de uma das três páginas-índice de editais.
_RE_PAGINA_EDITAL = re.compile(
    r"^(edital|editais|selecao|selecao-de|resultado-homologacao)")
_INDICES_EDITAL = {"editais-de-ingresso", "editais-de-bolsas",
                   "editais-de-professor-visitante"}


def pagina_e_edital(slug: str, titulo: str, caminho_pai: Optional[str]) -> bool:
    if normalizar(slug) in _INDICES_EDITAL:
        return False              # é a lista DE editais, não um edital
    if _RE_PAGINA_EDITAL.match(normalizar(slug)):
        return True
    if caminho_pai and caminho_pai.rsplit("/", 1)[-1] in _INDICES_EDITAL:
        return True
    return bool(re.match(r"^edital\b", normalizar(titulo)))


# ─────────────────────────────────────────────────────────────────────────────
# Normativos
# ─────────────────────────────────────────────────────────────────────────────

#: "resolucao-01-2019-regulamenta-a-producao-minima…" → (resolucao, 01, 2019)
_RE_NORMATIVO_SLUG = re.compile(
    r"^(resolucao|portaria|regimento|instrucao)[-_](\d{1,3})[-_](\d{4})")
#: "01/2024 – Normas para alocação e manutenção de bolsas"
_RE_NORMATIVO_TEXTO = re.compile(
    r"^(?:(resolu[çc][ãa]o|portaria)\s*)?(\d{1,3})\s*[/-]\s*(\d{4})\s*"
    r"[–—\-:]*\s*(.*)$", re.I)
#: "Regimento 2020", "Planejamento Estratégico 2025-2028"
_RE_NORMATIVO_ANO = re.compile(r"^(.*?)\s*(\d{4})(?:\s*[-–]\s*\d{4})?\s*$")


def _id_normativo(tipo: str, numero: Optional[str], ano: Optional[int]) -> str:
    partes = [tipo]
    if numero:
        partes.append(f"{int(numero):02d}")
    if ano:
        partes.append(str(ano))
    return "-".join(partes)


def tipo_normativo_da_secao(titulo_secao: Optional[str]) -> tuple[str, bool]:
    """Título da seção → (tipo, vigente). "Resoluções revogadas" → vigente=False."""
    n = normalizar(titulo_secao or "")
    vigente = "revogad" not in n and "anterior" not in n
    if "regimento" in n:
        return "regimento", vigente
    if "planejamento" in n:
        return "planejamento_estrategico", vigente
    if "portaria" in n:
        return "portaria", vigente
    if "resolu" in n:
        return "resolucao", vigente
    return "outro", vigente


# ─────────────────────────────────────────────────────────────────────────────
# Calendário do Google
# ─────────────────────────────────────────────────────────────────────────────

def descobrir_calendario(html: str) -> Optional[str]:
    """
    Extrai o e-mail do Google Calendar embutido na página do calendário.

    A página `/ppgc/calendario-ppgc/` é só um `<iframe>` do Google Agenda —
    zero texto. Se parássemos aí, o dataset teria a página "Calendário PPGC"
    sem nenhuma data dentro, e "quando é o prazo de matrícula?" não teria
    resposta. O `src` do iframe carrega o id do calendário em base64
    (`src=aW5mLnVmcGVs…`), e todo calendário público do Google publica um
    `.ics` — é de lá que saem os 430 eventos com data real.
    """
    soup = BeautifulSoup(html, "lxml")
    for iframe in soup.find_all("iframe", src=True):
        src = limpar_texto(iframe["src"])
        if "calendar.google.com" not in src:
            continue
        for valor in parse_qs(urlparse(src.replace("&#038;", "&")).query).get("src", []):
            valor = unquote(valor)
            if "@" in valor:
                return valor
            try:
                decodificado = base64.b64decode(valor + "==").decode("utf-8")
            except Exception:                                  # noqa: BLE001
                continue
            # ignora o calendário de feriados nacionais, que vem no mesmo embed
            if "@" in decodificado and "holiday" not in decodificado:
                return decodificado
    return None


def _desdobrar_ics(texto: str) -> list[str]:
    """Junta as continuações de linha do iCalendar (linha seguinte com espaço)."""
    linhas: list[str] = []
    for linha in texto.splitlines():
        if linha[:1] in (" ", "\t") and linhas:
            linhas[-1] += linha[1:]
        else:
            linhas.append(linha)
    return linhas


def _valor_ics(bruto: str) -> str:
    return (bruto.replace("\\n", "\n").replace("\\,", ",")
            .replace("\\;", ";").replace("\\\\", "\\").strip())


def parsear_ics(texto: str, url: str) -> list[dict]:
    """
    `.ics` → linhas de `ppgc_calendario_evento`.

    Só os campos que o estudante pergunta: o que é, quando começa, quando
    termina. `DTEND` no iCalendar de dia inteiro é EXCLUSIVO (um evento de um
    dia vai de 15/07 a 16/07), então subtraímos um dia — sem isso, todo prazo
    apareceria valendo um dia a mais do que vale.
    """
    eventos: list[dict] = []
    atual: Optional[dict] = None
    for linha in _desdobrar_ics(texto):
        if linha.startswith("BEGIN:VEVENT"):
            atual = {}
            continue
        if linha.startswith("END:VEVENT"):
            if atual and atual.get("uid") and atual.get("data_inicio"):
                eventos.append(atual)
            atual = None
            continue
        if atual is None or ":" not in linha:
            continue
        cabecalho, _, valor = linha.partition(":")
        campo = cabecalho.split(";", 1)[0].upper()
        dia_inteiro = "VALUE=DATE" in cabecalho.upper()

        def data(bruto: str) -> Optional[datetime]:
            bruto = bruto.strip().rstrip("Z")
            for formato in ("%Y%m%dT%H%M%S", "%Y%m%d"):
                try:
                    return datetime.strptime(bruto, formato)
                except ValueError:
                    continue
            return None

        if campo == "UID":
            atual["uid"] = _valor_ics(valor)
        elif campo == "SUMMARY":
            atual["titulo"] = _valor_ics(valor)
        elif campo == "DESCRIPTION":
            atual["descricao"] = _valor_ics(valor) or None
        elif campo == "LOCATION":
            atual["local"] = _valor_ics(valor) or None
        elif campo == "DTSTART":
            d = data(valor)
            if d:
                atual["data_inicio"] = d.date().isoformat()
                atual["hora_inicio"] = None if dia_inteiro else d.time().isoformat()
                atual["dia_inteiro"] = dia_inteiro
                atual["ano"] = d.year
                atual["mes"] = d.month
                atual["semestre"] = semestre_de(d.date())
                atual["data_por_extenso"] = data_por_extenso(d.date())
        elif campo == "DTEND":
            d = data(valor)
            if d:
                fim = d.date() - timedelta(days=1) if dia_inteiro else d.date()
                atual["data_fim"] = fim.isoformat()
        elif campo == "LAST-MODIFIED":
            d = data(valor)
            atual["atualizado_em"] = d.isoformat() if d else None
        elif campo == "STATUS":
            atual["situacao"] = _valor_ics(valor).lower()

    for e in eventos:
        e.setdefault("titulo", None)
        e["url_calendario"] = url
    return eventos


# ─────────────────────────────────────────────────────────────────────────────
# Requisitos, disciplinas, defesas
# ─────────────────────────────────────────────────────────────────────────────

def parsear_requisitos(pagina: dict, html: str) -> list[dict]:
    """
    Tabelas "REQUISITOS OBRIGATÓRIOS × PRAZO" do FAQ → uma linha por requisito.

    É a informação com maior chance de ser perguntada e a que mais dói errar:
    "até quando preciso comprovar proficiência em inglês?" tem resposta
    diferente no mestrado (3ª matrícula) e no doutorado (5ª matrícula). Em
    coluna, isso é `WHERE nivel = 'mestrado'`; em texto corrido, é uma aposta.

    A mesma tabela mistura os blocos obrigatório e opcional, separados por uma
    linha-cabeçalho no meio ("REQUISITOS OPCIONAIS | PRAZO") — daí o
    `obrigatorio` ser um estado que vira ao encontrar essa linha, e não uma
    propriedade da tabela.
    """
    soup = BeautifulSoup(html, "lxml")
    requisitos: list[dict] = []
    for tabela in soup.find_all("table"):
        contexto = ""
        anterior = tabela.find_previous(["h1", "h2", "h3", "h4", "h5", "p"])
        for _ in range(4):
            if anterior is None:
                break
            contexto = normalizar(anterior.get_text(" ", strip=True)) + " " + contexto
            if "requisitos para" in contexto:
                break
            anterior = anterior.find_previous(["h1", "h2", "h3", "h4", "h5", "p"])
        if "requisito" not in contexto:
            continue
        nivel = ("doutorado" if "doutorado" in contexto else
                 "mestrado" if "mestrado" in contexto else None)
        if nivel is None:
            continue

        obrigatorio = True
        for tr in tabela.find_all("tr"):
            celulas = [limpar_texto(td.get_text(" ", strip=True))
                       for td in tr.find_all(["td", "th"])]
            if len(celulas) < 2 or not celulas[0]:
                continue
            marcador = normalizar(celulas[0])
            if "requisitos obrigatorios" in marcador:
                obrigatorio = True
                continue
            if "requisitos opcionais" in marcador:
                obrigatorio = False
                continue
            requisitos.append({
                "nivel": nivel,
                "requisito": celulas[0],
                "prazo": celulas[1] or None,
                "obrigatorio": obrigatorio,
                "pagina_id": pagina["id"],
                "url": pagina["link"],
            })
    return requisitos


#: "Aprendizado de Máquina (4 créditos)"
_RE_DISCIPLINA = re.compile(r"^(.*?)\s*\((\d+)\s*cr[ée]ditos?\)\s*$", re.I)
_CAMPOS_DISCIPLINA = (
    ("aluno_especial", r"dispon[íi]vel para aluno especial\??"),
    ("responsavel",    r"respons[áa]ve(?:l|is)\s*:?"),
    ("ementa",         r"ementa\s*:?"),
    ("horario",        r"hor[áa]rio\s*:?"),
    ("modalidade",     r"modalidade\s*:?"),
)


def parsear_disciplinas(pagina: dict, html: str) -> list[dict]:
    """
    Lista de disciplinas ofertadas → nome, créditos, responsável, ementa.

    O `<li>` usa `<br>` como separador de campo, então quebramos por linha e
    casamos cada rótulo. A ementa é o campo que justifica a tabela: é ela que
    responde "tem alguma disciplina sobre visão computacional?" — e o loader a
    vetoriza junto com o nome, num embedding por disciplina.
    """
    soup = BeautifulSoup(html, "lxml")
    ano, semestre = ano_semestre_de(pagina["slug"],
                                    titulo_wp(pagina["title"]["rendered"]))
    disciplinas: list[dict] = []
    for li in soup.find_all("li"):
        if li.find("li"):
            continue
        forte = li.find(["strong", "b"])
        if forte is None:
            continue
        cabecalho = limpar_texto(forte.get_text(" ", strip=True))
        m = _RE_DISCIPLINA.match(cabecalho)
        if not m:
            continue

        linhas = [limpar_texto(x) for x in
                  html_para_texto(str(li)).lstrip("- ").split("\n")]
        registro = {
            "nome": m.group(1).strip(),
            "creditos": int(m.group(2)),
            "ano": ano,
            "semestre": semestre,
            "pagina_id": pagina["id"],
            "url": pagina["link"],
            "aluno_especial": None, "responsavel": None,
            "ementa": None, "horario": None, "modalidade": None,
        }
        for linha in linhas:
            for campo, rotulo in _CAMPOS_DISCIPLINA:
                m2 = re.match(rf"^{rotulo}\s*(.*)$", linha, re.I)
                if m2 and m2.group(1).strip():
                    registro[campo] = m2.group(1).strip()
                    break
        if registro["aluno_especial"] is not None:
            registro["aluno_especial"] = normalizar(
                registro["aluno_especial"]).startswith("sim")
        disciplinas.append(registro)
    return disciplinas


#: "Daiane Fonseca Freitas, 16 de Abril de 2026, 14:00"
_RE_DEFESA = re.compile(
    r"^(?P<nome>[^,]+),\s*(?P<dia>\d{1,2})\s+de\s+(?P<mes>[A-Za-zÀ-ÿ]+)\s+de\s+"
    r"(?P<ano>\d{4})(?:\s*,\s*(?P<hora>\d{1,2}[:h]\d{2}))?", re.I)


def parsear_defesas(pagina: dict, html: str) -> list[dict]:
    """Página de defesas → discente, data, hora e link da sala."""
    from wp_common import MES_POR_NOME_NORM

    soup = BeautifulSoup(html, "lxml")
    defesas: list[dict] = []
    tipo: Optional[str] = None
    for no in (soup.body or soup).find_all(["p", "li"]):
        texto = limpar_texto(no.get_text(" ", strip=True))
        if not texto:
            continue
        n = normalizar(texto)
        if "defesas de" in n:
            tipo = "tese" if "tese" in n else "dissertacao"
            continue
        m = _RE_DEFESA.match(texto)
        if not m or tipo is None:
            continue
        mes = MES_POR_NOME_NORM.get(normalizar(m.group("mes")))
        if not mes:
            continue
        try:
            data = datetime(int(m.group("ano")), mes, int(m.group("dia"))).date()
        except ValueError:
            continue
        link = no.find("a", href=True) or no.find_next("a", href=True)
        defesas.append({
            "tipo": tipo,
            "discente": m.group("nome").strip(),
            "data": data.isoformat(),
            "ano": data.year,
            "hora": (m.group("hora") or "").replace("h", ":") or None,
            "local": limpar_texto(link.get_text(" ", strip=True)) if link else None,
            "link": link["href"] if link else None,
            "pagina_id": pagina["id"],
            "url": pagina["link"],
        })
    return defesas


def parsear_linhas_pesquisa(pagina: dict, html: str,
                            slug_por_caminho: dict[str, int]) -> list[dict]:
    """
    Índice das linhas de pesquisa → nome, descrição e página da linha.

    A descrição fica no parágrafo SEGUINTE ao link, não dentro dele. E o link
    aponta para `/ppgc/linhas-de-pesquisas/fundamentos-da-computacao/` —
    permalink antigo, com "pesquisas" no plural, que não corresponde ao
    caminho real da página (`ppgc/fundamentos-da-computacao`). Por isso o
    casamento é pelo ÚLTIMO segmento da URL, e não pela URL inteira.
    """
    soup = BeautifulSoup(html, "lxml")
    linhas: list[dict] = []
    for p in soup.find_all("p"):
        a = p.find("a", href=True)
        if a is None:
            continue
        nome = limpar_texto(a.get_text(" ", strip=True))
        if len(nome) < 6:
            continue
        slug = [s for s in urlparse(a["href"]).path.split("/") if s][-1:]
        slug = slug[0] if slug else normalizar(nome).replace(" ", "-")

        descricao_partes: list[str] = []
        irmao = p.find_next_sibling()
        while irmao is not None and irmao.name == "p" and not irmao.find("a"):
            texto = limpar_texto(irmao.get_text(" ", strip=True))
            if texto:
                descricao_partes.append(texto)
            irmao = irmao.find_next_sibling()

        linhas.append({
            "slug": slug,
            "nome": nome,
            "descricao": "\n".join(descricao_partes) or None,
            "url": a["href"],
            "pagina_id": slug_por_caminho.get(slug),
            "pagina_indice_id": pagina["id"],
        })
    return linhas


# ─────────────────────────────────────────────────────────────────────────────
# Montagem
# ─────────────────────────────────────────────────────────────────────────────

def montar(cliente: WPClient, *, coletar_calendario: bool = True) -> PPGCDataset:
    ds = PPGCDataset()

    log.info("[1/6] Taxonomias e mídia")
    categorias = cliente.fetch_all("categories", hide_empty="false")
    tags = cliente.fetch_all("tags", hide_empty="false")
    autores = cliente.fetch_all("users")
    media = cliente.fetch_all("media")
    slug_por_categoria = {c["id"]: c["slug"] for c in categorias}
    slug_por_tag = {t["id"]: t["slug"] for t in tags}
    nome_por_autor = {u["id"]: titulo_wp(u.get("name")) for u in autores}
    media_por_url = {m["source_url"]: m for m in media if m.get("source_url")}

    log.info("[2/6] Páginas do Programa")
    paginas = cliente.fetch_all("pages", status="publish")
    por_id = {p["id"]: p for p in paginas}
    caminhos = {p["id"]: caminho_da_pagina(p, por_id) for p in paginas}
    paginas_ppgc = [p for p in paginas
                    if pagina_e_ppgc(caminhos[p["id"]], p["slug"],
                                     titulo_wp(p["title"]["rendered"]))]
    ids_ppgc = {p["id"] for p in paginas_ppgc}
    slug_para_id = {p["slug"]: p["id"] for p in paginas_ppgc}
    log.info("      %d de %d páginas são do PPGC", len(paginas_ppgc), len(paginas))

    log.info("[3/6] Conteúdo, editais e documentos")
    n_edital = n_doc = n_norm = 0
    pagina_calendario: Optional[dict] = None
    especiais: list[tuple[str, dict, str]] = []

    for p in paginas_ppgc:
        html = p["content"]["rendered"]
        texto = html_para_texto(html)
        caminho = caminhos[p["id"]]
        caminho_pai = caminho.rsplit("/", 1)[0] if "/" in caminho else None
        pai = p.get("parent") or None
        publicado = parse_data_wp(p.get("date"))
        modificado = parse_data_wp(p.get("modified"))
        titulo = titulo_wp(p["title"]["rendered"])
        slug_norm = normalizar(p["slug"])
        secoes = dividir_secoes(html)
        e_edital = pagina_e_edital(p["slug"], titulo, caminho_pai)

        categoria = ("edital" if e_edital else
                     "normativo" if "regimento" in caminho or
                     _RE_NORMATIVO_SLUG.match(slug_norm) else
                     "faq" if "faq" in slug_norm else
                     "calendario" if "calendario" in slug_norm else
                     "docentes" if slug_norm in ("docentes", "faculty") else
                     "linha_pesquisa" if "linhas-de-pesquisa" in slug_norm else
                     "disciplinas" if "disciplina" in slug_norm else
                     "institucional")

        ds.add("ppgc_pagina", {
            "pagina_id": p["id"],
            "slug": p["slug"],
            "caminho": caminho,
            "titulo": titulo,
            "url": p["link"],
            "pagina_pai_id": pai if pai in ids_ppgc else None,
            "caminho_pai": caminho_pai,
            "nivel": caminho.count("/"),
            "categoria": categoria,
            "idioma": idioma_de(p["link"], p["slug"]),
            "data_publicacao": iso(publicado.date()) if publicado else None,
            "data_modificacao": iso(modificado.date()) if modificado else None,
            "resumo": resumo_de(texto),
            "texto": texto or None,
            "n_palavras": contar_palavras(texto),
            "anos_citados": anos_citados(titulo, texto) or None,
        })

        for ordem, secao in enumerate(secoes):
            if not (secao["texto"] or "").strip():
                continue
            ds.add("ppgc_pagina_secao", {
                "pagina_id": p["id"], "ordem": ordem,
                "titulo_secao": secao["titulo"],
                "nivel_titulo": secao["nivel"] or None,
                "ancora": secao["ancora"],
                "texto": secao["texto"],
                "n_palavras": contar_palavras(secao["texto"]),
                "url": p["link"] + (f"#{secao['ancora']}" if secao["ancora"] else ""),
            })

        registrar_links_e_documentos(ds, "pagina", p["id"], titulo, p["link"],
                                     html, media_por_url,
                                     tabela_link="ppgc_link",
                                     tabela_doc="ppgc_documento")

        if e_edital:
            n_edital += 1
            n_doc += _registrar_edital(ds, p, titulo, texto, secoes,
                                       caminho_pai, publicado, modificado)

        if _RE_NORMATIVO_SLUG.match(slug_norm):
            n_norm += _registrar_normativo_pagina(ds, p, titulo, texto)
        if slug_norm in ("regimento-e-resolucoes",):
            n_norm += _registrar_normativos_indice(ds, p, secoes, media_por_url)

        if "faq" in slug_norm:
            especiais.append(("faq", p, html))
        if slug_norm in ("docentes", "professores-dinter-iffar"):
            especiais.append(("docentes", p, html))
        if slug_norm in ("linhas-de-pesquisa", "research-lines"):
            especiais.append(("linhas", p, html))
        if "lista-de-disciplinas" in slug_norm:
            especiais.append(("disciplinas", p, html))
        if slug_norm == "defesas":
            especiais.append(("defesas", p, html))
        if "calendario" in slug_norm and "iframe" in html:
            pagina_calendario = p

    log.info("      %d páginas de edital · %d documentos · %d normativos",
             n_edital, n_doc, n_norm)

    log.info("[4/6] Parsers específicos")
    contagens = {"faq": 0, "requisito": 0, "docente": 0, "linha": 0,
                 "disciplina": 0, "defesa": 0}
    for tipo, p, html in especiais:
        if tipo == "faq":
            for faq in parsear_faq(p, html):
                ds.add("ppgc_faq", faq)
                contagens["faq"] += 1
            for req in parsear_requisitos(p, html):
                ds.add("ppgc_requisito", req)
                contagens["requisito"] += 1
        elif tipo == "docentes":
            for pessoa in parsear_pessoas(p, html):
                ds.add("ppgc_docente", {
                    "nome": pessoa["nome"],
                    "url_perfil": pessoa["url_perfil"],
                    "lattes_url": pessoa["lattes_url"],
                    "servidor_id": pessoa["servidor_id"],
                    "vinculo": ("dinter" if "dinter" in normalizar(p["slug"])
                                else "permanente"),
                    "pagina_id": p["id"],
                    "url": p["link"],
                })
                contagens["docente"] += 1
        elif tipo == "linhas":
            for linha in parsear_linhas_pesquisa(p, html, slug_para_id):
                ds.add("ppgc_linha_pesquisa", linha)
                contagens["linha"] += 1
        elif tipo == "disciplinas":
            for disc in parsear_disciplinas(p, html):
                ds.add("ppgc_disciplina", disc)
                contagens["disciplina"] += 1
        elif tipo == "defesas":
            for defesa in parsear_defesas(p, html):
                ds.add("ppgc_defesa", defesa)
                contagens["defesa"] += 1
    log.info("      %s", " · ".join(f"{k} {v}" for k, v in contagens.items()))

    log.info("[5/6] Posts do Programa")
    posts = cliente.fetch_all("posts", status="publish")
    posts_ppgc = [p for p in posts if post_e_ppgc(p, slug_por_categoria)]
    log.info("      %d de %d posts são do PPGC", len(posts_ppgc), len(posts))

    for p in posts_ppgc:
        html = p["content"]["rendered"]
        texto = html_para_texto(html)
        publicado = parse_data_wp(p.get("date"))
        modificado = parse_data_wp(p.get("modified"))
        data_pub = publicado.date() if publicado else None
        titulo = titulo_wp(p["title"]["rendered"])

        registrar_links_e_documentos(ds, "post", p["id"], titulo, p["link"],
                                     html, media_por_url,
                                     tabela_link="ppgc_link",
                                     tabela_doc="ppgc_documento")

        ds.add("ppgc_post", {
            "post_id": p["id"],
            "slug": p["slug"],
            "titulo": titulo,
            "url": p["link"],
            "data_publicacao": iso(data_pub),
            "publicado_em": iso(publicado),
            "ano": data_pub.year if data_pub else None,
            "mes": data_pub.month if data_pub else None,
            "semestre": semestre_de(data_pub),
            "data_por_extenso": data_por_extenso(data_pub),
            "data_modificacao": iso(modificado.date()) if modificado else None,
            "autor_nome": nome_por_autor.get(p.get("author")),
            "assunto": assunto_do_post(titulo, texto),
            "resumo": (limpar_texto(html_para_texto(p["excerpt"]["rendered"]))
                       or resumo_de(texto)),
            "texto": texto or None,
            "n_palavras": contar_palavras(texto),
            "anos_citados": anos_citados(titulo, texto) or None,
            "url_curta": p.get("jetpack_shortlink") or None,
        })
        for tid in p.get("tags") or []:
            if slug_por_tag.get(tid):
                ds.add("ppgc_post_tag", {"post_id": p["id"],
                                         "tag_slug": slug_por_tag[tid]})

    log.info("[6/6] Calendário público do PPGC")
    if coletar_calendario and pagina_calendario is not None:
        _coletar_calendario(cliente, ds, pagina_calendario)
    elif coletar_calendario:
        log.warning("      página de calendário com iframe não encontrada")

    ds.add("ppgc_crawl_meta", {"chave": "fonte", "valor": SITE_URL + "/ppgc/"})
    ds.add("ppgc_crawl_meta", {"chave": "api", "valor": API_URL})
    ds.add("ppgc_crawl_meta", {"chave": "coletado_em",
                               "valor": datetime.now().astimezone().isoformat()})
    ds.add("ppgc_crawl_meta", {"chave": "requisicoes",
                               "valor": str(cliente.n_requisicoes)})
    ds.add("ppgc_crawl_meta", {"chave": "paginas_ppgc", "valor": str(len(paginas_ppgc))})
    ds.add("ppgc_crawl_meta", {"chave": "posts_ppgc", "valor": str(len(posts_ppgc))})
    return ds


#: Assunto do post — filtro grosso que o roteador usa antes do vetorial.
_ASSUNTOS: tuple[tuple[str, str], ...] = (
    ("defesa",     r"defesa|disserta|\btese\b|qualifica"),
    ("edital",     r"edital|selecao|inscri|processo seletivo|homologa|classifica"),
    ("bolsa",      r"bolsa|\bcapes\b|\bcnpq\b|\bfapergs\b"),
    ("matricula",  r"matricula|rematricula|oferta de disciplina|horario"),
    ("avaliacao",  r"quadrienal|avaliacao capes|nota \d|conceito \d"),
    ("evento",     r"seminario|palestra|workshop|congresso|semana|minicurso"),
)


def assunto_do_post(titulo: str, texto: str) -> str:
    return _primeiro(_ASSUNTOS, normalizar(f"{titulo} {texto[:600]}"), "geral")


def _registrar_edital(ds: Dataset, pagina: dict, titulo: str, texto: str,
                      secoes: list[dict], caminho_pai: Optional[str],
                      publicado, modificado) -> int:
    """Cria a linha de `ppgc_edital` e os PDFs de `ppgc_edital_documento`."""
    edital_id = f"pagina-{pagina['id']}"
    tipo_edital = tipo_do_edital(titulo, pagina["slug"])
    # o texto é o último recurso para o ano: "Resultado: Homologação" não tem
    # ano no slug nem no título, mas cita o edital de 2019 no primeiro parágrafo
    ano, semestre = ano_semestre_de(pagina["slug"], titulo, texto[:400])
    m_num = _RE_NUMERO_EDITAL.search(f"{titulo} {texto[:2000]}")

    ds.add("ppgc_edital", {
        "edital_id": edital_id,
        "origem_tipo": "pagina",
        "origem_id": pagina["id"],
        "titulo": titulo,
        "slug": pagina["slug"],
        "url": pagina["link"],
        "tipo": tipo_edital,
        "nivel": nivel_do_edital(titulo, pagina["slug"], texto, tipo_edital),
        "ano": ano,
        "semestre": semestre,
        "periodo_letivo": f"{ano}/{semestre}" if ano and semestre else (
            str(ano) if ano else None),
        "numero_oficial": (re.sub(r"\s+", "", m_num.group(1)) if m_num else None),
        "indice_pai": caminho_pai.rsplit("/", 1)[-1] if caminho_pai else None,
        "data_publicacao": iso(publicado.date()) if publicado else None,
        "data_modificacao": iso(modificado.date()) if modificado else None,
        "resumo": resumo_de(texto),
        "texto": texto or None,
    })

    n = 0
    for secao in secoes:
        for link in extrair_links(secao.get("html") or "", pagina["link"]):
            if link["tipo"] not in ("documento", "planilha"):
                continue
            m_doc = _RE_NUMERO_EDITAL.search(link["texto"] or "")
            ds.add("ppgc_edital_documento", {
                "edital_id": edital_id,
                "url": link["url"],
                "ordem": n,
                "titulo": link["texto"] or nome_arquivo_de(link["url"]),
                "secao": secao["titulo"],
                "tipo_documento": tipo_do_documento(secao["titulo"], link["texto"]),
                "numero_oficial": (re.sub(r"\s+", "", m_doc.group(1))
                                   if m_doc else None),
            })
            n += 1
    return n


def _registrar_normativo_pagina(ds: Dataset, pagina: dict, titulo: str,
                                texto: str) -> int:
    """Página de resolução/portaria → normativo COM a íntegra do texto."""
    m = _RE_NORMATIVO_SLUG.match(normalizar(pagina["slug"]))
    if not m:
        return 0
    tipo, numero, ano = m.group(1), m.group(2), int(m.group(3))
    ds.add("ppgc_normativo", {
        "normativo_id": _id_normativo(tipo, numero, ano),
        "tipo": tipo,
        "numero": f"{int(numero):02d}",
        "ano": ano,
        "titulo": titulo,
        "ementa": resumo_de(texto, 300),
        "vigente": None,               # decidido pelo índice, não pela página
        "url_pagina": pagina["link"],
        "url_pdf": None,
        "pagina_id": pagina["id"],
        "texto": texto or None,
    }, merge=True)
    return 1


def _registrar_normativos_indice(ds: Dataset, pagina: dict, secoes: list[dict],
                                 media_por_url: dict[str, dict]) -> int:
    """
    Índice "Regimento e Resoluções" → um normativo por link, com o PDF.

    A seção diz o tipo E a vigência ("Resoluções em vigor" × "Resoluções
    revogadas"). Sem isso o dataset responderia com uma resolução revogada
    achando que está certa, que é o pior erro possível numa pergunta sobre
    regra de bolsa ou prazo de defesa.
    """
    n = 0
    for secao in secoes:
        tipo, vigente = tipo_normativo_da_secao(secao["titulo"])
        for link in extrair_links(secao.get("html") or "", pagina["link"]):
            if link["tipo"] not in ("documento", "planilha"):
                continue
            rotulo = link["texto"] or ""
            m = _RE_NORMATIVO_TEXTO.match(rotulo)
            if m:
                numero, ano, ementa = m.group(2), int(m.group(3)), m.group(4)
                explicito = normalizar(m.group(1) or "")
                tipo_final = ("resolucao" if explicito.startswith("resolu")
                              else "portaria" if explicito.startswith("portaria")
                              else tipo)
            else:
                m2 = _RE_NORMATIVO_ANO.match(rotulo)
                if not m2:
                    continue
                numero, ano, ementa = None, int(m2.group(2)), m2.group(1).strip()
                tipo_final = tipo
            media = media_por_url.get(link["url"]) or {}
            ds.add("ppgc_normativo", {
                "normativo_id": _id_normativo(tipo_final, numero, ano),
                "tipo": tipo_final,
                "numero": f"{int(numero):02d}" if numero else None,
                "ano": ano,
                "titulo": rotulo or None,
                "ementa": limpar_texto(ementa) or None,
                "vigente": vigente,
                "url_pagina": pagina["link"],
                "url_pdf": link["url"],
                "pagina_id": None,
                "media_id": media.get("id"),
                "texto": None,
            }, merge=True)
            n += 1
    return n


def _coletar_calendario(cliente: WPClient, ds: Dataset, pagina: dict) -> None:
    calendario = descobrir_calendario(pagina["content"]["rendered"])
    if not calendario:
        log.warning("      id do Google Calendar não encontrado no iframe")
        return
    from urllib.parse import quote
    url = (f"https://calendar.google.com/calendar/ical/"
           f"{quote(calendario, safe='')}/public/basic.ics")
    log.info("      calendário %s", calendario)
    try:
        ics = cliente.get_texto(url)
    except Exception as exc:                                    # noqa: BLE001
        log.warning("      falha ao baixar o .ics (%s) — seguindo sem eventos",
                    str(exc)[:100])
        return
    eventos = parsear_ics(ics, url)
    for evento in eventos:
        evento["pagina_id"] = pagina["id"]
        ds.add("ppgc_calendario_evento", evento)
    log.info("      %d eventos de calendário", len(eventos))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Crawler do PPGC (pós-graduação) — dataset separado do portal")
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--sem-calendario", action="store_true",
                    help="não baixa o .ics do Google Calendar")
    args = ap.parse_args()

    cliente = WPClient(delay=args.delay,
                       cache_dir=Path(args.cache_dir) if args.cache_dir else None)
    ds = montar(cliente, coletar_calendario=not args.sem_calendario)

    destino = Path(args.output)
    ds.salvar(destino)
    log.info("Gravado %s (%.1f MB, %d requisições)", destino,
             destino.stat().st_size / 1e6, cliente.n_requisicoes)
    for tabela, n in ds.contagens().items():
        if n:
            log.info("  %-26s %6d", tabela, n)


if __name__ == "__main__":
    main()
