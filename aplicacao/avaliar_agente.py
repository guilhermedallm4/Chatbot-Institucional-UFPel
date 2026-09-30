"""
Avaliação do agente RAG (perguntas → respostas → métricas)
=============================================================================
Conceito — Módulo Avançado (Avaliação) aplicado ao agente

Roda um conjunto de perguntas pelo agente (agente_rag.py), registra a
resposta, as ferramentas chamadas e o tempo, e calcula métricas:

  • contem        : a resposta contém os trechos obrigatórios? (deve_conter / deve_conter_algum)
  • rougeL_f1     : sobreposição léxica com resposta_esperada (evaluation.rouge_l_score)
  • cos_emb       : similaridade cosseno dos embeddings resposta × esperada (bertscore_lite)
  • ferramentas   : quais foram usadas e se batem com ferramentas_esperadas
  • judge_*       : LLM-as-a-judge (o próprio LFM2.5, sem ferramentas) — fidelidade,
                    relevância e completude 0–10, com o contexto recuperado como evidência

Formato do arquivo de perguntas (JSONL, uma pergunta por linha):
  {"id": "q1", "pergunta": "Quem coordena o curso de Ciência da Computação?",
   "resposta_esperada": "O coordenador é Guilherme Tomaschewski Netto.",
   "deve_conter": ["Tomaschewski"],                 # todos precisam aparecer
   "deve_conter_algum": ["Palomino", "Agostini"],    # basta um
   "ferramentas_esperadas": ["buscar_por_nome", "consultar_sql"]}   # basta uma

Uso:
  python avaliar_agente.py --perguntas ../avaliacao/perguntas_exemplo.jsonl
  python avaliar_agente.py --perguntas ../avaliacao/perguntas_exemplo.jsonl --judge
  python avaliar_agente.py -q "Quantos projetos estão ativos?" -q "Quem coordena o PPGC?"
  python avaliar_agente.py --perguntas x.jsonl --saida ../avaliacao/resultados/run1
Saída: <saida>/resultados.jsonl (detalhado) e <saida>/relatorio.md (resumo).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional

import config  # noqa: F401  (carrega .env)
from agente_lfm import AgentTurn
from agente_rag import criar_agente


# =============================================================================
# Métricas
# =============================================================================

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s.lower()).strip()


def checa_conteudo(resposta: str, deve_conter: list[str], deve_conter_algum: list[str]) -> Optional[bool]:
    if not deve_conter and not deve_conter_algum:
        return None
    r = _norm(resposta)
    ok = all(_norm(t) in r for t in deve_conter)
    if deve_conter_algum:
        ok = ok and any(_norm(t) in r for t in deve_conter_algum)
    return ok


def rouge_l(resposta: str, esperada: str) -> Optional[float]:
    if not esperada:
        return None
    from evaluation import rouge_l_score
    r = rouge_l_score(resposta, esperada)
    return r.get("f1")


_emb = None


def cos_emb(resposta: str, esperada: str) -> Optional[float]:
    global _emb
    if not esperada:
        return None
    import numpy as np
    if _emb is None:
        from providers import get_embeddings
        _emb = get_embeddings()
    a, b = np.array(_emb.embed_query(resposta)), np.array(_emb.embed_query(esperada))
    return round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)), 4)


JUDGE_SYSTEM = "Você é um avaliador rigoroso de respostas de um chatbot institucional. Responda apenas com JSON."
JUDGE_PROMPT = """\
Avalie a resposta abaixo em três dimensões, com notas inteiras de 0 a 10:
- fidelidade : a resposta usa apenas informações presentes no CONTEXTO (sem inventar)?
- relevancia : a resposta responde de fato à PERGUNTA?
- completude : a resposta cobre o que foi perguntado?
{referencia}
PERGUNTA: {pergunta}

CONTEXTO (resultados das ferramentas):
{contexto}

RESPOSTA GERADA:
{resposta}

Retorne EXCLUSIVAMENTE um JSON no formato:
{{"fidelidade": <0-10>, "relevancia": <0-10>, "completude": <0-10>, "justificativa": "<uma frase>"}}"""


def llm_judge(agente, turno: AgentTurn, esperada: str = "", pensar: bool = False) -> dict:
    """
    LLM-as-a-judge com o próprio modelo. Por padrão SEM raciocínio (<think>), para
    que o orçamento de tokens vá todo para o JSON; --judge-pensar liga o raciocínio
    (mais lento, potencialmente mais criterioso).
    """
    referencia = f"\nRESPOSTA DE REFERÊNCIA (gabarito): {esperada}\n" if esperada else ""
    contexto = turno.contexto_recuperado()[:6000] or "(nenhuma ferramenta foi chamada)"
    saida = agente.gerar_simples(
        JUDGE_PROMPT.format(referencia=referencia, pergunta=turno.pergunta, contexto=contexto,
                            resposta=turno.resposta[:3000]),
        system=JUDGE_SYSTEM, max_new_tokens=1200 if pensar else 400, pensar=pensar,
    )
    m = re.search(r"\{[^{}]*\}", saida, re.S)

    def _nota(v):
        try:
            return max(0, min(10, int(float(v))))
        except (TypeError, ValueError):
            return None

    if not m:
        # JSON truncado/malformado: recupera as notas por regex
        notas = {k: re.search(rf'"{k}"\s*:\s*(\d+)', saida) for k in ("fidelidade", "relevancia", "completude")}
        just = re.search(r'"justificativa"\s*:\s*"([^"]*)', saida)
        if any(notas.values()):
            return {k: (_nota(v.group(1)) if v else None) for k, v in notas.items()} | {
                "justificativa": just.group(1) if just else None, "_bruto": saida[:300]}
    try:
        d = json.loads(m.group(0)) if m else {}
        return {"fidelidade": _nota(d.get("fidelidade")), "relevancia": _nota(d.get("relevancia")),
                "completude": _nota(d.get("completude")), "justificativa": d.get("justificativa"),
                "_bruto": None if m else saida[:300]}
    except json.JSONDecodeError:
        return {"fidelidade": None, "relevancia": None, "completude": None, "justificativa": None,
                "_bruto": saida[:300]}


# =============================================================================
# Execução
# =============================================================================

def carregar_perguntas(caminho: Optional[str], inline: list[str]) -> list[dict]:
    itens: list[dict] = []
    if caminho:
        p = Path(caminho)
        if p.suffix == ".json":
            itens += json.loads(p.read_text(encoding="utf-8"))
        else:
            for linha in p.read_text(encoding="utf-8").splitlines():
                linha = linha.strip()
                if linha and not linha.startswith("#"):
                    itens.append(json.loads(linha))
    for q in inline:
        itens.append({"pergunta": q})
    for i, it in enumerate(itens, 1):
        it.setdefault("id", f"q{i}")
    return itens


def avaliar(itens: list[dict], agente, judge: bool = False, usar_emb: bool = True,
            judge_pensar: bool = False) -> list[dict]:
    resultados = []
    for i, it in enumerate(itens, 1):
        print(f"\n[{i}/{len(itens)}] {it['id']}: {it['pergunta']}", file=sys.stderr, flush=True)
        agente.limpar()
        turno = agente.perguntar(it["pergunta"])
        esperada = it.get("resposta_esperada", "")
        ferramentas = [c.name for c in turno.chamadas]
        fe = it.get("ferramentas_esperadas") or []
        r = {
            **turno.to_dict(),
            "id": it["id"],
            "resposta_esperada": esperada,
            "ferramentas_usadas": ferramentas,
            "ferramenta_ok": (any(f in ferramentas for f in fe) if fe else None),
            "contem_ok": checa_conteudo(turno.resposta, it.get("deve_conter") or [], it.get("deve_conter_algum") or []),
            "rougeL_f1": rouge_l(turno.resposta, esperada),
            "cos_emb": cos_emb(turno.resposta, esperada) if usar_emb else None,
        }
        if judge:
            r["judge"] = llm_judge(agente, turno, esperada, pensar=judge_pensar)
        resultados.append(r)
        print(f"    → {turno.tempo_total:.1f}s | rodadas={turno.rodadas} | ferramentas={ferramentas} "
              f"| contem_ok={r['contem_ok']} | rougeL={r['rougeL_f1']} | cos={r['cos_emb']}"
              + (f" | judge={r['judge']}" if judge else ""), file=sys.stderr, flush=True)
        print(f"    Resposta: {turno.resposta[:300].replace(chr(10), ' ')}{'...' if len(turno.resposta) > 300 else ''}",
              file=sys.stderr, flush=True)
    return resultados


def _media(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 3) if vals else None


def relatorio_md(res: list[dict], judge: bool, meta: dict) -> str:
    n = len(res)
    contem = [r["contem_ok"] for r in res if r["contem_ok"] is not None]
    ferr = [r["ferramenta_ok"] for r in res if r["ferramenta_ok"] is not None]
    linhas = [
        f"# Avaliação do agente RAG — {meta['data']}",
        "",
        f"- Modelo: `{meta['modelo']}` | backend: {meta.get('backend','local')} | device: {meta.get('device','?')} "
        f"({meta.get('dtype','?')}) | web: {meta['web']} | raciocínio: {meta['pensar']}",
        f"- Perguntas: {n} | Tempo médio: {_media([r['tempo_total_s'] for r in res])}s | "
        f"Rodadas médias: {_media([r['rodadas'] for r in res])}",
        f"- Acerto de conteúdo (deve_conter): {sum(contem)}/{len(contem)}" if contem else "- Acerto de conteúdo: n/a",
        f"- Ferramenta esperada usada: {sum(ferr)}/{len(ferr)}" if ferr else "- Ferramenta esperada: n/a",
        f"- ROUGE-L F1 médio: {_media([r['rougeL_f1'] for r in res])} | Cosseno emb. médio: {_media([r['cos_emb'] for r in res])}",
        f"- Limite de rodadas atingido: {sum(1 for r in res if r['limite_atingido'])}/{n}",
    ]
    if judge:
        for k in ("fidelidade", "relevancia", "completude"):
            linhas.append(f"- Judge {k} médio: {_media([r['judge'].get(k) for r in res])}/10")
    linhas += ["", "| id | pergunta | ferramentas | contém | ROUGE-L | cos | " + ("fid | rel | comp | " if judge else "") + "s |",
               "|---|---|---|---|---|---|" + ("---|---|---|" if judge else "") + "---|"]
    for r in res:
        cel = lambda v: "-" if v is None else (("✅" if v else "❌") if isinstance(v, bool) else v)  # noqa: E731
        j = r.get("judge", {})
        linhas.append(
            f"| {r['id']} | {r['pergunta'][:70]} | {', '.join(dict.fromkeys(r['ferramentas_usadas'])) or '-'} | "
            f"{cel(r['contem_ok'])} | {cel(r['rougeL_f1'])} | {cel(r['cos_emb'])} | "
            + (f"{cel(j.get('fidelidade'))} | {cel(j.get('relevancia'))} | {cel(j.get('completude'))} | " if judge else "")
            + f"{r['tempo_total_s']} |"
        )
    linhas += ["", "## Respostas", ""]
    for r in res:
        linhas += [f"### {r['id']} — {r['pergunta']}", "",
                   f"**Ferramentas:** " + (" → ".join(f"`{c['ferramenta']}({json.dumps(c['argumentos'], ensure_ascii=False)})`"
                                                    for c in r["chamadas"]) or "nenhuma"), "",
                   r["resposta"] or "_(vazia)_", ""]
        if r["resposta_esperada"]:
            linhas += [f"> **Esperada:** {r['resposta_esperada']}", ""]
        if judge and r.get("judge", {}).get("justificativa"):
            linhas += [f"> **Judge:** {r['judge']['justificativa']}", ""]
    return "\n".join(linhas)


def main():
    ap = argparse.ArgumentParser(description="Avalia o agente RAG em lote")
    ap.add_argument("--perguntas", help="Arquivo JSONL/JSON com as perguntas")
    ap.add_argument("-q", "--pergunta", action="append", default=[], help="Pergunta inline (pode repetir)")
    ap.add_argument("--saida", help="Diretório de saída (padrão: ../avaliacao/resultados/<timestamp>)")
    ap.add_argument("--judge", action="store_true", help="Ativa LLM-as-a-judge com o próprio modelo")
    ap.add_argument("--judge-pensar", action="store_true", help="Judge com raciocínio <think> (mais lento)")
    ap.add_argument("--sem-emb", action="store_true", help="Não calcula similaridade por embeddings")
    ap.add_argument("--web", action="store_true", help="Habilita buscar_web no agente")
    ap.add_argument("--pensar", action="store_true", help="Liga o raciocínio <think> do modelo (padrão: desligado)")
    ap.add_argument("--sem-pensar", action="store_true", help=argparse.SUPPRESS)  # compatibilidade
    ap.add_argument("--max-rodadas", type=int, default=6)
    ap.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"],
                    help="Onde rodar o modelo local (padrão: auto — GPU se houver)")
    ap.add_argument("--backend", default=None, choices=["local", "openrouter"],
                    help="local = LFM2.5 nesta máquina | openrouter = mesmo modelo pela API gratuita")
    ap.add_argument("--mostrar-rastro", action="store_true", help="Mostra as chamadas de ferramenta em tempo real")
    args = ap.parse_args()

    import os
    if args.backend:
        os.environ["AGENT_BACKEND"] = args.backend
    if args.device:
        os.environ["AGENT_DEVICE"] = args.device
        if args.device == "cpu":
            os.environ.setdefault("EMBEDDING_DEVICE", "cpu")

    itens = carregar_perguntas(args.perguntas, args.pergunta)
    if not itens:
        ap.error("informe --perguntas ARQUIVO ou -q 'pergunta'")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saida = Path(args.saida or Path(__file__).resolve().parent.parent / "avaliacao" / "resultados" / stamp)
    saida.mkdir(parents=True, exist_ok=True)

    agente = criar_agente(web=args.web, pensar=args.pensar, stream=False, max_rodadas=args.max_rodadas,
                          verbose_tools=args.mostrar_rastro)
    t0 = time.time()
    res = avaliar(itens, agente, judge=args.judge, usar_emb=not args.sem_emb, judge_pensar=args.judge_pensar)

    meta = {"data": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "modelo": agente.model.config._name_or_path if agente.model else agente.model_id,
            "web": args.web, "pensar": args.pensar,
            "backend": getattr(agente, "backend", "local"),
            "device": getattr(agente, "device", "?"),
            "dtype": (str(next(agente.model.parameters()).dtype).replace("torch.", "") if agente.model else "api")}
    (saida / "resultados.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in res) + "\n", encoding="utf-8")
    md = relatorio_md(res, args.judge, meta)
    (saida / "relatorio.md").write_text(md, encoding="utf-8")

    print("\n" + "\n".join(md.split("\n## Respostas")[0].splitlines()))
    print(f"\nTempo total: {time.time() - t0:.0f}s | Arquivos em: {saida}")


if __name__ == "__main__":
    main()
