"""
Ponto de entrada do Minicurso RAG
=============================================================================
Permite executar cada etapa do curso de forma isolada ou em sequência.

Uso:
  python main.py                        # menu interativo
  python main.py --etapa providers      # testa embeddings e LLM
  python main.py --etapa chunking       # demonstra segmentação
  python main.py --etapa ingestao       # ingesta documentos de exemplo
  python main.py --etapa busca          # busca semântica + comparação de métricas
  python main.py --etapa busca "query"  # busca com pergunta personalizada
  python main.py --etapa rag            # pipeline RAG completo (uma pergunta)
  python main.py --etapa rag "query"    # RAG com pergunta personalizada
  python main.py --etapa chatbot        # chatbot interativo
  python main.py --etapa tudo           # ingestão + busca + chatbot

Flags adicionais:
  --reset   recria a coleção do zero antes da ingestão
"""
import argparse
import sys


ETAPAS = {
    # Fundamentos
    "providers": "Seção 1–2 : Embeddings e LLM",
    "chunking":  "Seção 3   : Segmentação de documentos",
    "ingestao":  "Seção 4   : Ingestão no banco vetorial",
    "busca":     "Seção 5   : Busca semântica e métricas",
    "rag":       "Seção 6–7 : Pipeline RAG e validação",
    "chatbot":   "Seção 8   : Chatbot interativo",
    # Avançados
    "hibrido":    "Avançado  : Busca híbrida BM25 + semântica",
    "reranker":   "Avançado  : Reranking com cross-encoder",
    "guardrails": "Avançado  : Guardrails e safeguards",
    "eval":       "Avançado  : Avaliação MRR + BERTScore + LLM Judge",
    # Agente local (LFM2.5 + SQL + pgvector)
    "agente":     "Agente    : Chat com agente LFM2.5 (SQL + busca vetorial + web)",
    "avaliar-agente": "Agente    : Avalia o agente em lote (avaliacao/perguntas_exemplo.jsonl)",
    # Atalhos
    "tudo":      "           Ingestão → Busca → Chatbot",
}


def _menu_interativo() -> str:
    print("\n" + "=" * 50)
    print("  Minicurso RAG — Selecione a etapa")
    print("=" * 50)
    opcoes = list(ETAPAS.items())
    for i, (key, desc) in enumerate(opcoes, 1):
        print(f"  [{i}] {key:<12} {desc}")
    print("  [0] Sair")
    print()
    while True:
        try:
            escolha = input("  Opção: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if escolha == "0":
            sys.exit(0)
        if escolha.isdigit() and 1 <= int(escolha) <= len(opcoes):
            return opcoes[int(escolha) - 1][0]
        print("  Opção inválida. Tente novamente.")


def _run(etapa: str, query: str, reset: bool):
    if etapa == "providers":
        import providers as p
        print(f"\n  NVIDIA disponível : {p.NVIDIA_AVAILABLE}")
        print(f"  Google disponível : {p.GOOGLE_AVAILABLE}\n")
        emb = p.get_embeddings()
        vetor = emb.embed_query("teste de embedding em português")
        print(f"  Embedding: {len(vetor)} dims | primeiros 5: "
              f"{[round(v, 4) for v in vetor[:5]]}")
        llm = p.get_llm()
        r = llm.invoke("Em uma frase: o que é um embedding?")
        print(f"  LLM: {r.content}")

    elif etapa == "chunking":
        from chunking import demo_chunking  # type: ignore[import]
        # chunking.py expõe o demo via __main__; reutilizamos o bloco aqui
        import chunking as c
        import importlib, runpy
        runpy.run_module("chunking", run_name="__main__", alter_sys=False)

    elif etapa == "ingestao":
        from store import demo_ingestao
        demo_ingestao(reset=reset)

    elif etapa == "busca":
        from search import demo_busca
        demo_busca(query or "prazo para entrega da dissertação de mestrado")

    elif etapa == "rag":
        from pipeline import demo_rag
        demo_rag(query or "Qual o prazo para entrega da dissertação de mestrado?")

    elif etapa == "chatbot":
        from chatbot import run_chatbot
        run_chatbot()

    elif etapa == "hibrido":
        from hybrid_search import demo_hibrido
        demo_hibrido(query or "prazo de matrícula")

    elif etapa == "reranker":
        from reranker import demo_reranker
        demo_reranker(query or "Qual o prazo para entregar a dissertação?")

    elif etapa == "guardrails":
        from guardrails import demo_guardrails
        demo_guardrails()

    elif etapa == "eval":
        from evaluation import demo_evaluation
        demo_evaluation()

    elif etapa == "agente":
        import sys as _sys
        from agente_rag import main as agente_main
        _sys.argv = ["agente_rag.py"] + (["-p", query] if query else [])
        agente_main()

    elif etapa == "avaliar-agente":
        import sys as _sys
        from pathlib import Path as _Path
        from avaliar_agente import main as avaliar_main
        perguntas = query or str(_Path(__file__).resolve().parent.parent / "avaliacao" / "perguntas_computacao.jsonl")
        _sys.argv = ["avaliar_agente.py", "--perguntas", perguntas]
        avaliar_main()

    elif etapa == "tudo":
        from store import demo_ingestao
        from search import demo_busca
        from chatbot import run_chatbot
        demo_ingestao(reset=reset)
        demo_busca(query or "prazo para entrega da dissertação de mestrado")
        run_chatbot()


def main():
    parser = argparse.ArgumentParser(
        description="Minicurso RAG — execute etapas individualmente",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:<12} {v}" for k, v in ETAPAS.items()),
    )
    parser.add_argument(
        "--etapa",
        choices=list(ETAPAS.keys()),
        default=None,
        metavar="ETAPA",
        help="Etapa a executar (omita para menu interativo)",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="Pergunta opcional para 'busca', 'rag' e 'agente'; arquivo JSONL para 'avaliar-agente'",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Recria a coleção do zero antes da ingestão",
    )
    args = parser.parse_args()

    etapa = args.etapa or _menu_interativo()
    _run(etapa, args.query, args.reset)


if __name__ == "__main__":
    main()
