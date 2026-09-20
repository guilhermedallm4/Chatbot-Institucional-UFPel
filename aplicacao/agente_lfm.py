"""
Núcleo do agente local — LiquidAI LFM2.5
=============================================================================
Conceito — Módulo Agente

Um *agente* é um LLM que, em vez de responder de imediato, pode decidir
chamar ferramentas (funções Python), ler o resultado e iterar até ter o que
precisa para responder. Este módulo implementa o laço genérico:

    prompt → geração → há chamada de ferramenta?
        sim → executa → devolve resultado ao modelo → gera de novo
        não → resposta final ao usuário

O LFM2.5 tem suporte nativo a tool calling. O chat template recebe os
schemas das ferramentas (formato OpenAI) e o modelo emite chamadas em
sintaxe Python entre tokens especiais:

    <|tool_call_start|>[buscar_semantica(consulta='...', top_k=5)]<|tool_call_end|>

O resultado da ferramenta volta em uma mensagem com role="tool".
O modelo também raciocina antes de agir: o template sempre abre um bloco
<think> ... </think>, que é ocultado do usuário.

As ferramentas concretas (busca vetorial, SQL, web) ficam em outros
módulos — aqui só está o mecanismo. Veja agente_rag.py para o uso.
"""
from __future__ import annotations

import ast
import inspect
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

MODEL_ID_DEFAULT = "LiquidAI/LFM2.5-2.6B"

# Tokens especiais do LFM2.5 (verificados no tokenizer)
TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"
THINK_START = "<think>"
THINK_END = "</think>"
IM_END = "<|im_end|>"

_TOOL_CALL_RE = re.compile(re.escape(TOOL_CALL_START) + r"(.*?)" + re.escape(TOOL_CALL_END), re.S)

CINZA, CIANO, NEGRITO, RESET = "\033[90m", "\033[36m", "\033[1m", "\033[0m"


# =============================================================================
# Registro de ferramentas
# =============================================================================

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


@dataclass
class Tool:
    """Uma ferramenta = função Python + schema (formato OpenAI) para o modelo."""

    name: str
    fn: Callable[..., Any]
    description: str
    parameters: dict  # JSON Schema dos argumentos

    @classmethod
    def from_function(cls, fn: Callable, description: Optional[str] = None,
                      param_docs: Optional[dict[str, str]] = None) -> "Tool":
        """Constrói o schema a partir da assinatura e anotações de tipo da função."""
        sig = inspect.signature(fn)
        props, required = {}, []
        for pname, p in sig.parameters.items():
            ann = p.annotation if p.annotation is not inspect.Parameter.empty else str
            # Optional[X] / X | None → X
            args = getattr(ann, "__args__", None)
            if args:
                ann = next((a for a in args if a is not type(None)), str)
            prop = {"type": _JSON_TYPES.get(ann, "string")}
            if param_docs and pname in param_docs:
                prop["description"] = param_docs[pname]
            if p.default is inspect.Parameter.empty:
                required.append(pname)
            else:
                prop["description"] = (prop.get("description", "") + f" (padrão: {p.default!r})").strip()
            props[pname] = prop
        return cls(
            name=fn.__name__,
            fn=fn,
            description=description or (inspect.getdoc(fn) or "").split("\n\n")[0],
            parameters={"type": "object", "properties": props, "required": required},
        )

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


@dataclass
class ToolCall:
    name: str
    arguments: dict
    result: str = ""
    elapsed: float = 0.0


@dataclass
class AgentTurn:
    """Registro completo de uma pergunta: resposta, chamadas e tempos (útil para avaliação)."""

    pergunta: str
    resposta: str = ""
    chamadas: list[ToolCall] = field(default_factory=list)
    pensamentos: list[str] = field(default_factory=list)
    rodadas: int = 0
    tempo_total: float = 0.0
    limite_atingido: bool = False

    def contexto_recuperado(self) -> str:
        """Concatena os resultados das ferramentas (o 'contexto' para o LLM-judge)."""
        return "\n\n".join(f"[{c.name}({c.arguments})]\n{c.result}" for c in self.chamadas)

    def to_dict(self) -> dict:
        return {
            "pergunta": self.pergunta,
            "resposta": self.resposta,
            "rodadas": self.rodadas,
            "tempo_total_s": round(self.tempo_total, 2),
            "limite_atingido": self.limite_atingido,
            "chamadas": [
                {"ferramenta": c.name, "argumentos": c.arguments,
                 "tempo_s": round(c.elapsed, 3), "resultado": c.result[:2000]}
                for c in self.chamadas
            ],
            "pensamentos": self.pensamentos,
        }


# =============================================================================
# Parsing das chamadas emitidas pelo modelo
# =============================================================================

def parse_tool_calls(texto: str, tools: dict[str, Tool]) -> list[ToolCall]:
    """
    Extrai chamadas no formato do LFM2.5:
        <|tool_call_start|>[f(a='x', b=3), g()]<|tool_call_end|>
    Argumentos posicionais são mapeados pela ordem dos parâmetros do schema.
    """
    chamadas: list[ToolCall] = []
    for bloco in _TOOL_CALL_RE.findall(texto):
        bloco = bloco.strip()
        try:
            arvore = ast.parse(bloco, mode="eval").body
        except SyntaxError:
            continue
        nos = arvore.elts if isinstance(arvore, (ast.List, ast.Tuple)) else [arvore]
        for no in nos:
            if not isinstance(no, ast.Call) or not isinstance(no.func, ast.Name):
                continue
            nome = no.func.id
            tool = tools.get(nome)
            param_names = list(tool.parameters["properties"].keys()) if tool else []
            args: dict[str, Any] = {}
            for i, a in enumerate(no.args):
                if i < len(param_names):
                    try:
                        args[param_names[i]] = ast.literal_eval(a)
                    except ValueError:
                        args[param_names[i]] = ast.unparse(a)
            for kw in no.keywords:
                try:
                    args[kw.arg] = ast.literal_eval(kw.value)
                except ValueError:
                    args[kw.arg] = ast.unparse(kw.value)
            chamadas.append(ToolCall(name=nome, arguments=args))
    return chamadas


def _strip_tool_calls(texto: str) -> str:
    return _TOOL_CALL_RE.sub("", texto).strip()


_PLANO_RE = re.compile(
    r"^\s*(vou |preciso |primeiro[, ]|deixe-me |deixa eu |let me |i need to |i'll |i will |first,? )"
    r"|\b(vou (usar|consultar|descrever|buscar|verificar|executar|fazer uma busca)|farei uma busca|"
    r"preciso (verificar|consultar|buscar))\b",
    re.I,
)


def _parece_plano(texto: str) -> bool:
    """Resposta curta que só anuncia o que o modelo *vai* fazer, sem chamada de ferramenta."""
    t = texto.strip()
    return bool(t) and len(t) < 700 and bool(_PLANO_RE.search(t))


def _separar_pensamento(texto: str) -> tuple[str, str]:
    texto = texto.replace(IM_END, "")
    if THINK_END in texto:
        pens, resto = texto.split(THINK_END, 1)
        return pens.replace(THINK_START, "").strip(), resto.strip()
    return "", texto.strip()


# =============================================================================
# Streamer: mostra só a resposta final (oculta <think> e chamadas)
# =============================================================================

def _make_streamer_class():
    from transformers import TextStreamer

    class AgentStreamer(TextStreamer):
        def __init__(self, tokenizer, mostrar_pensamento: bool = False, silencioso: bool = False):
            super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=False)
            self.mostrar_pensamento = mostrar_pensamento
            self.silencioso = silencioso
            self.reset_estado()

        def reset_estado(self, estado: str = "pensando"):
            self.estado = estado  # pensando → respondendo | ferramenta
            self._pens_aberto = False

        def _out(self, s: str):
            if not self.silencioso:
                print(s, end="", flush=True)

        def on_finalized_text(self, text: str, stream_end: bool = False):
            text = text.replace(IM_END, "")
            if self.estado == "pensando":
                if THINK_END in text:
                    antes, depois = text.split(THINK_END, 1)
                    if self.mostrar_pensamento:
                        self._print_pens(antes)
                        self._out(RESET + "\n")
                    self.estado = "respondendo"
                    text = depois.lstrip("\n")
                else:
                    if self.mostrar_pensamento:
                        self._print_pens(text)
                    return
            if self.estado == "respondendo":
                if TOOL_CALL_START in text:
                    self._out(text.split(TOOL_CALL_START, 1)[0])
                    self.estado = "ferramenta"
                    return
                self._out(text)
                if stream_end:
                    self._out("\n")

        def _print_pens(self, text: str):
            if not self._pens_aberto:
                self._out(f"{CINZA}[pensando] ")
                self._pens_aberto = True
            self._out(text)

    return AgentStreamer


# =============================================================================
# Agente
# =============================================================================

class AgenteLFM:
    """
    Laço agêntico genérico sobre o LFM2.5.

    Uso:
        agente = AgenteLFM(system_prompt="...", tools=[Tool.from_function(f), ...])
        turno = agente.perguntar("pergunta")      # AgentTurn com resposta e rastro
        agente.limpar()                            # nova conversa
    """

    def __init__(
        self,
        system_prompt: str,
        tools: list[Tool],
        model_id: str = MODEL_ID_DEFAULT,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        pensar: bool = True,
        mostrar_pensamento: bool = False,
        max_tool_rounds: int = 6,
        stream: bool = True,
        verbose_tools: bool = True,
        exigir_ferramenta: bool = False,
        max_calls_per_round: int = 4,
        max_tool_result_chars: int = 5000,
        max_context_tokens: int = 20000,
    ):
        """
        exigir_ferramenta      : se o modelo responder à 1ª rodada sem chamar ferramenta, recebe um
                                 lembrete e gera de novo (evita resposta "de cabeça" em modelos pequenos).
        max_calls_per_round    : limite de chamadas executadas por rodada (o resto é adiado).
        max_tool_result_chars  : corte de cada resultado de ferramenta antes de entrar no histórico.
        max_context_tokens     : acima disso, resultados antigos de ferramentas são resumidos no histórico
                                 para não estourar a janela (32k) nem a memória da GPU.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.system_prompt = system_prompt
        self.tools = {t.name: t for t in tools}
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.pensar = pensar
        self.max_tool_rounds = max_tool_rounds
        self.verbose_tools = verbose_tools
        self.exigir_ferramenta = exigir_ferramenta
        self.max_calls_per_round = max(1, max_calls_per_round)
        self.max_tool_result_chars = max_tool_result_chars
        self.max_context_tokens = max_context_tokens

        print(f"[agente] Carregando {model_id} ...", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype=torch.bfloat16)
        self.model.eval()
        self.im_end_id = self.tokenizer.convert_tokens_to_ids(IM_END)
        self.streamer = _make_streamer_class()(self.tokenizer, mostrar_pensamento, silencioso=not stream)
        self.historico: list[dict] = []
        self.limpar()

    # ------------------------------------------------------------------
    def limpar(self):
        self.historico = [{"role": "system", "content": self.system_prompt}]

    def set_stream(self, ativo: bool):
        self.streamer.silencioso = not ativo

    # ------------------------------------------------------------------
    def _gerar(self, historico: list[dict], tools: bool = True, max_new_tokens: Optional[int] = None) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            historico,
            tools=[t.schema() for t in self.tools.values()] if tools and self.tools else None,
            add_generation_prompt=True,
            tokenize=False,
        )
        if not self.pensar:
            prompt += THINK_END + "\n"  # fecha o <think> aberto pelo template → pula raciocínio

        input_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            self.model.device
        )
        self.streamer.reset_estado("respondendo" if not self.pensar else "pensando")

        amostrar = self.temperature > 0
        with torch.no_grad():
            saida = self.model.generate(
                input_ids,
                do_sample=amostrar,
                temperature=self.temperature if amostrar else None,
                top_k=50 if amostrar else None,
                repetition_penalty=1.1,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                eos_token_id=self.im_end_id,
                pad_token_id=self.tokenizer.pad_token_id or self.im_end_id,
                streamer=self.streamer,
            )
        return self.tokenizer.decode(saida[0, input_ids.shape[1]:], skip_special_tokens=False)

    def gerar_simples(self, prompt: str, system: Optional[str] = None, max_new_tokens: int = 512,
                      pensar: Optional[bool] = None) -> str:
        """
        Geração sem ferramentas nem histórico (usado pelo LLM-judge da avaliação).
        pensar=False pula o bloco <think>: útil quando se quer só um JSON curto e
        o raciocínio consumiria o orçamento de tokens.
        """
        hist = [{"role": "system", "content": system}] if system else []
        hist.append({"role": "user", "content": prompt})
        silencioso_antes, pensar_antes = self.streamer.silencioso, self.pensar
        self.streamer.silencioso = True
        if pensar is not None:
            self.pensar = pensar
        try:
            bruto = self._gerar(hist, tools=False, max_new_tokens=max_new_tokens)
        finally:
            self.streamer.silencioso, self.pensar = silencioso_antes, pensar_antes
        _, conteudo = _separar_pensamento(bruto)
        return _strip_tool_calls(conteudo)

    # ------------------------------------------------------------------
    # Hook opcional: recebe uma chamada a ferramenta inexistente e devolve um resultado
    # (ou None para cair no erro padrão). Usado para tratar "view(arg=...)" como SQL.
    on_unknown_tool: Optional[Callable[[ToolCall], Optional[str]]] = None

    def _executar(self, chamada: ToolCall) -> str:
        tool = self.tools.get(chamada.name)
        t0 = time.time()
        if tool is None:
            resultado = None
            if self.on_unknown_tool is not None:
                try:
                    resultado = self.on_unknown_tool(chamada)
                except Exception as e:  # noqa: BLE001
                    resultado = f"Erro ao tratar chamada desconhecida {chamada.name}: {e}"
            if resultado is None:
                resultado = f"Erro: ferramenta desconhecida '{chamada.name}'. Disponíveis: {', '.join(self.tools)}."
            chamada.elapsed = time.time() - t0
            chamada.result = resultado
            return resultado
        try:
            resultado = str(tool.fn(**chamada.arguments))
        except TypeError as e:
            resultado = f"Erro nos argumentos de {chamada.name}: {e}"
        except Exception as e:  # noqa: BLE001
            resultado = f"Erro ao executar {chamada.name}: {type(e).__name__}: {e}"
        chamada.elapsed = time.time() - t0
        if len(resultado) > self.max_tool_result_chars:
            resultado = resultado[: self.max_tool_result_chars] + "\n[... resultado cortado; refine a consulta se precisar do resto]"
        chamada.result = resultado
        return resultado

    def _n_tokens(self, historico: list[dict]) -> int:
        ids = self.tokenizer.apply_chat_template(
            historico, tools=[t.schema() for t in self.tools.values()] or None,
            add_generation_prompt=True, tokenize=True,
        )
        return len(ids["input_ids"] if isinstance(ids, dict) else ids)

    def _compactar_historico(self):
        """Resume os resultados de ferramentas mais antigos quando o prompt passa de max_context_tokens."""
        if self._n_tokens(self.historico) <= self.max_context_tokens:
            return
        idx_tools = [i for i, m in enumerate(self.historico) if m["role"] == "tool"]
        for i in idx_tools[:-1]:  # preserva o resultado mais recente
            m = self.historico[i]
            if len(m["content"]) > 700:
                m["content"] = m["content"][:600] + "\n[... resultado antigo resumido para caber no contexto]"
                if self._n_tokens(self.historico) <= self.max_context_tokens:
                    return

    def perguntar(self, pergunta: str) -> AgentTurn:
        turno = AgentTurn(pergunta=pergunta)
        t_ini = time.time()
        self.historico.append({"role": "user", "content": pergunta})
        nudge_idx: Optional[int] = None  # posição do lembrete "use uma ferramenta", para limpar depois
        plano_nudges = 0

        for rodada in range(self.max_tool_rounds + 1):
            turno.rodadas = rodada + 1
            self._compactar_historico()
            bruto = self._gerar(self.historico)
            pensamento, conteudo = _separar_pensamento(bruto)
            if pensamento:
                turno.pensamentos.append(pensamento)
            chamadas = parse_tool_calls(conteudo, self.tools)

            # ---- anunciou o plano ("Vou consultar...") mas não emitiu a chamada: pede a chamada ----
            if not chamadas and self.tools and plano_nudges < 2 and _parece_plano(_strip_tool_calls(conteudo)):
                plano_nudges += 1
                self.historico.append({"role": "assistant", "content": _strip_tool_calls(conteudo)})
                self.historico.append({"role": "user", "content": (
                    "[sistema] Você descreveu o que vai fazer, mas não chamou nenhuma ferramenta. "
                    "Emita agora a chamada de ferramenta correspondente (ou, se já tem a informação, escreva a resposta final).")})
                bruto = self._gerar(self.historico)
                pensamento, conteudo = _separar_pensamento(bruto)
                if pensamento:
                    turno.pensamentos.append(pensamento)
                del self.historico[-2:]
                chamadas = parse_tool_calls(conteudo, self.tools)

            # ---- respondeu sem consultar nada: lembra uma vez e gera de novo -----
            if not chamadas and self.exigir_ferramenta and not turno.chamadas and nudge_idx is None and self.tools:
                nudge_idx = len(self.historico)
                self.historico.append({"role": "assistant", "content": _strip_tool_calls(conteudo)})
                self.historico.append({
                    "role": "user",
                    "content": (
                        "[sistema] Você respondeu sem consultar a base. Antes de responder, chame pelo menos uma "
                        "ferramenta (buscar_por_nome, buscar_semantica ou consultar_sql) e baseie a resposta no "
                        "resultado. Se a pergunta for claramente fora do escopo institucional, diga apenas isso."
                    ),
                })
                continue

            # ---- resposta final --------------------------------------------
            if not chamadas or rodada == self.max_tool_rounds:
                resposta = _strip_tool_calls(conteudo)
                if not chamadas and not resposta:
                    # O modelo fechou o <think> e parou sem escrever nada (acontece em modelos
                    # pequenos quando a resposta "ficou" dentro do raciocínio). Pede o texto final.
                    self.historico.append({"role": "assistant", "content": ""})
                    self.historico.append({"role": "user", "content": (
                        "[sistema] Sua resposta ficou vazia. Se você pretendia consultar uma ferramenta, emita a chamada "
                        "agora; caso contrário, escreva a resposta final ao usuário em texto, com base no que já foi obtido.")})
                    bruto = self._gerar(self.historico)
                    pensamento, conteudo = _separar_pensamento(bruto)
                    if pensamento:
                        turno.pensamentos.append(pensamento)
                    del self.historico[-2:]
                    resposta = _strip_tool_calls(conteudo)
                    chamadas_retry = parse_tool_calls(conteudo, self.tools)
                    if chamadas_retry and rodada < self.max_tool_rounds:
                        chamadas = chamadas_retry  # decidiu buscar mais: segue para a rodada de ferramentas
                if chamadas and not resposta:
                    # Limite estourado e o modelo ainda quer chamar ferramenta:
                    # força uma resposta em texto com o que já foi coletado.
                    turno.limite_atingido = True
                    self.historico.append({
                        "role": "tool",
                        "content": (
                            "As ferramentas não estão mais disponíveis nesta conversa. Responda ao "
                            "usuário agora, em texto, com as informações já obtidas. Se não encontrou "
                            "a informação exata, diga isso claramente e indique o que encontrou de mais próximo."
                        ),
                    })
                    bruto = self._gerar(self.historico)
                    pensamento, conteudo = _separar_pensamento(bruto)
                    resposta = _strip_tool_calls(conteudo)
                msg: dict[str, Any] = {"role": "assistant", "content": resposta}
                if pensamento:
                    msg["thinking"] = pensamento
                self.historico.append(msg)
                if nudge_idx is not None:  # remove a resposta prematura e o lembrete do histórico
                    del self.historico[nudge_idx:nudge_idx + 2]
                turno.resposta = resposta
                turno.tempo_total = time.time() - t_ini
                return turno

            # ---- rodada de ferramentas -------------------------------------
            adiadas = chamadas[self.max_calls_per_round:]
            chamadas = chamadas[: self.max_calls_per_round]
            msg = {
                "role": "assistant",
                "content": conteudo.split(TOOL_CALL_START, 1)[0].strip(),
                "tool_calls": [
                    {"type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in chamadas
                ],
            }
            if pensamento:
                msg["thinking"] = pensamento
            self.historico.append(msg)

            resultados = []
            for c in chamadas:
                if self.verbose_tools:
                    args_str = ", ".join(f"{k}={v!r}" for k, v in c.arguments.items())
                    print(f"{CIANO}🔧 {c.name}({args_str}){RESET}", file=sys.stderr, flush=True)
                resultados.append(self._executar(c))
                turno.chamadas.append(c)

            conteudo_tool = "\n\n".join(resultados) if len(resultados) > 1 else resultados[0]
            if adiadas:
                conteudo_tool += (
                    f"\n\n[Aviso: só as {self.max_calls_per_round} primeiras chamadas foram executadas; "
                    f"{len(adiadas)} ficaram de fora ({', '.join(c.name for c in adiadas)}). Prefira uma consulta SQL "
                    "que traga tudo de uma vez, ou peça o restante na próxima rodada.]"
                )
            if rodada >= self.max_tool_rounds - 1:
                conteudo_tool += (
                    "\n\n[Aviso: limite de chamadas atingido. Responda ao usuário agora com as "
                    "informações já obtidas, sem novas chamadas de ferramenta.]"
                )
            self.historico.append({"role": "tool", "content": conteudo_tool})

        turno.tempo_total = time.time() - t_ini
        return turno  # inalcançável
