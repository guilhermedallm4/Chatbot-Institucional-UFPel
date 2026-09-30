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

Dois backends, mesma interface:

  local       transformers + GPU/CPU desta máquina. O modelo emite as chamadas
              em texto, entre <|tool_call_start|> e <|tool_call_end|>.
  openrouter  API OpenAI-compatível do OpenRouter, sem GPU e sem baixar pesos.
              As chamadas vêm estruturadas em JSON (campo tool_calls), então o
              parsing de texto não é usado nesse caminho.

Escolha por AGENT_BACKEND=local|openrouter no .env, ou pelo parâmetro backend.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

MODEL_ID_DEFAULT = "LiquidAI/LFM2.5-2.6B"

# Mesmo modelo, servido pelo OpenRouter na faixa gratuita (:free).
OPENROUTER_MODEL_DEFAULT = "liquid/lfm-2.5-2.6b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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


def _args_json(bruto: Any) -> dict:
    """A API devolve `arguments` como string JSON; modelos pequenos às vezes mandam algo torto."""
    if isinstance(bruto, dict):
        return bruto
    try:
        d = json.loads(bruto or "{}")
        return d if isinstance(d, dict) else {"_": d}
    except (TypeError, json.JSONDecodeError):
        return {}


@dataclass
class ToolCall:
    name: str
    arguments: dict
    result: str = ""
    elapsed: float = 0.0
    id: str = ""          # usado só pelo backend OpenRouter (amarra resultado × chamada)


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
        device: str = "auto",
        dtype: Any = "auto",
        backend: str = "local",
        api_key: Optional[str] = None,
        base_url: str = OPENROUTER_BASE_URL,
        timeout: float = 120.0,
    ):
        """
        exigir_ferramenta      : se o modelo responder à 1ª rodada sem chamar ferramenta, recebe um
                                 lembrete e gera de novo (evita resposta "de cabeça" em modelos pequenos).
        max_calls_per_round    : limite de chamadas executadas por rodada (o resto é adiado).
        max_tool_result_chars  : corte de cada resultado de ferramenta antes de entrar no histórico.
        max_context_tokens     : acima disso, resultados antigos de ferramentas são resumidos no histórico
                                 para não estourar a janela (32k) nem a memória da GPU.
        device                 : "auto" (GPU se houver) | "cuda" | "cpu" (AGENT_DEVICE no .env).
        dtype                  : "auto" (bfloat16 na GPU, float32 na CPU) | "bfloat16" | "float32".
        backend                : "local" (transformers nesta máquina) | "openrouter" (API, sem GPU).
        api_key                : chave do OpenRouter; se vazia, lê OPENROUTER_API_KEY do ambiente.
        """
        self.backend = (backend or "local").strip().lower()
        if self.backend not in ("local", "openrouter"):
            raise ValueError(f"backend '{backend}' inválido: use 'local' ou 'openrouter'.")

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

        if self.backend == "openrouter":
            self._init_openrouter(model_id, api_key, base_url, timeout, mostrar_pensamento, stream)
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # device: "auto" (GPU se houver), "cuda", "cpu". dtype: "auto" escolhe bfloat16 na GPU e
        # float32 na CPU — em CPU sem AVX-512/AMX o bfloat16 é emulado e fica MAIS lento que fp32.
        if device in ("", "auto", None):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype in ("", "auto", None):
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        elif isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if device == "cpu":
            torch.set_num_threads(int(os.getenv("AGENT_CPU_THREADS", os.cpu_count() or 8)))

        print(f"[agente] Carregando {model_id} em {device} ({str(dtype).replace('torch.', '')}) ...", file=sys.stderr)
        t_carga = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=("auto" if device != "cpu" else "cpu"), dtype=dtype)
        self.model.eval()
        self.device = device
        print(f"[agente] Pronto em {time.time() - t_carga:.1f}s", file=sys.stderr)
        self.im_end_id = self.tokenizer.convert_tokens_to_ids(IM_END)
        self.streamer = _make_streamer_class()(self.tokenizer, mostrar_pensamento, silencioso=not stream)
        self.historico: list[dict] = []
        self.limpar()

    # ------------------------------------------------------------------
    # Backend OpenRouter
    # ------------------------------------------------------------------

    def _init_openrouter(self, model_id, api_key, base_url, timeout, mostrar_pensamento, stream):
        """Nenhum peso é baixado: só a sessão HTTP e a checagem da chave."""
        import requests

        self.model_id = (model_id if model_id and "/" in model_id and not model_id.startswith("LiquidAI")
                         else OPENROUTER_MODEL_DEFAULT)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not self.api_key or self.api_key == "sua_chave_aqui":
            raise EnvironmentError(
                "OPENROUTER_API_KEY não configurada. Crie uma chave gratuita em "
                "https://openrouter.ai/keys e coloque em aplicacao/.env como OPENROUTER_API_KEY=sk-or-...")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._sessao = requests.Session()
        self._sessao.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # opcionais do OpenRouter, aparecem nas estatísticas da conta
            "HTTP-Referer": "https://github.com/guilhermedallm4/Minicurso-RAG",
            "X-Title": "Chatbot Institucional UFPel",
        })
        self.device = "openrouter"
        self.tokenizer = None
        self.model = None
        self.streamer = None
        self._mostrar_pensamento = mostrar_pensamento
        self._stream = stream
        print(f"[agente] Backend OpenRouter — modelo {self.model_id} (sem GPU, sem download)", file=sys.stderr)
        self.historico: list[dict] = []
        self.limpar()

    @staticmethod
    def _mensagens_openai(historico: list[dict]) -> list[dict]:
        """
        Converte o histórico interno para o formato de mensagens da API OpenAI.

        Duas diferenças que importam: `arguments` viaja como STRING JSON (não dict),
        e cada resultado de ferramenta é uma mensagem própria, amarrada pelo
        tool_call_id. O histórico interno guarda os resultados em `resultados`
        justamente para permitir essa separação.
        """
        msgs: list[dict] = []
        for m in historico:
            papel = m["role"]
            if papel == "assistant" and m.get("tool_calls"):
                msgs.append({
                    "role": "assistant",
                    "content": m.get("content") or "",
                    "tool_calls": [
                        {"id": tc.get("id") or f"call_{i}", "type": "function",
                         "function": {"name": tc["function"]["name"],
                                      "arguments": json.dumps(tc["function"]["arguments"], ensure_ascii=False)}}
                        for i, tc in enumerate(m["tool_calls"])
                    ],
                })
            elif papel == "tool":
                ids = m.get("ids") or []
                resultados = m.get("resultados")
                if resultados:
                    for i, res in enumerate(resultados):
                        msgs.append({"role": "tool", "tool_call_id": ids[i] if i < len(ids) else f"call_{i}",
                                     "content": res})
                    if m.get("avisos"):
                        msgs.append({"role": "user", "content": m["avisos"]})
                else:  # mensagem de sistema disfarçada de tool (cobranças do runner)
                    msgs.append({"role": "user", "content": m.get("content", "")})
            else:
                msgs.append({"role": papel, "content": m.get("content", "")})
        return msgs

    def _gerar_openrouter(self, historico: list[dict], tools: bool,
                          max_new_tokens: Optional[int]) -> tuple[str, Optional[list[ToolCall]]]:
        corpo: dict[str, Any] = {
            "model": self.model_id,
            "messages": self._mensagens_openai(historico),
            "max_tokens": max_new_tokens or self.max_new_tokens,
            "temperature": self.temperature,
        }
        if tools and self.tools:
            corpo["tools"] = [t.schema() for t in self.tools.values()]
            corpo["tool_choice"] = "auto"

        ultimo_erro = ""
        for tentativa in range(3):
            try:
                r = self._sessao.post(f"{self.base_url}/chat/completions", json=corpo, timeout=self.timeout)
            except Exception as e:  # noqa: BLE001 — rede instável não pode derrubar o laço
                ultimo_erro = f"{type(e).__name__}: {e}"
                time.sleep(2 * (tentativa + 1))
                continue
            if r.status_code == 429:  # faixa gratuita tem limite por minuto
                espera = float(r.headers.get("Retry-After", 8 * (tentativa + 1)))
                print(f"[agente] limite de requisições do OpenRouter; aguardando {espera:.0f}s",
                      file=sys.stderr, flush=True)
                time.sleep(espera)
                ultimo_erro = "429 (limite de requisições)"
                continue
            if not r.ok:
                raise RuntimeError(f"OpenRouter respondeu {r.status_code}: {r.text[:300]}")
            dados = r.json()
            if "error" in dados and not dados.get("choices"):
                raise RuntimeError(f"OpenRouter: {str(dados['error'])[:300]}")
            msg = dados["choices"][0]["message"]
            texto = msg.get("content") or ""
            if msg.get("reasoning") and self._mostrar_pensamento:
                print(f"{CINZA}[pensando] {str(msg['reasoning'])[:2000]}{RESET}", file=sys.stderr)
            chamadas = [
                ToolCall(name=tc["function"]["name"], arguments=_args_json(tc["function"].get("arguments")),
                         id=tc.get("id") or f"call_{i}")
                for i, tc in enumerate(msg.get("tool_calls") or [])
            ]
            if self._stream and texto:
                print(texto, flush=True)
            return texto, chamadas
        raise RuntimeError(f"OpenRouter falhou em 3 tentativas. Último erro: {ultimo_erro}")

    # ------------------------------------------------------------------
    def limpar(self):
        self.historico = [{"role": "system", "content": self.system_prompt}]

    def set_stream(self, ativo: bool):
        self.streamer.silencioso = not ativo

    # ------------------------------------------------------------------
    def _gerar(self, historico: list[dict], tools: bool = True,
               max_new_tokens: Optional[int] = None) -> tuple[str, Optional[list[ToolCall]]]:
        """
        Devolve (texto_bruto, chamadas). No backend local `chamadas` é None e quem
        extrai as chamadas é parse_tool_calls sobre o texto; no OpenRouter elas já
        vêm estruturadas e o parsing de texto não é usado.
        """
        if self.backend == "openrouter":
            return self._gerar_openrouter(historico, tools, max_new_tokens)

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
        return self.tokenizer.decode(saida[0, input_ids.shape[1]:], skip_special_tokens=False), None

    def gerar_simples(self, prompt: str, system: Optional[str] = None, max_new_tokens: int = 512,
                      pensar: Optional[bool] = None) -> str:
        """
        Geração sem ferramentas nem histórico (usado pelo LLM-judge da avaliação).
        pensar=False pula o bloco <think>: útil quando se quer só um JSON curto e
        o raciocínio consumiria o orçamento de tokens.
        """
        hist = [{"role": "system", "content": system}] if system else []
        hist.append({"role": "user", "content": prompt})
        pensar_antes = self.pensar
        silencioso_antes = self.streamer.silencioso if self.streamer else None
        stream_antes = getattr(self, "_stream", None)
        if self.streamer:
            self.streamer.silencioso = True
        else:
            self._stream = False
        if pensar is not None:
            self.pensar = pensar
        try:
            bruto, _ = self._gerar(hist, tools=False, max_new_tokens=max_new_tokens)
        finally:
            self.pensar = pensar_antes
            if self.streamer:
                self.streamer.silencioso = silencioso_antes
            else:
                self._stream = stream_antes
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
        if self.backend == "openrouter":
            # sem tokenizer local: estimativa por caracteres (~3,5 por token em português)
            chars = sum(len(str(m.get("content", ""))) for m in historico) + len(self.system_prompt)
            return int(chars / 3.5)
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
            if len(m.get("content", "")) > 700:
                corte = "\n[... resultado antigo resumido para caber no contexto]"
                m["content"] = m["content"][:600] + corte
                if m.get("resultados"):
                    m["resultados"] = [r[:600] + corte if len(r) > 700 else r for r in m["resultados"]]
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
            bruto, chamadas_api = self._gerar(self.historico)
            pensamento, conteudo = _separar_pensamento(bruto)
            if pensamento:
                turno.pensamentos.append(pensamento)
            chamadas = chamadas_api if chamadas_api is not None else parse_tool_calls(conteudo, self.tools)

            # ---- anunciou o plano ("Vou consultar...") mas não emitiu a chamada: pede a chamada ----
            if not chamadas and self.tools and plano_nudges < 2 and _parece_plano(_strip_tool_calls(conteudo)):
                plano_nudges += 1
                self.historico.append({"role": "assistant", "content": _strip_tool_calls(conteudo)})
                self.historico.append({"role": "user", "content": (
                    "[sistema] Você descreveu o que vai fazer, mas não chamou nenhuma ferramenta. "
                    "Emita agora a chamada de ferramenta correspondente (ou, se já tem a informação, escreva a resposta final).")})
                bruto, chamadas_api = self._gerar(self.historico)
                pensamento, conteudo = _separar_pensamento(bruto)
                if pensamento:
                    turno.pensamentos.append(pensamento)
                del self.historico[-2:]
                chamadas = chamadas_api if chamadas_api is not None else parse_tool_calls(conteudo, self.tools)

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
                    bruto, chamadas_api = self._gerar(self.historico)
                    pensamento, conteudo = _separar_pensamento(bruto)
                    if pensamento:
                        turno.pensamentos.append(pensamento)
                    del self.historico[-2:]
                    resposta = _strip_tool_calls(conteudo)
                    chamadas_retry = (chamadas_api if chamadas_api is not None
                                      else parse_tool_calls(conteudo, self.tools))
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
                    bruto, _ = self._gerar(self.historico)
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
                    {"id": c.id or f"call_{rodada}_{i}", "type": "function",
                     "function": {"name": c.name, "arguments": c.arguments}}
                    for i, c in enumerate(chamadas)
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
            avisos = ""
            if adiadas:
                avisos += (
                    f"\n\n[Aviso: só as {self.max_calls_per_round} primeiras chamadas foram executadas; "
                    f"{len(adiadas)} ficaram de fora ({', '.join(c.name for c in adiadas)}). Prefira uma consulta SQL "
                    "que traga tudo de uma vez, ou peça o restante na próxima rodada.]"
                )
            if rodada >= self.max_tool_rounds - 1:
                avisos += (
                    "\n\n[Aviso: limite de chamadas atingido. Responda ao usuário agora com as "
                    "informações já obtidas, sem novas chamadas de ferramenta.]"
                )
            self.historico.append({
                "role": "tool",
                "content": conteudo_tool + avisos,          # backend local: tudo numa mensagem só
                "resultados": resultados,                    # OpenRouter: uma mensagem por chamada
                "ids": [c.id or f"call_{rodada}_{i}" for i, c in enumerate(chamadas)],
                "avisos": avisos.strip(),
            })

        turno.tempo_total = time.time() - t_ini
        return turno  # inalcançável
