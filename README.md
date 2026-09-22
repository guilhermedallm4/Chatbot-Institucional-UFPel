# Minicurso RAG — SACOMP (Semana Academica da Comnputação) - UFPel

Material prático do minicurso de **RAG (Retrieval-Augmented Generation)**. O projeto implementa, passo a passo, um chatbot institucional que responde perguntas sobre disciplinas, projetos, servidores, cursos e unidades da UFPel, combinando busca vetorial (PostgreSQL + pgvector), busca híbrida (BM25 + semântica), reranking, roteamento por tipo de conteúdo, guardrails e avaliação automática.

![Arquitetura RAG Institucional](RAG_institucional.png)

## Sumário

- [Arquitetura do projeto](#arquitetura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Variáveis de ambiente (.env)](#variáveis-de-ambiente-env)
- [Coleta e ingestão de dados (crawler)](#coleta-e-ingestão-de-dados-crawler)
- [Executando a aplicação](#executando-a-aplicação)
- [Interface web (Streamlit)](#interface-web-streamlit)
- [Notebook do curso](#notebook-do-curso)
- [Acesso ao banco de dados (psql / pgAdmin)](#acesso-ao-banco-de-dados-psql--pgadmin)

## Arquitetura do projeto

```
minicurso_rag/
├── aplicacao/            # Código principal do RAG
│   ├── main.py           # Ponto de entrada (CLI com etapas do curso)
│   ├── app.py            # Interface web (Streamlit)
│   ├── config.py         # Configuração central (DB, coleções, chunking)
│   ├── providers.py      # Provedores de embeddings e LLM (NVIDIA / Google)
│   ├── chunking.py       # Segmentação de documentos
│   ├── store.py          # Ingestão no banco vetorial (pgvector)
│   ├── search.py         # Busca semântica
│   ├── hybrid_search.py  # Busca híbrida (BM25 + semântica)
│   ├── reranker.py       # Reranking com cross-encoder
│   ├── router.py         # Roteamento de perguntas para a coleção correta
│   ├── keyword_extractor.py  # Extração de palavras-chave da pergunta
│   ├── pipeline.py       # Pipeline RAG completo (retrieval + geração)
│   ├── rag.py            # Cadeia RAG básica
│   ├── chatbot.py        # Chatbot interativo via terminal
│   ├── guardrails.py     # Safeguards (entrada/saída do LLM)
│   └── evaluation.py     # Avaliação (MRR, BERTScore, ROUGE-L, LLM Judge)
├── crawler/               # Coleta de dados do portal UFPel
│   ├── crawl_ufpel.py     # Crawler (BFS + por tipo de página)
│   ├── ingest_ufpel.py    # Conversão e ingestão segmentada por coleção
│   ├── run_pipeline.py    # Orquestra crawl + ingestão em um único comando
│   └── dados_ufpel.json   # Dados já coletados (exemplo)
├── notebook/
│   └── minicurso_rag.ipynb  # Notebook didático do minicurso
├── setup/
│   ├── setup_ambiente.sh  # Script de instalação (PostgreSQL, pgvector, pgAdmin, venv)
│   └── requirements.txt   # Dependências Python da aplicação
└── README.md
```

### Fluxo geral

1. O **crawler** coleta páginas do portal da UFPel e classifica cada página por tipo (`disciplina`, `projeto`, `servidor`, `unidade`, `curso`, `portal_geral`).
2. A **ingestão** segmenta os textos em chunks e grava os embeddings em coleções separadas no **PostgreSQL + pgvector**, uma por tipo de conteúdo.
3. Ao receber uma pergunta, o **router** decide qual(is) coleção(ões) consultar.
4. O **pipeline RAG** recupera os trechos mais relevantes (busca semântica, híbrida e/ou reranking) e gera a resposta com o LLM, podendo aplicar **guardrails** e **extração de keywords**.
5. A **avaliação** mede a qualidade da recuperação (MRR) e da geração (BERTScore, ROUGE-L, LLM-as-judge).

## Pré-requisitos

- Ubuntu 24.04 LTS (testado) ou similar
- Python 3.12+
- PostgreSQL com extensão [pgvector](https://github.com/pgvector/pgvector)
- Chaves de API:
  - **NVIDIA NIM** (embeddings + LLM de fallback) — obtenha em https://build.nvidia.com
  - **Google Gemini** (LLM principal) — obtenha em https://aistudio.google.com/app/apikey

## Instalação

O script `setup/setup_ambiente.sh` automatiza toda a configuração do ambiente: instala PostgreSQL, pgvector e pgAdmin 4, cria o banco e o usuário, cria as tabelas/índices de referência, cria o ambiente virtual Python, instala as dependências e gera o arquivo `.env`.

```bash
cd setup
bash setup_ambiente.sh
```

Ao final, ative o ambiente virtual nas próximas sessões com:

```bash
source setup/.venv/bin/activate
```

### Instalação manual (alternativa)

Se preferir não usar o script:

```bash
# 1. Dependências do sistema
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib postgresql-client \
    postgresql-16-pgvector libpq-dev python3 python3-pip python3-venv python3-dev

# 2. Banco de dados
sudo -u postgres psql -c "CREATE ROLE gdlima WITH LOGIN SUPERUSER PASSWORD '12345';"
sudo -u postgres psql -c "CREATE DATABASE semanticdb OWNER gdlima;"
sudo -u postgres psql -d semanticdb -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 3. Ambiente virtual e dependências Python
python3 -m venv .venv
source .venv/bin/activate
pip install -r setup/requirements.txt

# (opcional) dependências do crawler
pip install -r crawler/requirements_crawler.txt
```

## Variáveis de ambiente (.env)

Crie o arquivo `aplicacao/.env` (o `setup_ambiente.sh` cria automaticamente um modelo) com:

```dotenv
# NVIDIA NIM (OBRIGATÓRIO — embeddings; LLM de fallback)
NVIDIA_API_KEY=sua_chave_aqui

# Google Gemini (OBRIGATÓRIO — LLM principal)
GOOGLE_API_KEY=sua_chave_aqui

# PostgreSQL (opcional — usa os valores abaixo como padrão)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=semanticdb
DB_USER=gdlima
DB_PASS=12345
```

## Coleta e ingestão de dados (crawler)

O crawler vive em `crawler/` e tem dependências próprias:

```bash
cd crawler
pip install -r requirements_crawler.txt
```

Executar o pipeline completo (crawl + ingestão segmentada):

```bash
# Crawl de todos os tipos (disciplinas, projetos, servidores, unidades, cursos) + ingestão
python run_pipeline.py --all-types --reset

# Apenas alguns tipos
python run_pipeline.py --disciplines-only --projects-only --reset

# Servidores, limitando a 200 páginas
python run_pipeline.py --servidores-only --servidores-max 200 --reset

# Crawl geral (BFS), 150 páginas, profundidade 3 (padrão)
python run_pipeline.py

# Re-ingerir a partir de um JSON já coletado, sem novo crawl
python run_pipeline.py --from-json dados_ufpel.json --reset
```

Os dados coletados são salvos em `dados_ufpel.json` e cada tipo de página é gravado em sua própria coleção no pgvector (`ufpel_disciplinas`, `ufpel_projetos`, `ufpel_servidores`, `ufpel_unidades`, `ufpel_cursos`, `ufpel_portal_geral`).

## Executando a aplicação

Todos os comandos abaixo devem ser executados dentro de `aplicacao/`, com o ambiente virtual ativado:

```bash
cd aplicacao
source ../setup/.venv/bin/activate   # ajuste o caminho do venv se necessário
```

O ponto de entrada `main.py` permite executar cada etapa do curso isoladamente ou em sequência:

```bash
python main.py                        # menu interativo
python main.py --etapa providers      # testa embeddings e LLM
python main.py --etapa chunking       # demonstra segmentação de documentos
python main.py --etapa ingestao       # ingere documentos de exemplo
python main.py --etapa busca          # busca semântica + comparação de métricas
python main.py --etapa busca "query"  # busca com pergunta personalizada
python main.py --etapa rag            # pipeline RAG completo (uma pergunta)
python main.py --etapa rag "query"    # RAG com pergunta personalizada
python main.py --etapa chatbot        # chatbot interativo via terminal
python main.py --etapa tudo           # ingestão + busca + chatbot

# Etapas avançadas
python main.py --etapa hibrido        # busca híbrida BM25 + semântica
python main.py --etapa reranker       # reranking com cross-encoder
python main.py --etapa guardrails     # guardrails e safeguards
python main.py --etapa eval           # avaliação: MRR + BERTScore + LLM Judge

# Flag adicional
python main.py --etapa ingestao --reset   # recria a coleção do zero antes de ingerir
```

## Interface web (Streamlit)

```bash
cd aplicacao
streamlit run app.py
```

A interface permite:
- Escolher o modo de recuperação: **base** (semântico), **híbrido** (BM25 + semântico) ou **completo** (híbrido + reranker);
- Ativar a extração de palavras-chave da pergunta;
- Ingerir documentos de exemplo ou recriar a coleção do zero;
- Enviar um PDF para ingestão sob demanda;
- Conversar com o chatbot, que identifica automaticamente a coleção correta para cada pergunta.

## Agente local (LFM2.5 + SQL + pgvector) — branch `feature/agente-lfm2`

Alternativa ao pipeline RAG fixo: um **agente** baseado no modelo local
[LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) que decide, a cada pergunta,
**como** consultar a base — SQL nas views, busca vetorial, busca por nome, ficha completa da entidade
ou leitura da página oficial — e encadeia essas ferramentas até ter a resposta. Não depende de API
externa: embeddings (`BAAI/bge-m3`, 1024 dims) e LLM rodam na GPU local.

O banco segue o modelo **relacional + vetorial** descrito em
[`crawler/README_computacao.md`](crawler/README_computacao.md) e
[`crawler/README_portal_ppgc.md`](crawler/README_portal_ppgc.md): três acervos no mesmo PostgreSQL
(institucional `curso/disciplina/servidor/projeto/turma`, portal `port_*`, PPGC `ppgc_*`), views `vw_*`
como superfície plana para o LLM escrever SQL e tabelas `emb_*` com **um embedding por faceta** da
entidade, sempre carregando a chave (`servidor_id`, `curso_codigo`, `edital_id`…). Essa chave é o que
permite a mescla: um acerto semântico vira `JOIN`, e um filtro relacional restringe a busca vetorial.

```
pergunta ──▶ LFM2.5 (raciocínio <think>) ──▶ ferramenta ──▶ PostgreSQL (vw_* / emb_*) ──┐
                 ▲                                                                      │
                 └──────────── resultado volta ao modelo (até 6 rodadas) ◀──────────────┘
                 └──▶ resposta final com fontes (URLs)
```

| Ferramenta | Quando o modelo usa | Implementação |
|---|---|---|
| `buscar_por_nome(nome, entidade)` | nome próprio: pessoa, disciplina, curso, projeto, edital, norma, notícia | `pg_trgm` + `unaccent` nas colunas de nome; devolve a **chave** |
| `buscar_semantica(consulta, fonte)` | tema, "algo ligado a…", "como faço para…" | ANN nas 16 tabelas `emb_*` (índice HNSW sobre `halfvec`), com `escopo`, `metadata` e chave |
| `consultar_sql(sql)` | lista, contagem, soma, filtro por data, "todos os X" | `SELECT` somente leitura (papel `rag_leitor`, timeout, `LIMIT` forçado) contra as views `vw_*` |
| `detalhar(entidade, chave)` | ficha completa após localizar | mescla todas as tabelas ligadas (servidor → formação, áreas, cursos, projetos, turmas…) |
| `descrever_tabela(nome)` | coluna desconhecida / SQL falhou | `information_schema` |
| `ler_pagina(url)` | conferir/atualizar pela página oficial | HTTP restrito a `*.ufpel.edu.br` |
| `buscar_web(consulta)` | só com `--web`; assuntos fora da base | DuckDuckGo (`ddgs`) |

O prompt de sistema é montado a partir do banco (contagens por acervo, colunas de todas as views) mais
as regras de negócio dos READMEs do crawler (`vinculo_ativo`, `versao_atual`, normas vigentes, nível do
edital, prazos por nível, escolha do acervo). Veja-o com `python agente_rag.py --prompt`.

### 1. Banco em Docker (acessível externamente)

```bash
cd docker
cp .env.example .env          # ajuste POSTGRES_PASSWORD antes de expor na rede!
docker compose up -d          # PostgreSQL 16 + pgvector 0.8, porta 5432 em 0.0.0.0
docker compose --profile admin up -d   # (opcional) pgAdmin em http://<host>:5050
sudo ufw allow 5432/tcp       # se o firewall estiver ativo
```

O `init/01_extensoes.sql` cria `vector`, `pg_trgm`, `unaccent` e o papel somente leitura `rag_leitor`
(usado pela ferramenta SQL do agente). Teste de fora: `psql -h <IP-da-máquina> -U gdlima -d semanticdb`.

### 2. Schema + carga (relacional e vetorial) com embeddings locais

```bash
cp aplicacao/.env.example aplicacao/.env      # EMBEDDING_PROVIDER=local, EMBEDDING_DIMS=1024
cd crawler
python load_computacao.py        --input computacao.json        --schema   # 31 tabelas + 5 views + 5 emb_*
python load_portal_computacao.py --input portal_computacao.json --schema   # port_*  (antes do PPGC!)
python load_ppgc.py              --input ppgc.json              --schema   # ppgc_*  (cria vw_computacao_noticia)
```

Os DDLs foram escritos para o `nemotron-3-embed-1b` (2048 dims). Os loaders substituem `vector(2048)`
e `halfvec(2048)` por `EMBEDDING_DIMS` ao aplicar o schema, então o mesmo SQL serve para o `bge-m3`
local (1024). Para voltar à NVIDIA: `EMBEDDING_PROVIDER=nvidia`, `NVIDIA_API_KEY`,
`EMBEDDING_MODEL_NVIDIA=nvidia/nemotron-3-embed-1b`, `EMBEDDING_DIMS=2048` e rode os três loaders
com `--schema` de novo. A ingestão antiga (`ingest_ufpel.py`, tabelas `*_info`/`doc_completos`) continua
funcionando e convive no mesmo banco, mas o agente usa o schema novo.

### 3. Perguntar ao agente

```bash
cd aplicacao
python agente_rag.py                                      # chat interativo
python agente_rag.py -p "Quem coordena o curso de Ciência da Computação?"
python agente_rag.py -p "Quais professores pesquisam codificação de vídeo e sua titulação?" --json
python agente_rag.py --pensar --mostrar-pensamento        # liga e exibe o raciocínio <think> (padrão: desligado)
python agente_rag.py --web                                # habilita busca na internet
python agente_rag.py --device cpu                         # roda o LLM na RAM, sem GPU (veja a seção abaixo)
python main.py --etapa agente "sua pergunta"              # via menu do minicurso
```

### 3b. Rodar o LLM na RAM (sem GPU)

`--device cpu` (ou `AGENT_DEVICE=cpu` no `.env`) carrega o modelo na memória principal.
Cabe com folga, mas a latência inviabiliza uso interativo. Medições nesta máquina
(24 núcleos, AVX2 sem AVX-512/AMX, 125 GB de RAM, RTX 4090):

| | RAM/VRAM | prefill (4.765 tokens) | geração | pergunta `coord_cc` ponta a ponta |
|---|---|---|---|---|
| GPU, bfloat16 | 6,3 GB de VRAM | 0,21 s | 32,4 tok/s | **6,0 s** |
| CPU, float32 (transformers) | 12,2 GB de RAM | ~60 s | 2,0 tok/s | **287,4 s** |
| CPU, Q4_K_M (llama.cpp) | 1,6 GB de RAM | 22,8 s (209 tok/s) | 14,2 tok/s | ~60–90 s (estimado) |

Por que dói tanto: **99 % da latência é geração do LLM** — as consultas ao PostgreSQL
custam 0,03 s por pergunta em média. E o prompt de sistema tem ~4,7 mil tokens (esquema
das views + regras), que são re-processados a cada rodada de ferramenta; em CPU o prefill
domina, não a geração.

Em `float32` de propósito: esta CPU só tem AVX2, então `bfloat16` é emulado e fica **mais
lento** (1,7 tok/s) que `float32` (2,0 tok/s). O `--device cpu` já escolhe `float32`
sozinho; `AGENT_DTYPE` força outro.

Os embeddings (`BAAI/bge-m3`) rodam em CPU sem problema: 1,2 GB de RAM e 95 ms por
consulta, contra ~10 ms na GPU. Só o LLM é inviável.

Se a GPU estiver ocupada e for preciso responder mesmo assim, o caminho prático é o
GGUF oficial com llama.cpp (`LiquidAI/LFM2.5-2.6B-GGUF`, `Q4_K_M`, 1,6 GB), que é cerca
de 7× mais rápido que o `transformers` em CPU e ainda usa um oitavo da RAM. Isso exige
trocar o backend de geração do agente, o que ainda não está implementado.

### 4. Avaliar perguntas e respostas em lote

```bash
python avaliar_agente.py --perguntas ../avaliacao/perguntas_computacao.jsonl --judge
python avaliar_agente.py -q "Quantas turmas há em 2026/2?" -q "Qual o edital mais recente do mestrado?"
```

Cada linha do JSONL tem `pergunta` e, opcionalmente, `resposta_esperada`, `deve_conter`,
`deve_conter_algum` e `ferramentas_esperadas`. O script gera `avaliacao/resultados/<run>/resultados.jsonl`
(resposta, ferramentas chamadas com argumentos e resultados, tempos, pensamentos) e `relatorio.md`
com acerto de conteúdo, ROUGE-L, similaridade de embeddings e, com `--judge`, notas 0–10 de
fidelidade/relevância/completude dadas pelo próprio modelo usando o contexto recuperado como evidência.
`perguntas_exemplo.jsonl` é o conjunto antigo (schema `*_info`); `perguntas_computacao.jsonl` cobre o
schema novo (SQL, nome, semântica, híbrida, PPGC e portal). Na avaliação de 20/09/2026 o raciocínio `<think>`
não melhorou a acurácia do LFM2.5-2.6B neste schema e quadruplicou o tempo, por isso ele vem desligado por
padrão (`--pensar` liga). O agente também corrige três falhas típicas de modelos pequenos: resposta sem
consultar nada, plano anunciado sem chamada ("Vou consultar…") e chamada a uma view como se fosse função.

Arquivos: [`aplicacao/agente_lfm.py`](aplicacao/agente_lfm.py) (laço agêntico genérico e parsing de tool calls),
[`aplicacao/agente_rag.py`](aplicacao/agente_rag.py) (ferramentas, prompt com esquema, CLI),
[`aplicacao/avaliar_agente.py`](aplicacao/avaliar_agente.py), [`docker/`](docker/).

## Notebook do curso

O notebook didático com a explicação passo a passo de cada conceito está em `notebook/minicurso_rag.ipynb`:

```bash
cd notebook
jupyter notebook minicurso_rag.ipynb
```

## Acesso ao banco de dados (psql / pgAdmin)

```bash
psql -h localhost -U gdlima -d semanticdb
```

Consulta de exemplo (busca por similaridade de cosseno):

```sql
SELECT id, conteudo,
       embedding <=> '[0.1,0.2,...]' AS distancia
FROM documentos
ORDER BY embedding <=> '[0.1,0.2,...]'
LIMIT 5;
```

Operadores disponíveis no pgvector:
- `<=>` distância por cosseno (padrão NLP)
- `<->` distância euclidiana (L2)
- `<#>` produto interno negativo

O **pgAdmin 4** fica disponível em `http://127.0.0.1/pgadmin4` após a instalação (registre um servidor apontando para `localhost:5432`, usuário `gdlima`).
