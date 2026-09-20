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
**como** consultar a base — SQL, busca vetorial, busca por nome, leitura do registro completo ou
leitura da página oficial — e encadeia essas ferramentas até ter a resposta. Não depende de nenhuma API
externa: embeddings (`BAAI/bge-m3`, 1024 dims) e LLM rodam na GPU local.

```
pergunta ──▶ LFM2.5 (raciocínio <think>) ──▶ chamada de ferramenta ──▶ PostgreSQL/pgvector ──┐
                 ▲                                                                            │
                 └──────────────── resultado volta ao modelo (até 6 rodadas) ◀────────────────┘
                 └──▶ resposta final com fontes (URLs)
```

| Ferramenta | Quando o modelo usa | Implementação |
|---|---|---|
| `buscar_por_nome(nome, tipo)` | pergunta cita pessoa/disciplina/curso/projeto | `pg_trgm` + `unaccent` em `doc_completos.titulo` |
| `buscar_semantica(consulta, tipo)` | pergunta temática, sem nome exato | pgvector `<=>` nas tabelas `disciplinas/projetos/servidores/cursos` |
| `consultar_sql(sql)` | contagens, filtros, listas | `SELECT` somente leitura (papel `rag_leitor`, timeout 15 s, `LIMIT` forçado) |
| `obter_registro(doc_id)` | matriz curricular, turmas, equipe, currículo | JSONB de `doc_completos` |
| `ler_pagina(url)` | conferir/atualizar a partir da URL oficial | HTTP restrito a `*.ufpel.edu.br` |
| `buscar_web(consulta)` | só com `--web`; assuntos fora da base | DuckDuckGo (`ddgs`) |

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

### 2. Ingestão com embeddings locais

```bash
cp aplicacao/.env.example aplicacao/.env      # EMBEDDING_PROVIDER=local já vem configurado
cd crawler && python ingest_ufpel.py --reset --delay 0
docker exec -i rag_postgres psql -U gdlima -d semanticdb < ../docker/sql/pos_ingestao.sql   # índices HNSW/GIN
```

`providers.get_embeddings()` agora escolhe o provedor por `EMBEDDING_PROVIDER` (`local` | `nvidia`);
todo o restante do minicurso (ingestão, busca, pipeline) continua funcionando sem chave NVIDIA.

### 3. Perguntar ao agente

```bash
cd aplicacao
python agente_rag.py                                      # chat interativo
python agente_rag.py -p "Quem coordena o curso de Ciência da Computação?"
python agente_rag.py -p "Quantos projetos estão ativos?" --json   # resposta + rastro das ferramentas
python agente_rag.py --mostrar-pensamento                 # exibe o raciocínio
python agente_rag.py --web                                # habilita busca na internet
python main.py --etapa agente "sua pergunta"              # via menu do minicurso
```

### 4. Avaliar perguntas e respostas em lote

```bash
python avaliar_agente.py --perguntas ../avaliacao/perguntas_exemplo.jsonl --judge
python avaliar_agente.py -q "Quantos servidores há na base?" -q "Quem coordena o PPGC?"
```

Cada linha do JSONL tem `pergunta` e, opcionalmente, `resposta_esperada`, `deve_conter`,
`deve_conter_algum` e `ferramentas_esperadas`. O script gera `avaliacao/resultados/<run>/resultados.jsonl`
(resposta, ferramentas chamadas com argumentos e resultados, tempos, pensamentos) e `relatorio.md`
com acerto de conteúdo, ROUGE-L, similaridade de embeddings e, com `--judge`, notas 0–10 de
fidelidade/relevância/completude dadas pelo próprio modelo usando o contexto recuperado como evidência.

Arquivos: [`aplicacao/agente_lfm.py`](aplicacao/agente_lfm.py) (laço agêntico genérico e parsing de tool calls),
[`aplicacao/agente_rag.py`](aplicacao/agente_rag.py) (ferramentas, prompt com esquema SQL, CLI),
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
