# Dataset de Computação — modelo relacional + vetorial

Pipeline de coleta e modelagem dos **5 cursos de Computação da UFPel**, desenhado
para que o LLM possa **escolher entre montar um SQL ou fazer busca semântica** —
e, quando útil, combinar os dois.

| Código | Curso | Nível / Grau |
|--------|-------|--------------|
| 3900 | Ciência da Computação | Graduação / Bacharelado |
| 3910 | Engenharia de Computação | Graduação / Bacharelado |
| 7057 | Computação | Pós-Graduação / Mestrado Acadêmico |
| 8102 | Computação | Pós-Graduação / Doutorado |
| 9130 | Especialização em Computação na Educação Básica | Pós-Graduação / Especialização |

## Arquivos

| Arquivo | Papel |
|---------|-------|
| `crawl_computacao.py` | Crawler async. Saída: JSON com **linhas de tabela** já normalizadas |
| `schema_computacao.sql` | DDL das 31 tabelas + 5 views + índices (relacional e pgvector) |
| `load_computacao.py` | Carga no Postgres: relacional (TRUNCATE+INSERT) e embeddings |

```bash
pip install -r requirements_crawler.txt

# 1. coletar  (~2 min, ~400 requisições)
python crawl_computacao.py --output computacao.json

# 2. inspecionar sem banco: contagens + textos que serão vetorizados
python load_computacao.py --input computacao.json --dry-run

# 3. criar o schema e carregar tudo
python load_computacao.py --input computacao.json --schema

# variações
python load_computacao.py --input computacao.json --skip-embeddings   # só relacional
python load_computacao.py --input computacao.json --only-embeddings   # só revetorizar
python crawl_computacao.py --cursos 3900 3910 --cache-dir .cache_html # subconjunto + cache
```

## A decisão central: SQL ou semântico?

A regra que o roteador do RAG deve seguir:

| A pergunta pede… | Rota | Onde |
|---|---|---|
| lista, contagem, soma, filtro exato, "todos os X" | **SQL** | views `vw_*` |
| um nome próprio conhecido (curso, disciplina, professor) | **SQL** | `pg_trgm` no nome |
| tema, assunto, semelhança, "algo ligado a…" | **Semântico** | tabelas `emb_*` |
| tema **e** fato ("professores com projeto de IA e sua titulação") | **Híbrido** | `emb_*` → chave → `JOIN` |

O que torna o híbrido possível: **toda tabela `emb_*` carrega a chave da
entidade** (`curso_codigo`, `disciplina_codigo`, `servidor_id`, `projeto_id`,
`turma_id`). Um acerto vetorial sempre pode virar um JOIN relacional, e um
filtro relacional sempre pode restringir a busca vetorial.

## Modelo de dados

### Domínio 1 — Curso

```
curso (codigo_ufpel PK)
 ├─ curso_conceito         Enade/CPC por ano
 ├─ curso_info_secao       aba Informações — 1 linha por seção  ──► emb_curso
 ├─ curso_forma_ingresso   legenda das siglas de cota (AC, LB_EP, LI_PPI…)
 ├─ curso_vaga             vagas por processo × cota (formato longo)
 └─ curriculo_versao       versões de currículo (1027, 1028 ATUAL, 1029)
```

A aba **Informações** é modelada como `(curso, secao, texto)` em vez de colunas
fixas porque o conjunto de seções muda por nível de curso: graduação tem
*Contextualização / Objetivos / Perfil do Egresso / Competências e habilidades /
Organização Curricular / Procedimentos e metodologias / Avaliação / Integração
com a Pesquisa / Acompanhamento de Egressos*; pós tem *Apresentação / Área de
Concentração / Linhas de Pesquisa / Créditos necessários*. Cada seção vira **um
embedding próprio** — é a granularidade que faz "qual o perfil do egresso de
Engenharia de Computação?" recuperar a seção certa em vez de diluir o sinal.

`curso_vaga` sai da matriz publicada (processo × cota) para formato longo, o que
transforma "quantas vagas o SISU 2026 ofereceu" num `SUM ... GROUP BY`.

### Domínio 2 — Disciplina e matriz curricular

```
disciplina (codigo PK)
 ├─ disciplina_conteudo      Ementa/Objetivos/Conteúdo/Bibliografia  ──► emb_disciplina
 ├─ disciplina_bibliografia  1 linha por referência (básica/complementar)
 └─ disciplina_equivalencia  equivalentes por curso de destino

curso_matriz (curso, versao, disciplina)  ← a GRADE
 └─ matriz_prerequisito (curso, versao, disciplina, prereq)
```

**A versão de currículo está na chave da matriz.** É o ponto que a pergunta
original levanta: a grade muda com o ano de ingresso. Modelar `curso_matriz` sem
`versao` obrigaria a escolher uma grade só e daria a resposta errada para quem
ingressou antes.

Limite real da fonte: a aba *Matriz Curricular* do portal publica **apenas a
versão vigente**. As versões anteriores existem no portal só indiretamente, na
aba de turmas — capturadas em `turma_curriculo`. Por isso:

* consulta de grade **sem versão explícita** → filtrar `curriculo_versao.is_atual`;
* consulta sobre versão antiga → o que existe é a **oferta** daquela versão
  (`turma_curriculo`), não a grade completa.

`disciplina` é entidade independente de curso: `ALGORITMOS E PROGRAMAÇÃO`
(22000294) serve CC, EC e sete engenharias. `matriz_prerequisito` fica no par
(curso, versão) porque cursos diferentes exigem pré-requisitos diferentes para a
mesma disciplina.

### Domínio 3 — Professor

```
servidor (servidor_id PK)   ← campos do vínculo CORRENTE + flag vinculo_ativo
 ├─ servidor_funcao          "Coordenador de Curso de Graduação" + gratificação
 ├─ servidor_formacao        titulação/área/instituição/ano (parseado do Lattes)
 ├─ servidor_area_atuacao    área_geral + subárea  ◄── responde "a área de cada um"
 ├─ servidor_curso           ponte professor × curso  ◄── define "da computação"
 ├─ servidor_projeto         papel + ênfase + CH no projeto
 └─ servidor_disciplina_ministrada   histórico dos 3 últimos semestres
```

`servidor_area_atuacao` é a tabela que resolve o caso citado na especificação —
"buscar todos os professores da computação e o LLM sintetizar a área de cada um"
sai em **um SELECT com `string_agg`**, sem embedding nenhum.

### `vinculo_ativo` — corpo docente atual vs. histórico

Os campos de `servidor` vêm **sempre do bloco de vínculo corrente**, nunca dos
vínculos encerrados que o portal empilha na mesma ficha. Além disso, cada
servidor recebe `vinculo_ativo`, calculado a partir de duas evidências
independentes — porque nenhuma delas aparece em todas as fichas:

* **`Situação`** — presente nos vínculos permanentes ("Ativo Permanente",
  "Contr. Prof. Substituto"); valores como "Aposentado" marcam encerramento;
* **`Data de saída do Cargo`** — quando existe e já passou, o vínculo acabou.
  É o **único** sinal em fichas que não publicam `Situação`, o caso típico dos
  professores substitutos com contrato vencido.

Quem tem `vinculo_ativo = FALSE` **permanece no dataset de propósito**: essas
pessoas ministraram turmas nos últimos semestres, e removê-las quebraria
`turma_professor` e `servidor_disciplina_ministrada` (no dado atual, 6 dos 7
não-correntes têm disciplinas registradas). Portanto:

* pergunta sobre **quem é professor hoje** → `WHERE vinculo_ativo`;
* pergunta sobre **quem deu tal disciplina em 2026/1** → sem o filtro.

O texto vetorizado em `emb_servidor` também declara "Vínculo NÃO corrente:
encerrado em …" — sem isso o LLM apresentaria um ex-professor como docente
atual ao sintetizar o trecho recuperado. O `metadata` carrega
`vinculo_ativo` para permitir o mesmo filtro na busca semântica.

### Domínio 4 — Projeto (só vigentes)

```
projeto (projeto_id PK)
 ├─ projeto_info_secao   Objetivo Geral/Justificativa/Metodologia/Indicadores ──► emb_projeto
 └─ projeto_equipe       inclui discentes (servidor_id NULL)
```

### Domínio 5 — Turmas ofertadas

```
turma (turma_id PK = <disciplina>-<ano>-<sem>-<codigo>)
 ├─ turma_curriculo   curso × versão × semestre  ← onde a turma aparece
 ├─ turma_professor   nome + servidor_id resolvido
 └─ turma_horario     período, dia, hora início/fim
```

**Turma e `turma_curriculo` são tabelas separadas por um motivo concreto:** a
mesma turma (T1 de Cálculo 1, 2026/2) aparece sob **três versões de currículo**
na aba do curso. Se a versão entrasse na tabela `turma`, cada turma seria
gravada 3× e todo `COUNT` sairia inflado. No dado real: 122 turmas distintas
geram 283 vínculos curriculares.

O portal publica apenas o **nome** do professor na aba de turmas. O `servidor_id`
vem de um passo de reconciliação que cruza `(disciplina, ano, semestre, turma)`
com a aba *Disciplinas ministradas* da página do professor — no dataset atual,
143 de 143 turmas resolvidas.

### Camada vetorial

Cinco tabelas, todas com a mesma forma de colunas para que o retrieval seja
genérico:

```
<chave da entidade> | escopo | titulo | url | texto | embedding vector(2048) | metadata
```

* **`escopo`** é a faceta que gerou o texto (`ficha`, `ementa`, `objetivos`,
  `areas_atuacao`, `projetos`, `oferta`…). Permite busca dirigida:
  `WHERE escopo = 'ementa'`.
* **`metadata`** (JSONB, índice GIN) carrega filtros pré-busca — curso, unidade,
  titulação — sem precisar de JOIN.
* Cada texto é **autocontido**: começa identificando a entidade
  (`Disciplina ALGORITMOS E PROGRAMAÇÃO (22000294) — Ementa`). Um chunk
  recuperado tem que fazer sentido sozinho dentro do prompt.
* Textos longos são divididos em ~2000 caracteres (`objetivos`, `objetivos#2`…).
  O `nemotron-3-embed-1b` aceita ~4096 tokens (65.536 caracteres), mas chunk
  grande **dilui o sinal**: um termo perdido em 20 mil caracteres pontua quase
  como texto irrelevante. 2000 equilibra fragmentação e densidade.

`emb_servidor` inclui um escopo `projetos` com os projetos vigentes do professor.
É o que faz *"quais professores têm projetos de IA"* acertar já na primeira
busca, sem depender do segundo salto por `emb_projeto`.

## Views para o gerador de SQL

Exponha **as views** no prompt do gerador, não as 31 tabelas — o LLM acerta
muito mais escrevendo contra uma superfície plana do que montando 5 JOINs.

| View | Para quê |
|------|----------|
| `vw_professor_computacao` | professor + cursos + áreas já agregados em arrays |
| `vw_disciplina_curso` | matriz achatada: curso, versão, semestre, disciplina, ementa |
| `vw_turma_completa` | turma + disciplina + curso + versão + professores + horários — **uma linha por (turma × curso × versão)**: agregue ou filtre a versão quando a pergunta for sobre a turma em si |
| `vw_projeto_professor` | projeto vigente × professor (com papel e CH) |
| `vw_matriz_versao` | resumo por versão: nº disciplinas, créditos e horas totais |

## Consultas de referência

**Todos os professores da computação e a área de cada um** — SQL puro:

```sql
SELECT nome, titulacao, lotacao_nome,
       array_to_string(areas_atuacao, '; ') AS areas
FROM vw_professor_computacao
WHERE cardinality(cursos_codigos) > 0
  AND vinculo_ativo                     -- só o corpo docente atual
ORDER BY nome;
```

Sem o `AND vinculo_ativo` a mesma consulta responde "quem ministrou nos
últimos três semestres", incluindo substitutos com contrato já vencido.

**Grade do 4º semestre de Ciência da Computação, versão vigente:**

```sql
SELECT semestre_rotulo, disciplina_codigo, disciplina_nome,
       creditos, horas, carater
FROM vw_disciplina_curso
WHERE curso_codigo = '3900' AND versao_atual AND semestre_num = 4
ORDER BY disciplina_nome;
```

**Pré-requisitos de uma disciplina, no contexto do curso:**

```sql
SELECT d.nome AS disciplina, p.prereq_nome
FROM matriz_prerequisito p
JOIN disciplina d ON d.codigo = p.disciplina_codigo
WHERE p.curso_codigo = '3900'
  AND p.versao = (SELECT versao FROM curriculo_versao
                   WHERE curso_codigo = '3900' AND is_atual);
```

**Vagas por processo seletivo:**

```sql
SELECT processo, ano, semestre, SUM(vagas) AS total
FROM curso_vaga
WHERE curso_codigo = '3900' AND cota <> 'TOTAL'
GROUP BY processo, ano, semestre;
```

**Onde eu aproveito uma disciplina que já fiz:**

```sql
SELECT e.equivalente_nome, e.curso_nome
FROM disciplina_equivalencia e
JOIN disciplina d ON d.codigo = e.disciplina_codigo
WHERE d.nome ILIKE '%algoritmos e programação%';
```

**Aulas de um professor nesta semana** — agregando as versões, senão a turma
aparece uma vez por versão de currículo em que é ofertada:

```sql
SELECT disciplina_nome, codigo_turma, curso_nome, horarios,
       array_agg(DISTINCT versao ORDER BY versao) AS versoes
FROM vw_turma_completa
WHERE 'GUILHERME TOMASCHEWSKI NETTO' = ANY(professores)
  AND ano = 2026 AND semestre = 2
GROUP BY disciplina_nome, codigo_turma, curso_nome, horarios;
```

**Quais professores têm projetos ligados a inteligência artificial** — híbrido
(semântico + JOIN relacional):

```sql
-- split_part descarta o sufixo de chunk: 'objetivo_geral#2' → 'objetivo_geral'
WITH vizinhos AS (                      -- 1º: ANN, usa o índice HNSW
    SELECT projeto_id, embedding <=> %(q)s::vector AS dist
    FROM emb_projeto
    WHERE split_part(escopo, '#', 1) IN ('ficha', 'objetivo_geral', 'justificativa')
    ORDER BY embedding <=> %(q)s::vector
    LIMIT 60
), achados AS (                         -- 2º: um projeto pode acertar em vários escopos
    SELECT projeto_id, MIN(dist) AS dist
    FROM vizinhos
    GROUP BY projeto_id
)
SELECT vp.professor_nome, vp.professor_lotacao,
       vp.titulo AS projeto, vp.enfase,
       ROUND((1 - a.dist)::numeric, 3) AS score
FROM achados a
JOIN vw_projeto_professor vp ON vp.projeto_id = a.projeto_id
WHERE 1 - a.dist > 0.35 AND vp.professor_nome IS NOT NULL
ORDER BY a.dist;
```

**Busca semântica restrita por filtro relacional** (metadata evita o JOIN):

```sql
SELECT disciplina_codigo, titulo, texto,
       1 - (embedding <=> %(q)s::vector) AS score
FROM emb_disciplina
WHERE split_part(escopo, '#', 1) = 'ementa'
  AND metadata @> '{"cursos": ["Ciência da Computação"]}'
ORDER BY embedding <=> %(q)s::vector
LIMIT 5;
```

## Decisões de modelagem que valem registro

1. **Campo ausente = `NULL`, não a string `"Não há informações disponíveis"`.**
   A sentinela em texto quebra `COUNT`, `WHERE ... IS NULL` e qualquer agregação;
   renderizar "não informado" é responsabilidade da camada de síntese, não do
   banco.

2. **Chaves naturais do portal** (`codigo_ufpel`, código da disciplina,
   `servidor_id`, `projeto_id`) em vez de UUID. Torna a carga idempotente e
   deixa os JOINs legíveis para o LLM — `turma_id` é derivado
   (`22000294-2026-2-M11`), não uma sequence.

3. **`projeto_equipe.servidor_id` sem FK rígida.** A equipe traz discentes e
   servidores fora do escopo do crawl. `LEFT JOIN`, sem descartar a linha.

4. **Carga por refresh completo.** `TRUNCATE` + `INSERT` em lote na ordem de
   dependência: idempotente por construção, sem precisar de `ON CONFLICT` em
   nenhuma tabela.

## Armadilhas do portal tratadas no crawler

Vale conhecer porque cada uma delas produzia dado silenciosamente errado:

| Armadilha | Consequência se ignorada |
|---|---|
| Vínculos encerrados do servidor ficam no **mesmo** `div.ficha-dados`, num `div.vinculo.oculta-exibe-conteudo` colapsado | Ler a ficha achatada faz o vínculo antigo sobrescrever o ativo: cargo e titulação errados (ex.: "Professor Temporário / Graduação" no lugar de "Professor do Magistério Superior / Doutorado") |
| Fichas de substituto muitas vezes **não têm** o rótulo `Situação` — o encerramento só aparece em `Data de saída do Cargo` | Filtrar só por `Situação` deixa passar como ativo quem tem vínculo único já vencido (a página não ganha o rótulo "(VÍNCULO ENCERRADO)" porque não há outro vínculo para colapsar) |
| A aba de projetos do professor usa **uma** tabela com `th.tabela-quebra` separando Ensino/Extensão/Pesquisa | Pegar o primeiro `<th>` atribui a mesma ênfase a todos os projetos |
| A mesma turma aparece sob várias versões de currículo | Turma duplicada 3×, `COUNT` inflado |
| O link do Lattes está em `ul.contatos`, **fora** da `div#lattes` | `lattes_url` sempre vazio |
| Pré-requisitos ficam em `span.tabela-detalhe-info` na célula "Disciplina / Pré-requisitos" — e a coluna "Código" **também** tem link de disciplina | Pegar a primeira célula com link de disciplina não encontra o span: zero pré-requisitos |
| Projetos encerrados vêm marcados com `tr.finalizado` | Sinal mais confiável que só comparar datas |
| A aba "Turmas Ofertadas" da **disciplina** não tem curso/versão/semestre | Por isso a oferta é capturada na página do **curso**, e a da disciplina é ignorada de propósito |

## Escopo e limites

* **Coletado:** os 5 cursos e tudo que eles referenciam — 166 disciplinas,
  80 professores (73 com vínculo corrente), 164 projetos vigentes, 122 turmas de 2026/2.
* Professores de outras unidades (Instituto de Física e Matemática, por exemplo)
  entram quando ministram disciplina para os cursos de Computação. Filtre por
  `servidor.lotacao_nome` se quiser só o CDTec.
* **Não coletado:** abas *Alunos* e *Egressos* (fora do pedido); grade completa
  de versões antigas (o portal não publica); histórico de turmas além dos três
  últimos semestres.
* O curso 9130 é novo: sem turmas ofertadas, sem coordenador publicado e sem aba
  Professores. Recebe a versão sintética `unica` em `curriculo_versao`, para que
  a matriz tenha uma versão à qual se referir.
