# Portal da Computação-UFPel e PPGC — coleta, modelo e busca

Dois datasets novos, **separados entre si e separados do institucional**,
extraídos de `https://wp.ufpel.edu.br/computacao/`.

| Acervo | Fonte | Tabelas | Responde |
|---|---|---|---|
| **Institucional** (já existia) | `institucional.ufpel.edu.br` | sem prefixo (`curso`, `disciplina`, `servidor`…) | currículo, matriz, turmas, projetos, servidores |
| **Portal** (novo) | `wp.ufpel.edu.br/computacao` | `port_*` | notícias, páginas dos menus, FAQ da graduação, calendário acadêmico |
| **PPGC** (novo) | `.../computacao/ppgc` + editais soltos | `ppgc_*` | editais, normas, requisitos, FAQ de alunos, agenda do Programa, linhas de pesquisa |

Os três convivem no mesmo banco. O prefixo da tabela **é** a escolha de acervo
que o roteador do RAG faz.

## Arquivos

| Arquivo | Papel |
|---|---|
| `wp_common.py` | base comum: cliente da REST API, HTML→texto, seções, links, datas, partição portal × PPGC |
| `crawl_portal_computacao.py` | crawler do portal → `portal_computacao.json` |
| `crawl_ppgc.py` | crawler do PPGC → `ppgc.json` |
| `schema_portal_computacao.sql` | 17 tabelas + 3 `emb_*` + 6 views + índices |
| `schema_ppgc.sql` | 18 tabelas + 8 `emb_*` + 7 views + índices |
| `load_wp_common.py` | infraestrutura de carga (relacional + embeddings) |
| `load_portal_computacao.py` / `load_ppgc.py` | carga no Postgres |

```bash
pip install -r requirements_crawler.txt

# 1. coletar  (~25 requisições cada, ~1 min)
python crawl_portal_computacao.py --output portal_computacao.json
python crawl_ppgc.py             --output ppgc.json

# 2. inspecionar sem banco: contagens + os textos que serão vetorizados
python load_portal_computacao.py --input portal_computacao.json --dry-run
python load_ppgc.py              --input ppgc.json              --dry-run

# 3. criar o schema e carregar
python load_portal_computacao.py --input portal_computacao.json --schema
python load_ppgc.py              --input ppgc.json              --schema

# variações
python load_ppgc.py --input ppgc.json --skip-embeddings   # só relacional
python load_ppgc.py --input ppgc.json --only-embeddings   # só revetorizar
python crawl_ppgc.py --cache-dir .cache_wp                # reusa o HTTP baixado
python crawl_portal_computacao.py --relatorio-particao    # confere a divisão
```

> **Ordem importa na primeira carga:** `schema_ppgc.sql` cria a view
> `vw_computacao_noticia` (portal + PPGC unidos) somente se `port_post` já
> existir. Carregue o portal primeiro; se inverter, basta reexecutar
> `schema_ppgc.sql` depois.

## Duas decisões que definem tudo

### 1. REST API, não scraping

O site é WordPress com a API aberta em `/wp-json/wp/v2/`. Isso muda a natureza
do problema: `date`, `modified`, `author`, `categories` e `tags` chegam como
**campos**, não como texto a adivinhar. O enunciado pede "capture sempre a
data dos posts" — pela API isso é uma coluna; por HTML, seria uma expressão
regular contra "Publicado em 15 de julho de 2026" e a esperança de que o tema
nunca mude.

Custo: **~25 requisições** para os 746 posts, 191 páginas, 8 categorias e 445
arquivos de mídia. Um crawl de HTML equivalente passaria de mil.

O domínio está atrás de um WAF (SafeLine) que devolve 403 para cliente sem
cara de navegador. Os cabeçalhos que passam estão em `wp_common.HEADERS`, com
o aviso — mexer ali quebra o crawl inteiro de uma vez.

### 2. A partição portal × PPGC

`wp_common.pagina_e_ppgc` e `wp_common.post_e_ppgc` são as **únicas** funções
que decidem o acervo, e os dois crawlers chamam as mesmas. Se a regra morasse
em cada um, qualquer divergência criaria conteúdo duplicado (nos dois) ou
órfão (em nenhum) — e nada acusaria o erro.

Resultado: 191 páginas → 65 portal / 126 PPGC; 746 posts → 442 / 304.

**Para páginas** a regra é estrutural: subárvore (`ppgc/…`, `pos-graduacao/…`,
`graduate-program-in-computing/…`), depois slug (`selecao-…`, `edital-…`,
`dinter-…`), depois título. O slug importa porque o WordPress deixou vários
editais **na raiz do site**, sem `parent` — inclusive os de 2026.

**Para posts** a categoria `ppgc` resolve 287 dos 746. Ela sozinha não bastaria,
e o buraco era grande: **os 8 posts de 2026 inteiros** — nota 6 da CAPES,
editais de seleção 2026/1 e 2026/2, oferta de disciplinas — foram publicados
na categoria legada `noticias`, sem `ppgc`. Um acervo de pós-graduação que
para em 2025 é o pior resultado possível, porque quem pergunta sobre edital
quer o vigente. Daí o desempate em três passos:

1. tem a categoria `ppgc` → PPGC;
2. tem categoria informativa (`ccomp`, `ecomp`, `noticia`) → portal, sem discussão;
3. só tem categoria genérica/legada → decide pelo título.

O passo 2 é o que segura a heurística: um post marcado como Ciência da
Computação continua no portal mesmo citando "mestrado" de passagem.

## O que foi coletado

### Portal — `port_*`

```
port_secao (8)              Notícias · Sobre · Graduação · Pós-Graduação ·
                            Pesquisa · Ensino · Extensão · Calendário
 └─ port_menu_item (37)     o menu do cabeçalho, com hierarquia

port_pagina (65)            páginas dos menus, pt e en
 └─ port_pagina_secao (81)  a página quebrada nos títulos  ──► emb_port_pagina

port_post (442)             notícias, com data, autor, categorias, tags
 ├─ port_post_categoria (1.102) / port_post_tag (21)      ──► emb_port_post
 └─ port_link (958) / port_documento (145)

port_faq (62)               FAQ da Computação, pergunta a pergunta ──► emb_port_faq
port_pessoa (45)            docentes e técnicos, com id do institucional
port_grupo_pesquisa (14)    GACI, GAIA, GEPESC, Datalab, GASLN, ViTech, LUPS
port_calendario_evento (334) calendário acadêmico dia a dia (2021–2023)
```

### PPGC — `ppgc_*`

```
ppgc_pagina (126)           páginas do Programa
 └─ ppgc_pagina_secao (315)                               ──► emb_ppgc_pagina
ppgc_post (304)             notícias, com `assunto`       ──► emb_ppgc_post

ppgc_edital (47)            tipo · nível · ano/semestre · número oficial
 └─ ppgc_edital_documento (245)  os PDFs de cada edital, TIPADOS
                                                          ──► emb_ppgc_edital
ppgc_documento (372)        um registro por ARQUIVO + colunas de extração
 └─ ppgc_documento_chunk    (vazia — a etapa dos PDFs)    ──► emb_ppgc_documento

ppgc_normativo (46)         regimento, resoluções, portarias — COM vigência
                                                          ──► emb_ppgc_normativo
ppgc_faq (28)               FAQ de alunos                 ──► emb_ppgc_faq
ppgc_requisito (13)         créditos, proficiência, prazos, por nível
ppgc_linha_pesquisa (5)     as 5 linhas, com descrição    ──► emb_ppgc_linha_pesquisa
ppgc_disciplina (9)         ementa, responsável, aluno especial ──► emb_ppgc_disciplina
ppgc_docente (32)           corpo docente, Lattes, id do institucional
ppgc_calendario_evento (430) agenda do Programa, datada
ppgc_defesa (1)             defesas anunciadas
```

## Escolhas de modelagem que mudam a resposta

**A data é cidadã de primeira classe.** `data_publicacao` DATE resolve
ordenação e filtro; `ano`/`mes`/`semestre` resolvem agregação sem `EXTRACT`; e
`data_por_extenso` ("15 de julho de 2026") vai para dentro do **texto
vetorizado**, porque "o que saiu em julho de 2026?" casa por similaridade com
"julho de 2026" e não casa com "2026-07-15".

**Uma linha por seção, não uma por página.** O FAQ da Computação tem 43 mil
caracteres e o do PPGC, 42 mil. Vetorizados inteiros, "como peço segunda
chamada?" pontuaria igual a estágio, e-mail institucional e TCC — tudo no
mesmo vetor. Quebrados em Q&A com âncora, o acerto é cirúrgico e a resposta já
sai com `.../faq-da-computacao/#matricula` para citar.

**O calendário do PPGC veio do Google Agenda.** A página
`/ppgc/calendario-ppgc/` é só um `<iframe>` — zero texto. O `src` carrega o id
do calendário em base64, e todo calendário público do Google publica um
`.ics`: são de lá os 430 eventos datados. Sem isso, o dataset teria a página
"Calendário PPGC" e nenhuma data dentro. (O `DTEND` de evento de dia inteiro é
exclusivo no iCalendar; `data_fim` já vem corrigida, senão todo prazo
apareceria valendo um dia a mais.)

**A vigência das normas.** O índice "Regimento e Resoluções" separa
"Resoluções em vigor" de "revogadas", e é só daí que a vigência sai. As
páginas individuais têm a íntegra do texto mas não sabem se a norma ainda
vale. As duas fontes se fundem pela chave `resolucao-01-2024`, campo a campo
(`Dataset.add(..., merge=True)`) — a fusão por "linha mais rica" apagaria
`vigente`, que é exatamente o dado que impede responder com norma revogada.
`vigente` também é escrita **dentro do texto vetorizado**: a coluna protege o
`WHERE`, mas só o texto protege a síntese. Use `vw_ppgc_normativo_vigente`.

**Requisitos em coluna, não em prosa.** "Até quando comprovo proficiência em
inglês?" tem resposta diferente no mestrado (3ª matrícula) e no doutorado (5ª).
Em `ppgc_requisito` isso é `WHERE nivel = 'mestrado'`; em texto corrido, seria
o modelo escolhendo entre dois números parecidos.

**Documentos tipados pela seção da página.** Numa página de seleção, o mesmo
tipo de PDF aparece sob "Edital", "Outros Formulários" e "Processo Seletivo".
O título da seção é o sinal mais forte que existe — só ele distingue o
resultado final do formulário de autodeclaração.

## SQL ou semântico?

| A pergunta pede… | Rota | Onde |
|---|---|---|
| lista, contagem, "as últimas N", filtro por data | **SQL** | `vw_port_noticia`, `vw_computacao_noticia` |
| "qual o edital de mestrado de 2026?" | **SQL** | `vw_ppgc_edital`, `vw_ppgc_edital_mais_recente` |
| prazo, requisito, feriado, data de matrícula | **SQL** | `vw_ppgc_requisito`, `vw_ppgc_agenda`, `vw_port_calendario` |
| "como faço para…", procedimento | **Semântico** | `emb_port_faq`, `emb_ppgc_faq` |
| tema, assunto, "algo ligado a…" | **Semântico** | `emb_port_post`, `emb_ppgc_pagina` |
| tema **e** fato | **Híbrido** | `emb_*` → chave → `JOIN` |

Toda tabela `emb_*` carrega a chave da entidade, então um acerto vetorial
sempre vira JOIN relacional e um filtro relacional sempre restringe a busca.

O `ORDER BY` precisa repetir **exatamente** a expressão do índice HNSW, senão
o planner faz seq scan:

```sql
ORDER BY embedding::halfvec(2048) <=> $1::halfvec(2048)
```

(2048 dims não cabem em HNSW sobre `vector`; o índice é sobre a projeção
`halfvec`, e a coluna guarda a precisão integral.)

### Consultas de exemplo

```sql
-- últimas notícias dos DOIS acervos
SELECT acervo, data_por_extenso, titulo, url
  FROM vw_computacao_noticia ORDER BY data_publicacao DESC LIMIT 10;

-- edital de ingresso mais recente para mestrado
SELECT periodo_letivo, numero_oficial, titulo, url_edital_pdf
  FROM vw_ppgc_edital
 WHERE tipo = 'ingresso_regular' AND nivel IN ('mestrado', 'ambos')
 ORDER BY ano DESC, semestre DESC LIMIT 1;

-- o resultado da seleção 2026/1 já saiu?
SELECT documento, url FROM vw_ppgc_edital_documento
 WHERE ano = 2026 AND semestre = 1 AND tipo = 'ingresso_regular'
   AND tipo_documento = 'resultado';

-- prazos do doutorado
SELECT requisito, prazo FROM vw_ppgc_requisito WHERE nivel = 'doutorado';

-- feriados de setembro de 2022
SELECT data, dia_semana, descricao FROM vw_port_calendario
 WHERE ano = 2022 AND mes = 9 AND tipo ILIKE '%Feriado%';

-- ponte com o dataset institucional
SELECT d.nome, s.titulacao, s.cargo
  FROM ppgc_docente d JOIN servidor s ON s.servidor_id = d.servidor_id;
```

## Integração com a aplicação

Os dois acervos já estão ligados ao pipeline de RAG:

| Onde | O que mudou |
|---|---|
| `aplicacao/busca_semantica.py` | 11 fontes novas em `FONTES`, cada uma com `acervo` (`institucional` \| `portal` \| `ppgc`), `join_contexto` e fallback léxico. `buscar(acervos=[...])` restringe a busca; `descrever_fontes()` agrupa por acervo no prompt do roteador |
| `aplicacao/text_to_sql.py` | `SCHEMA_CARD` ganhou as views e tabelas `ppgc_*`/`port_*`; 8 regras de negócio novas (vigência de norma, nível do edital, prazo por nível, escolha do acervo) e 10 exemplos few-shot verificados |
| `aplicacao/pipeline_computacao.py` | roteador conhece os três acervos; `_SINAIS_PROCEDIMENTO` e `_SINAIS_SQL_FORTE` separam "COMO comprovo proficiência" (semântico) de "ATÉ QUANDO comprovo" (SQL) |
| `aplicacao/eval_ppgc.py` | testset: 19 perguntas só-SQL, 14 só-semânticas e 14 de roteamento |

```bash
python eval_ppgc.py --listar             # o testset comentado
python eval_ppgc.py --sql --referencia   # valida o schema, sem custo de API
python eval_ppgc.py --tudo               # as três suítes
```

Um filtro por acervo não é otimização: são 16 fontes vetoriais, cada busca sem
filtro é uma consulta ANN por fonte, e os 51 editais do PPGC disputariam o
top-8 com 599 notícias de graduação.

## A próxima etapa: os PDFs dos editais

`ppgc_documento` já é a **fila de trabalho**: 372 arquivos com URL, tipo, data
e edital de origem, e quatro colunas vazias esperando a extração —
`texto_extraido`, `n_paginas`, `sha256`, `extraido_em`.

O fluxo previsto:

1. para cada `ppgc_documento` com `texto_extraido IS NULL`, baixar, extrair o
   texto, gravar as quatro colunas e as linhas de `ppgc_documento_chunk`
   (`pagina_pdf` permite citar "página 3 do Edital 253/2025");
2. `python load_ppgc.py --input ppgc.json --only-embeddings` → `emb_ppgc_documento`
   passa a existir.

**Nenhuma mudança de schema é necessária.** E o loader já protege o
investimento: `ppgc_documento` e `ppgc_documento_chunk` ficam de fora do
`TRUNCATE`, os metadados entram por UPSERT e as colunas de extração não são
sobrescritas. Um recrawl atualiza os vínculos edital↔arquivo sem custar a
reextração de centenas de PDFs. (`--reset-documentos` força o descarte, quando
for isso mesmo que se quer.)

O `sha256` existe para que a reingestão pule arquivo que não mudou — o mesmo
formulário de autodeclaração é referenciado por praticamente toda seleção.
