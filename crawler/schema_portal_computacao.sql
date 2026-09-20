-- =============================================================================
-- Schema Relacional + Vetorial — PORTAL da Computação-UFPel
-- Fonte: https://wp.ufpel.edu.br/computacao/   (WordPress, REST API)
-- =============================================================================
-- Executar:  psql "$DATABASE_URL" -f schema_portal_computacao.sql
--        ou: python load_portal_computacao.py --input portal_computacao.json --schema
--
-- Este schema é INDEPENDENTE dos outros dois do projeto e pode conviver com
-- eles no mesmo banco:
--
--   schema_computacao.sql          institucional.ufpel.edu.br — currículo,
--                                  disciplinas, servidores, turmas, projetos
--   schema_portal_computacao.sql   ESTE — notícias e páginas do portal do curso
--   schema_ppgc.sql                pós-graduação: editais, normativos, FAQ,
--                                  calendário, linhas de pesquisa
--
-- O prefixo `port_` existe justamente para isso: nenhum nome colide, e o
-- roteador do RAG escolhe o acervo escolhendo o prefixo da tabela.
--
-- Duas camadas sobre a mesma coleta
-- ---------------------------------
--   1. RELACIONAL — perguntas factuais, filtros e agregações. O LLM monta SQL.
--        "quantas notícias saíram em 2025?"
--        "qual a última notícia sobre a Semana Acadêmica?"
--        "quando começa o período de matrícula segundo o calendário de 2022?"
--
--   2. VETORIAL (emb_port_*) — perguntas semânticas e abertas.
--        "tem alguma notícia sobre robótica?"
--        "como faço para trancar uma disciplina?"
--
-- Regra de ouro (a mesma de schema_computacao.sql): TODA tabela emb_* carrega
-- a chave da entidade. Um acerto vetorial vira JOIN relacional, e um filtro
-- relacional restringe a busca vetorial.
--
-- Convenções
--   * Chaves naturais do WordPress (post_id, pagina_id, categoria_id) —
--     carga idempotente e joins legíveis para o LLM.
--   * Ausente = NULL, nunca string sentinela. "Não informado" é decisão da
--     camada de síntese, não do dado.
--   * A DATA é cidadã de primeira classe: `data_publicacao` DATE + `ano`,
--     `mes` e `semestre` derivados. O enunciado pede que a resposta sempre
--     saiba quando o conteúdo foi publicado, e `data_por_extenso` existe para
--     entrar no texto vetorizado ("15 de julho de 2026") — "2026-07-15" não
--     ativa nada num embedding.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;


-- #############################################################################
-- DOMÍNIO 1 — ESTRUTURA DO SITE
-- #############################################################################

-- -----------------------------------------------------------------------------
-- port_crawl_meta — proveniência da coleta (quando, de onde, quanto)
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS port_crawl_meta CASCADE;
CREATE TABLE port_crawl_meta (
    chave   TEXT PRIMARY KEY,
    valor   TEXT
);

-- -----------------------------------------------------------------------------
-- port_secao — as 8 seções do menu principal
--
-- É o filtro mais barato do dataset: "Notícias", "Sobre a Computação-UFPel",
-- "Graduação", "Pós-Graduação", "Pesquisa", "Ensino", "Extensão" e "Calendário
-- Acadêmico". Tanto páginas quanto posts carregam `secao_slug`.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS port_secao CASCADE;
CREATE TABLE port_secao (
    secao_slug  TEXT PRIMARY KEY,   -- noticias | sobre | graduacao | pos-graduacao
                                    -- pesquisa | ensino | extensao | calendario
    nome        TEXT NOT NULL,
    ordem       INTEGER,            -- posição no menu do site
    url         TEXT
);

-- -----------------------------------------------------------------------------
-- port_menu_item — o menu do cabeçalho, com hierarquia
--
-- Guardado porque o menu e a árvore de páginas DISCORDAM, e cada um responde
-- uma coisa. "Liga Acadêmica de Robótica" aparece sob Graduação e sob Ensino;
-- "Ciência da Computação" é filha de Graduação no menu, mas página raiz na
-- árvore. Para "o que tem no menu Extensão?", a fonte certa é esta tabela.
--
-- `pagina_id` NÃO tem FK de propósito: 7 itens apontam para páginas do PPGC,
-- que vivem em `ppgc_pagina` (outro schema). `pagina_dataset` diz onde
-- procurar — 'portal' ou 'ppgc'.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS port_menu_item CASCADE;
CREATE TABLE port_menu_item (
    item_id         BIGINT PRIMARY KEY,       -- id do menu-item no WordPress
    secao_slug      TEXT REFERENCES port_secao(secao_slug) ON DELETE SET NULL,
    parent_item_id  BIGINT,
    ordem           INTEGER,
    nivel           INTEGER,                  -- 0 = item de topo
    titulo          TEXT NOT NULL,
    url             TEXT NOT NULL,
    alvo_tipo       TEXT,                     -- pagina | interno | externo
    pagina_id       BIGINT,                   -- sem FK: ver comentário acima
    pagina_dataset  TEXT                      -- portal | ppgc | NULL
);

-- -----------------------------------------------------------------------------
-- port_autor / port_categoria / port_tag — taxonomias do WordPress
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS port_autor CASCADE;
CREATE TABLE port_autor (
    autor_id    BIGINT PRIMARY KEY,
    nome        TEXT NOT NULL,
    slug        TEXT,
    descricao   TEXT,
    url         TEXT
);

DROP TABLE IF EXISTS port_categoria CASCADE;
CREATE TABLE port_categoria (
    categoria_id  BIGINT PRIMARY KEY,
    slug          TEXT NOT NULL,   -- ccomp | ecomp | noticia | ppgc | todos | …
    nome          TEXT NOT NULL,
    descricao     TEXT,
    total_posts   INTEGER,         -- contagem publicada pela própria API
    parent_id     BIGINT
);

DROP TABLE IF EXISTS port_tag CASCADE;
CREATE TABLE port_tag (
    tag_id       BIGINT PRIMARY KEY,
    slug         TEXT NOT NULL,
    nome         TEXT NOT NULL,
    total_posts  INTEGER
);


-- #############################################################################
-- DOMÍNIO 2 — PÁGINAS
-- #############################################################################

-- -----------------------------------------------------------------------------
-- port_pagina — conteúdo estável do site (uma linha por página publicada)
--
-- `caminho` é a hierarquia reconstruída por `parent`
-- ("sobre-a-computacao-ufpel/laboratorios"), não o permalink: o WordPress
-- reescreve permalinks e várias páginas ficaram com URL na raiz mesmo tendo
-- pai. Para prefixo hierárquico, use `caminho`; para citar a fonte, `url`.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS port_pagina CASCADE;
CREATE TABLE port_pagina (
    pagina_id         BIGINT PRIMARY KEY,
    slug              TEXT NOT NULL,
    caminho           TEXT NOT NULL,          -- hierarquia real, sem barra inicial
    titulo            TEXT NOT NULL,
    url               TEXT NOT NULL,
    pagina_pai_id     BIGINT REFERENCES port_pagina(pagina_id) ON DELETE SET NULL,
    caminho_pai       TEXT,
    nivel             INTEGER,                -- profundidade na árvore
    secao_slug        TEXT REFERENCES port_secao(secao_slug) ON DELETE SET NULL,
    idioma            TEXT,                   -- pt | en  (o site usa Polylang)
    ordem_menu        INTEGER,
    data_publicacao   DATE,
    data_modificacao  DATE,
    resumo            TEXT,
    texto             TEXT,                   -- página inteira, texto limpo
    n_palavras        INTEGER,
    template          TEXT
);

-- -----------------------------------------------------------------------------
-- port_pagina_secao — a página quebrada nos seus títulos
--
-- Esta é a granularidade que a busca semântica usa. Uma página de 40 mil
-- caracteres vetorizada inteira dilui o sinal: "como peço segunda chamada?"
-- pontuaria igual a estágio e a e-mail institucional, porque tudo está no
-- mesmo vetor. Uma linha por seção resolve — e `ancora` permite citar
-- `.../faq-da-computacao/#matricula` em vez de mandar o usuário procurar.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS port_pagina_secao CASCADE;
CREATE TABLE port_pagina_secao (
    pagina_id     BIGINT NOT NULL REFERENCES port_pagina(pagina_id) ON DELETE CASCADE,
    ordem         INTEGER NOT NULL,
    titulo_secao  TEXT,                       -- NULL = página sem títulos
    nivel_titulo  INTEGER,                    -- 1..6 (h1..h6); 4 = negrito-título
    ancora        TEXT,                       -- fragmento da URL (#…)
    texto         TEXT NOT NULL,
    n_palavras    INTEGER,
    url           TEXT,                       -- url + #ancora, pronta para citar
    PRIMARY KEY (pagina_id, ordem)
);


-- #############################################################################
-- DOMÍNIO 3 — NOTÍCIAS
-- #############################################################################

-- -----------------------------------------------------------------------------
-- port_post — as notícias do portal
--
-- Contém os posts que NÃO são do PPGC; os do Programa estão em `ppgc_post`
-- (schema_ppgc.sql). A view `vw_computacao_noticia` daquele schema une os dois
-- para quando a pergunta for simplesmente "quais as últimas notícias?".
--
-- Por que tanta coluna de data: o enunciado exige informar ao usuário quando o
-- conteúdo foi publicado. `data_publicacao` responde ordenação e filtro;
-- `ano`/`mes`/`semestre` respondem agregação sem `EXTRACT`; e
-- `data_por_extenso` é o que entra no texto vetorizado, porque "julho de 2026"
-- casa semanticamente e "2026-07-15" não.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS port_post CASCADE;
CREATE TABLE port_post (
    post_id              BIGINT PRIMARY KEY,
    slug                 TEXT NOT NULL,
    titulo               TEXT NOT NULL,
    url                  TEXT NOT NULL,
    secao_slug           TEXT REFERENCES port_secao(secao_slug) ON DELETE SET NULL,
    data_publicacao      DATE,
    publicado_em         TIMESTAMP,           -- data + hora (fuso do site)
    ano                  INTEGER,
    mes                  INTEGER,
    semestre             SMALLINT,            -- 1 | 2
    data_por_extenso     TEXT,                -- "15 de julho de 2026"
    data_modificacao     DATE,
    autor_id             BIGINT REFERENCES port_autor(autor_id) ON DELETE SET NULL,
    escopo_curso         TEXT,                -- ccomp | ecomp | ambos | geral
    resumo               TEXT,
    texto                TEXT,
    n_palavras           INTEGER,
    imagem_destaque_url  TEXT,
    n_documentos         INTEGER,             -- PDFs/DOCs anexados ao post
    anos_citados         INTEGER[],           -- anos mencionados no texto
    url_curta            TEXT
);

DROP TABLE IF EXISTS port_post_categoria CASCADE;
CREATE TABLE port_post_categoria (
    post_id       BIGINT NOT NULL REFERENCES port_post(post_id) ON DELETE CASCADE,
    categoria_id  BIGINT NOT NULL REFERENCES port_categoria(categoria_id) ON DELETE CASCADE,
    PRIMARY KEY (post_id, categoria_id)
);

DROP TABLE IF EXISTS port_post_tag CASCADE;
CREATE TABLE port_post_tag (
    post_id  BIGINT NOT NULL REFERENCES port_post(post_id) ON DELETE CASCADE,
    tag_id   BIGINT NOT NULL REFERENCES port_tag(tag_id) ON DELETE CASCADE,
    PRIMARY KEY (post_id, tag_id)
);


-- #############################################################################
-- DOMÍNIO 4 — LINKS E ANEXOS
-- #############################################################################

-- -----------------------------------------------------------------------------
-- port_link — todos os links do conteúdo, para rastreabilidade
--
-- Polimórfica de propósito (`origem_tipo` + `origem_id`), sem FK: a mesma
-- tabela serve posts e páginas, e uma FK exigiria duas tabelas quase iguais.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS port_link CASCADE;
CREATE TABLE port_link (
    origem_tipo  TEXT NOT NULL,               -- post | pagina
    origem_id    BIGINT NOT NULL,
    ordem        INTEGER NOT NULL,
    url          TEXT NOT NULL,
    texto        TEXT,
    host         TEXT,
    tipo         TEXT,                        -- documento | planilha | imagem
                                              -- email | interno | externo
    extensao     TEXT,
    PRIMARY KEY (origem_tipo, origem_id, ordem)
);

-- -----------------------------------------------------------------------------
-- port_documento — os anexos (PDF/DOC/planilha) referenciados no conteúdo
--
-- Uma linha por ARQUIVO. O rótulo do link ("Formulário de matrícula") é o
-- melhor título que o arquivo tem, e a biblioteca de mídia do WordPress
-- fornece mime e data de upload quando o arquivo é hospedado no próprio site.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS port_documento CASCADE;
CREATE TABLE port_documento (
    url             TEXT PRIMARY KEY,
    nome_arquivo    TEXT,
    titulo          TEXT,
    extensao        TEXT,
    mime            TEXT,
    media_id        BIGINT,                   -- id na biblioteca de mídia
    data_upload     DATE,
    origem_tipo     TEXT,                     -- post | pagina
    origem_id       BIGINT,
    origem_titulo   TEXT,
    origem_url      TEXT,
    texto_link      TEXT
);


-- #############################################################################
-- DOMÍNIO 5 — CONTEÚDO NORMALIZADO A PARTIR DE PÁGINAS ESPECÍFICAS
-- #############################################################################

-- -----------------------------------------------------------------------------
-- port_faq — o FAQ da Computação, pergunta a pergunta
--
-- A página tem ~43 mil caracteres. Em linha por Q&A, cada pergunta vira um
-- embedding próprio e a recuperação passa a ser cirúrgica; `ancora` leva o
-- usuário ao ponto exato da página.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS port_faq CASCADE;
CREATE TABLE port_faq (
    faq_id     BIGSERIAL PRIMARY KEY,
    pagina_id  BIGINT NOT NULL REFERENCES port_pagina(pagina_id) ON DELETE CASCADE,
    ordem      INTEGER NOT NULL,
    secao      TEXT,                          -- "MATRÍCULA", "TCC", …
    ancora     TEXT,
    pergunta   TEXT NOT NULL,
    resposta   TEXT NOT NULL,
    url        TEXT,
    UNIQUE (pagina_id, ordem)
);

-- -----------------------------------------------------------------------------
-- port_pessoa — docentes e técnico-administrativos listados no portal
--
-- `servidor_id` é a MESMA chave de `servidor(servidor_id)` em
-- schema_computacao.sql — os links da página apontam para
-- `institucional.ufpel.edu.br/servidores/id/NNNNN`. Quem tiver os dois
-- datasets carregados junta o nome daqui com titulação, projetos e
-- disciplinas ministradas de lá. Sem FK porque o outro schema é opcional.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS port_pessoa CASCADE;
CREATE TABLE port_pessoa (
    pessoa_id    BIGSERIAL PRIMARY KEY,
    pagina_id    BIGINT NOT NULL REFERENCES port_pagina(pagina_id) ON DELETE CASCADE,
    nome         TEXT NOT NULL,
    idioma       TEXT,
    categoria    TEXT,                        -- docente | tecnico_administrativo
                                              -- | aposentado
    setor        TEXT,                        -- "Secretaria PPGC", "NRC", …
    servidor_id  TEXT,                        -- ver comentário acima
    lattes_url   TEXT,
    url_perfil   TEXT,
    url_origem   TEXT,
    UNIQUE (pagina_id, nome)
);

-- -----------------------------------------------------------------------------
-- port_grupo_pesquisa — grupos listados em Pesquisa › Grupos de Pesquisa
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS port_grupo_pesquisa CASCADE;
CREATE TABLE port_grupo_pesquisa (
    grupo_id    BIGSERIAL PRIMARY KEY,
    nome        TEXT NOT NULL UNIQUE,
    idioma      TEXT,
    sigla       TEXT,                         -- GACI, GAIA, LUPS, ViTech…
    url         TEXT,
    pagina_id   BIGINT REFERENCES port_pagina(pagina_id) ON DELETE SET NULL,
    url_origem  TEXT
);

-- -----------------------------------------------------------------------------
-- port_calendario_evento — o calendário acadêmico, dia a dia
--
-- Extraído do widget do Cobalto embutido nas páginas de calendário. Vale a
-- normalização porque a pergunta típica é factual e datada — "quando começa o
-- período de matrícula?", "quais os feriados de setembro?". Com `data` em
-- coluna DATE isso é um WHERE; em texto corrido, seria uma aposta.
--
-- `tipo` é o nome do calendário de origem ("Feriados e pontos facultativos",
-- "Calendário Acadêmico - Cursos semestrais"), que vem do `title` da bolinha
-- colorida da legenda.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS port_calendario_evento CASCADE;
CREATE TABLE port_calendario_evento (
    evento_id   BIGSERIAL PRIMARY KEY,
    pagina_id   BIGINT NOT NULL REFERENCES port_pagina(pagina_id) ON DELETE CASCADE,
    data        DATE NOT NULL,
    ano         INTEGER,
    mes         INTEGER,
    dia_semana  TEXT,
    descricao   TEXT NOT NULL,
    tipo        TEXT,
    url         TEXT,
    UNIQUE (pagina_id, data, descricao)
);


-- #############################################################################
-- CAMADA VETORIAL
-- #############################################################################
-- Uma linha = um trecho vetorizado, SEMPRE com a chave da entidade de volta.
-- `escopo` identifica a faceta ('titulo_resumo', 'corpo#2', 'secao#0'…) e é o
-- que permite filtrar antes da busca. `metadata` carrega os campos que o
-- roteador usa para pré-filtrar sem JOIN (ano, seção, categoria).

-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS emb_port_post CASCADE;
CREATE TABLE emb_port_post (
    emb_id         BIGSERIAL PRIMARY KEY,
    post_id        BIGINT NOT NULL REFERENCES port_post(post_id) ON DELETE CASCADE,
    escopo         TEXT NOT NULL,
    titulo         TEXT NOT NULL,
    url            TEXT,
    texto          TEXT NOT NULL,
    embedding      vector(2048) NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{}',
    atualizado_em  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (post_id, escopo)
);

-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS emb_port_pagina CASCADE;
CREATE TABLE emb_port_pagina (
    emb_id         BIGSERIAL PRIMARY KEY,
    pagina_id      BIGINT NOT NULL REFERENCES port_pagina(pagina_id) ON DELETE CASCADE,
    escopo         TEXT NOT NULL,
    titulo         TEXT NOT NULL,
    url            TEXT,
    texto          TEXT NOT NULL,
    embedding      vector(2048) NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{}',
    atualizado_em  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (pagina_id, escopo)
);

-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS emb_port_faq CASCADE;
CREATE TABLE emb_port_faq (
    emb_id         BIGSERIAL PRIMARY KEY,
    faq_id         BIGINT NOT NULL REFERENCES port_faq(faq_id) ON DELETE CASCADE,
    escopo         TEXT NOT NULL,
    titulo         TEXT NOT NULL,
    url            TEXT,
    texto          TEXT NOT NULL,
    embedding      vector(2048) NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{}',
    atualizado_em  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (faq_id, escopo)
);


-- #############################################################################
-- VIEWS — as formas prontas que o LLM deve preferir ao montar SQL
-- #############################################################################

-- Notícia com autor, categorias e tags já agregados. Evita que o LLM tenha de
-- descobrir sozinho as duas tabelas de junção.
CREATE OR REPLACE VIEW vw_port_noticia AS
SELECT p.post_id,
       p.titulo,
       p.url,
       p.data_publicacao,
       p.data_por_extenso,
       p.ano,
       p.mes,
       p.semestre,
       p.escopo_curso,
       p.resumo,
       p.n_palavras,
       p.n_documentos,
       a.nome                                         AS autor,
       COALESCE(c.categorias, ARRAY[]::TEXT[])        AS categorias,
       COALESCE(t.tags, ARRAY[]::TEXT[])              AS tags
  FROM port_post p
  LEFT JOIN port_autor a ON a.autor_id = p.autor_id
  LEFT JOIN LATERAL (
        SELECT array_agg(cat.nome ORDER BY cat.nome) AS categorias
          FROM port_post_categoria pc
          JOIN port_categoria cat ON cat.categoria_id = pc.categoria_id
         WHERE pc.post_id = p.post_id
  ) c ON TRUE
  LEFT JOIN LATERAL (
        SELECT array_agg(tg.nome ORDER BY tg.nome) AS tags
          FROM port_post_tag pt
          JOIN port_tag tg ON tg.tag_id = pt.tag_id
         WHERE pt.post_id = p.post_id
  ) t ON TRUE;

-- Página com a seção do menu e o caminho hierárquico legível.
CREATE OR REPLACE VIEW vw_port_pagina AS
SELECT pg.pagina_id,
       pg.titulo,
       pg.url,
       pg.caminho,
       pg.nivel,
       pg.idioma,
       pg.data_modificacao,
       pg.n_palavras,
       s.nome  AS secao,
       pg.secao_slug,
       pg.resumo
  FROM port_pagina pg
  LEFT JOIN port_secao s ON s.secao_slug = pg.secao_slug;

-- Menu inteiro com o caminho de títulos ("Graduação › Semana Acadêmica").
CREATE OR REPLACE VIEW vw_port_menu AS
WITH RECURSIVE arvore AS (
    SELECT item_id, parent_item_id, titulo, url, secao_slug, nivel, ordem,
           pagina_id, pagina_dataset, titulo::TEXT AS caminho_titulos
      FROM port_menu_item
     WHERE parent_item_id IS NULL
    UNION ALL
    SELECT m.item_id, m.parent_item_id, m.titulo, m.url, m.secao_slug, m.nivel,
           m.ordem, m.pagina_id, m.pagina_dataset,
           a.caminho_titulos || ' › ' || m.titulo
      FROM port_menu_item m
      JOIN arvore a ON a.item_id = m.parent_item_id
)
SELECT * FROM arvore;

-- Calendário pronto para responder "o que acontece em X?".
CREATE OR REPLACE VIEW vw_port_calendario AS
SELECT e.evento_id,
       e.data,
       e.ano,
       e.mes,
       e.dia_semana,
       e.descricao,
       e.tipo,
       e.url,
       pg.titulo AS calendario
  FROM port_calendario_evento e
  JOIN port_pagina pg ON pg.pagina_id = e.pagina_id;

-- FAQ com a página de origem.
CREATE OR REPLACE VIEW vw_port_faq AS
SELECT f.faq_id, f.secao, f.pergunta, f.resposta, f.url, pg.titulo AS pagina
  FROM port_faq f
  JOIN port_pagina pg ON pg.pagina_id = f.pagina_id;

-- Anexos com o conteúdo que os cita.
CREATE OR REPLACE VIEW vw_port_documento AS
SELECT d.url, d.titulo, d.extensao, d.mime, d.data_upload,
       d.origem_tipo, d.origem_titulo, d.origem_url
  FROM port_documento d;


-- #############################################################################
-- ÍNDICES
-- #############################################################################

-- ── Relacional: os filtros que aparecem em quase toda pergunta ──────────────
CREATE INDEX IF NOT EXISTS idx_port_post_data     ON port_post (data_publicacao DESC);
CREATE INDEX IF NOT EXISTS idx_port_post_ano      ON port_post (ano);
CREATE INDEX IF NOT EXISTS idx_port_post_secao    ON port_post (secao_slug);
CREATE INDEX IF NOT EXISTS idx_port_post_escopo   ON port_post (escopo_curso);
CREATE INDEX IF NOT EXISTS idx_port_pagina_secao  ON port_pagina (secao_slug);
CREATE INDEX IF NOT EXISTS idx_port_pagina_caminho ON port_pagina (caminho);
CREATE INDEX IF NOT EXISTS idx_port_pagina_idioma ON port_pagina (idioma);
CREATE INDEX IF NOT EXISTS idx_port_cal_data      ON port_calendario_evento (data);
CREATE INDEX IF NOT EXISTS idx_port_cal_ano_mes   ON port_calendario_evento (ano, mes);
CREATE INDEX IF NOT EXISTS idx_port_link_origem   ON port_link (origem_tipo, origem_id);
CREATE INDEX IF NOT EXISTS idx_port_doc_origem    ON port_documento (origem_tipo, origem_id);

-- ── Trigram: casar nome próprio que veio do LLM ("prof. Ana Marilza") ───────
CREATE INDEX IF NOT EXISTS idx_port_post_titulo_trgm
    ON port_post USING gin (titulo gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_port_pagina_titulo_trgm
    ON port_pagina USING gin (titulo gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_port_pessoa_nome_trgm
    ON port_pessoa USING gin (nome gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_port_grupo_nome_trgm
    ON port_grupo_pesquisa USING gin (nome gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_port_cal_desc_trgm
    ON port_calendario_evento USING gin (descricao gin_trgm_ops);

-- ── Full-text em português: filtro léxico barato antes do vetorial ──────────
CREATE INDEX IF NOT EXISTS idx_port_post_fts
    ON port_post USING gin (to_tsvector('portuguese',
        COALESCE(titulo, '') || ' ' || COALESCE(texto, '')));
CREATE INDEX IF NOT EXISTS idx_port_pagina_secao_fts
    ON port_pagina_secao USING gin (to_tsvector('portuguese', texto));
CREATE INDEX IF NOT EXISTS idx_port_faq_fts
    ON port_faq USING gin (to_tsvector('portuguese', pergunta || ' ' || resposta));

-- ── HNSW cosseno nas tabelas vetoriais ──────────────────────────────────────
-- O nemotron-3-embed-1b produz 2048 dimensões e o pgvector NÃO indexa `vector`
-- acima de 2000 dims em HNSW. A saída canônica é indexar a projeção em
-- `halfvec` (até 4000 dims); a coluna segue `vector(2048)`, com precisão
-- integral no armazenamento — só o índice é aproximado.
--
-- Consequência para quem consulta: o ORDER BY precisa repetir EXATAMENTE a
-- expressão do índice, senão o planner ignora o HNSW e faz seq scan:
--     ORDER BY embedding::halfvec(2048) <=> $1::halfvec(2048)
CREATE INDEX IF NOT EXISTS idx_emb_port_post_hnsw
    ON emb_port_post   USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 128);
CREATE INDEX IF NOT EXISTS idx_emb_port_pagina_hnsw
    ON emb_port_pagina USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 128);
CREATE INDEX IF NOT EXISTS idx_emb_port_faq_hnsw
    ON emb_port_faq    USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- Pré-filtros baratos antes do vetorial
CREATE INDEX IF NOT EXISTS idx_emb_port_post_escopo   ON emb_port_post   (escopo);
CREATE INDEX IF NOT EXISTS idx_emb_port_pagina_escopo ON emb_port_pagina (escopo);
CREATE INDEX IF NOT EXISTS idx_emb_port_post_meta
    ON emb_port_post   USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_emb_port_pagina_meta
    ON emb_port_pagina USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_emb_port_faq_meta
    ON emb_port_faq    USING gin (metadata jsonb_path_ops);
