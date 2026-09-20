-- =============================================================================
-- Schema Relacional + Vetorial — PPGC (Pós-Graduação em Computação, UFPel)
-- Fonte: https://wp.ufpel.edu.br/computacao/ppgc/  (+ editais soltos na raiz)
-- =============================================================================
-- Executar:  psql "$DATABASE_URL" -f schema_ppgc.sql
--        ou: python load_ppgc.py --input ppgc.json --schema
--
-- Acervo SEPARADO, por decisão de projeto
-- ---------------------------------------
-- Este schema é acionado quando o roteador conclui que a resposta NÃO está no
-- portal do curso. Manter as tabelas separadas transforma essa decisão num
-- filtro de tabela — e mantém o índice vetorial do Programa com sinal denso
-- de pós-graduação, sem centenas de notícias de graduação disputando
-- vizinhança com "prazo de proficiência em inglês".
--
-- Convive no mesmo banco com `schema_computacao.sql` (institucional) e
-- `schema_portal_computacao.sql` (portal). Prefixo `ppgc_`, sem colisões.
--
-- O que este acervo responde que o portal não responde
-- ----------------------------------------------------
--   editais       tipo, nível, ano/semestre, número oficial, e a lista de
--                 PDFs de cada um já tipada (edital, retificação, resultado…)
--   normativos    regimento, resoluções e portarias — COM a vigência, que é o
--                 que impede responder com resolução revogada
--   requisitos    créditos, proficiência, prazos — separados por nível
--   calendário    430 eventos datados do calendário público do Programa
--   FAQ           o passo a passo de matrícula, defesa, bolsa, trancamento
--   linhas        as 5 linhas de pesquisa, com a descrição que o candidato lê
--
-- Os PDFs dos editais (etapa seguinte)
-- ------------------------------------
-- `ppgc_documento` já nasce com uma linha por arquivo e as colunas de extração
-- vazias (`texto_extraido`, `n_paginas`, `sha256`, `extraido_em`).
-- `ppgc_documento_chunk` recebe o texto fatiado e `emb_ppgc_documento`
-- vetoriza os chunks. Nada mais no schema precisa mudar quando os PDFs
-- entrarem.
--
-- IMPORTANTE: `load_ppgc.py` NÃO faz TRUNCATE em `ppgc_documento` nem em
-- `ppgc_documento_chunk` — atualiza os metadados por UPSERT e preserva o texto
-- extraído. Um recrawl não pode custar a reextração de 372 PDFs.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;


-- #############################################################################
-- DOMÍNIO 1 — PÁGINAS E NOTÍCIAS DO PROGRAMA
-- #############################################################################

DROP TABLE IF EXISTS ppgc_crawl_meta CASCADE;
CREATE TABLE ppgc_crawl_meta (
    chave  TEXT PRIMARY KEY,
    valor  TEXT
);

-- -----------------------------------------------------------------------------
-- ppgc_pagina — as páginas do Programa
--
-- `categoria` pré-classifica a página e é o filtro que o roteador usa antes
-- de qualquer busca: edital | normativo | faq | calendario | docentes |
-- linha_pesquisa | disciplinas | institucional.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS ppgc_pagina CASCADE;
CREATE TABLE ppgc_pagina (
    pagina_id         BIGINT PRIMARY KEY,
    slug              TEXT NOT NULL,
    caminho           TEXT NOT NULL,
    titulo            TEXT NOT NULL,
    url               TEXT NOT NULL,
    pagina_pai_id     BIGINT REFERENCES ppgc_pagina(pagina_id) ON DELETE SET NULL,
    caminho_pai       TEXT,
    nivel             INTEGER,
    categoria         TEXT,
    idioma            TEXT,                   -- pt | en
    data_publicacao   DATE,
    data_modificacao  DATE,
    resumo            TEXT,
    texto             TEXT,
    n_palavras        INTEGER,
    anos_citados      INTEGER[]
);

-- -----------------------------------------------------------------------------
-- ppgc_pagina_secao — a página quebrada nos títulos (granularidade do embedding)
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS ppgc_pagina_secao CASCADE;
CREATE TABLE ppgc_pagina_secao (
    pagina_id     BIGINT NOT NULL REFERENCES ppgc_pagina(pagina_id) ON DELETE CASCADE,
    ordem         INTEGER NOT NULL,
    titulo_secao  TEXT,
    nivel_titulo  INTEGER,
    ancora        TEXT,
    texto         TEXT NOT NULL,
    n_palavras    INTEGER,
    url           TEXT,
    PRIMARY KEY (pagina_id, ordem)
);

-- -----------------------------------------------------------------------------
-- ppgc_post — as notícias do Programa
--
-- `assunto` (edital | defesa | bolsa | matricula | avaliacao | evento | geral)
-- é um filtro grosso e barato: "quando é a próxima defesa?" não precisa
-- passar por busca vetorial em 304 posts.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS ppgc_post CASCADE;
CREATE TABLE ppgc_post (
    post_id           BIGINT PRIMARY KEY,
    slug              TEXT NOT NULL,
    titulo            TEXT NOT NULL,
    url               TEXT NOT NULL,
    data_publicacao   DATE,
    publicado_em      TIMESTAMP,
    ano               INTEGER,
    mes               INTEGER,
    semestre          SMALLINT,
    data_por_extenso  TEXT,                   -- vai no texto vetorizado
    data_modificacao  DATE,
    autor_nome        TEXT,
    assunto           TEXT,
    resumo            TEXT,
    texto             TEXT,
    n_palavras        INTEGER,
    anos_citados      INTEGER[],
    url_curta         TEXT
);

DROP TABLE IF EXISTS ppgc_post_tag CASCADE;
CREATE TABLE ppgc_post_tag (
    post_id   BIGINT NOT NULL REFERENCES ppgc_post(post_id) ON DELETE CASCADE,
    tag_slug  TEXT NOT NULL,
    PRIMARY KEY (post_id, tag_slug)
);


-- #############################################################################
-- DOMÍNIO 2 — DOCUMENTOS (e a fila de ingestão dos PDFs)
-- #############################################################################

-- -----------------------------------------------------------------------------
-- ppgc_documento — uma linha por ARQUIVO referenciado no site do Programa
--
-- Separada de `ppgc_edital_documento` de propósito. Aqui vive o arquivo e o
-- resultado da extração; lá vive o VÍNCULO entre um edital e um arquivo. A
-- separação importa por dois motivos concretos:
--
--   1. o mesmo PDF é referenciado por vários editais (os formulários de
--      autodeclaração aparecem em toda seleção) — extrair duas vezes seria
--      desperdício e duplicaria os chunks no índice vetorial;
--   2. um recrawl reconstrói os vínculos, mas NÃO pode apagar o texto
--      extraído. Por isso o loader dá TRUNCATE em `ppgc_edital_documento` e
--      UPSERT aqui.
--
-- As quatro últimas colunas são o contrato com a ingestão futura de PDFs.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS ppgc_documento CASCADE;
CREATE TABLE ppgc_documento (
    url             TEXT PRIMARY KEY,
    nome_arquivo    TEXT,
    titulo          TEXT,
    extensao        TEXT,
    mime            TEXT,
    media_id        BIGINT,
    data_upload     DATE,
    origem_tipo     TEXT,                     -- pagina | post
    origem_id       BIGINT,
    origem_titulo   TEXT,
    origem_url      TEXT,
    texto_link      TEXT,
    -- preenchidos pela ingestão de PDFs, não pelo crawler
    texto_extraido  TEXT,
    n_paginas       INTEGER,
    sha256          TEXT,                     -- pula reextração de arquivo igual
    extraido_em     TIMESTAMPTZ
);

-- -----------------------------------------------------------------------------
-- ppgc_documento_chunk — o PDF fatiado, pronto para virar embedding
--
-- Vazia após o crawl. `pagina_pdf` permite citar "página 3 do Edital 253/2025"
-- na resposta, que é a diferença entre uma citação verificável e um "consta no
-- edital".
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS ppgc_documento_chunk CASCADE;
CREATE TABLE ppgc_documento_chunk (
    chunk_id       BIGSERIAL PRIMARY KEY,
    url            TEXT NOT NULL REFERENCES ppgc_documento(url) ON DELETE CASCADE,
    ordem          INTEGER NOT NULL,
    pagina_pdf     INTEGER,
    texto          TEXT NOT NULL,
    n_caracteres   INTEGER,
    UNIQUE (url, ordem)
);


-- #############################################################################
-- DOMÍNIO 3 — EDITAIS
-- #############################################################################

-- -----------------------------------------------------------------------------
-- ppgc_edital — um edital por linha, normalizado
--
-- `edital_id` é textual e derivado da origem ("pagina-8842"): legível em SQL,
-- estável entre recargas e sem depender de sequence.
--
-- `tipo` e `nivel` são o que fazem a pergunta do candidato virar consulta:
-- "qual o edital de mestrado aberto agora?" é
--     WHERE tipo = 'ingresso_regular' AND nivel IN ('mestrado','ambos')
--     ORDER BY ano DESC, semestre DESC
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS ppgc_edital CASCADE;
CREATE TABLE ppgc_edital (
    edital_id         TEXT PRIMARY KEY,
    origem_tipo       TEXT,                   -- pagina
    origem_id         BIGINT,
    titulo            TEXT NOT NULL,
    slug              TEXT,
    url               TEXT NOT NULL,
    tipo              TEXT,                   -- ingresso_regular | ingresso_especial
                                              -- bolsa_classificacao | bolsa_sanduiche
                                              -- bolsa_posdoc | professor_visitante
                                              -- dinter | resultado | outro
    nivel             TEXT,                   -- mestrado | doutorado | ambos
                                              -- posdoc | docente
    ano               INTEGER,
    semestre          SMALLINT,
    periodo_letivo    TEXT,                   -- "2026/1"
    numero_oficial    TEXT,                   -- "253/2025" (numeração SEI/UFPel)
    indice_pai        TEXT,                   -- editais-de-ingresso | editais-de-bolsas
    data_publicacao   DATE,
    data_modificacao  DATE,
    resumo            TEXT,
    texto             TEXT
);

-- -----------------------------------------------------------------------------
-- ppgc_edital_documento — quais arquivos pertencem a qual edital, e como
--
-- `tipo_documento` sai do TÍTULO DA SEÇÃO da página ("Edital", "Outros
-- Formulários", "Processo Seletivo"), com o texto do link desempatando os
-- casos inequívocos (retificação, cronograma). Sem isso, "resultado final" e
-- "formulário de autodeclaração" seriam o mesmo tipo de coisa — e a resposta
-- a "qual o resultado da seleção?" apontaria para um formulário em branco.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS ppgc_edital_documento CASCADE;
CREATE TABLE ppgc_edital_documento (
    edital_id       TEXT NOT NULL REFERENCES ppgc_edital(edital_id) ON DELETE CASCADE,
    url             TEXT NOT NULL REFERENCES ppgc_documento(url) ON DELETE CASCADE,
    ordem           INTEGER,
    titulo          TEXT,
    secao           TEXT,                     -- bloco da página onde estava
    tipo_documento  TEXT,                     -- edital | retificacao | formulario
                                              -- resultado | cronograma | anexo | outro
    numero_oficial  TEXT,
    PRIMARY KEY (edital_id, url)
);


-- #############################################################################
-- DOMÍNIO 4 — NORMAS
-- #############################################################################

-- -----------------------------------------------------------------------------
-- ppgc_normativo — regimento, resoluções e portarias
--
-- `normativo_id` = "resolucao-01-2024", "portaria-04-2013", "regimento-2020".
-- É essa chave que FUNDE as duas fontes que o site tem para a mesma norma:
--   * o índice "Regimento e Resoluções", que traz o PDF e — decisivo — a
--     VIGÊNCIA, porque o link está sob "Resoluções em vigor" ou "revogadas";
--   * a página própria de algumas resoluções, que traz a íntegra do texto.
--
-- `vigente` NULL significa "não listado no índice atual" — nem vigente nem
-- explicitamente revogado. Responder norma com `vigente IS NULL` sem ressalva
-- é o erro mais caro deste acervo; use `vw_ppgc_normativo_vigente`.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS ppgc_normativo CASCADE;
CREATE TABLE ppgc_normativo (
    normativo_id  TEXT PRIMARY KEY,
    tipo          TEXT,                       -- resolucao | portaria | regimento
                                              -- planejamento_estrategico | outro
    numero        TEXT,                       -- "01", "07" (zero à esquerda)
    ano           INTEGER,
    titulo        TEXT,
    ementa        TEXT,
    vigente       BOOLEAN,                    -- NULL = não listado no índice
    url_pagina    TEXT,
    url_pdf       TEXT,
    media_id      BIGINT,
    pagina_id     BIGINT REFERENCES ppgc_pagina(pagina_id) ON DELETE SET NULL,
    texto         TEXT                        -- íntegra, quando tem página
);


-- #############################################################################
-- DOMÍNIO 5 — O PROGRAMA: PESQUISA, PESSOAS, REGRAS, AGENDA
-- #############################################################################

-- -----------------------------------------------------------------------------
-- ppgc_linha_pesquisa — as 5 linhas, com a descrição usada na captação
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS ppgc_linha_pesquisa CASCADE;
CREATE TABLE ppgc_linha_pesquisa (
    slug              TEXT PRIMARY KEY,
    nome              TEXT NOT NULL,
    descricao         TEXT,
    url               TEXT,
    pagina_id         BIGINT REFERENCES ppgc_pagina(pagina_id) ON DELETE SET NULL,
    pagina_indice_id  BIGINT REFERENCES ppgc_pagina(pagina_id) ON DELETE SET NULL
);

-- -----------------------------------------------------------------------------
-- ppgc_docente — corpo docente do Programa
--
-- `servidor_id` é a mesma chave de `servidor(servidor_id)` em
-- schema_computacao.sql. Sem FK porque aquele schema é opcional.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS ppgc_docente CASCADE;
CREATE TABLE ppgc_docente (
    docente_id   BIGSERIAL PRIMARY KEY,
    nome         TEXT NOT NULL UNIQUE,
    url_perfil   TEXT,
    lattes_url   TEXT,
    servidor_id  TEXT,
    vinculo      TEXT,                        -- permanente | dinter
    pagina_id    BIGINT REFERENCES ppgc_pagina(pagina_id) ON DELETE SET NULL,
    url          TEXT
);

-- -----------------------------------------------------------------------------
-- ppgc_faq — o FAQ de alunos, pergunta a pergunta
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS ppgc_faq CASCADE;
CREATE TABLE ppgc_faq (
    faq_id     BIGSERIAL PRIMARY KEY,
    pagina_id  BIGINT NOT NULL REFERENCES ppgc_pagina(pagina_id) ON DELETE CASCADE,
    ordem      INTEGER NOT NULL,
    secao      TEXT,                          -- "Mestrado e Doutorado", "Doutorado"
    ancora     TEXT,
    pergunta   TEXT NOT NULL,
    resposta   TEXT NOT NULL,
    url        TEXT,
    UNIQUE (pagina_id, ordem)
);

-- -----------------------------------------------------------------------------
-- ppgc_requisito — créditos, proficiência e prazos, por nível
--
-- A tabela com maior razão valor/tamanho do acervo. "Até quando preciso
-- comprovar proficiência em inglês?" tem resposta DIFERENTE no mestrado (3ª
-- matrícula) e no doutorado (5ª). Em coluna, é `WHERE nivel = 'mestrado'`;
-- em texto corrido, seria o modelo escolhendo entre dois números parecidos.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS ppgc_requisito CASCADE;
CREATE TABLE ppgc_requisito (
    requisito_id  BIGSERIAL PRIMARY KEY,
    nivel         TEXT NOT NULL,              -- mestrado | doutorado
    requisito     TEXT NOT NULL,
    prazo         TEXT,
    obrigatorio   BOOLEAN,
    pagina_id     BIGINT REFERENCES ppgc_pagina(pagina_id) ON DELETE SET NULL,
    url           TEXT,
    UNIQUE (nivel, requisito)
);

-- -----------------------------------------------------------------------------
-- ppgc_disciplina — disciplinas ofertadas, com ementa e responsável
--
-- A ementa é o que responde "tem alguma disciplina sobre visão computacional?"
-- e `aluno_especial` é o que responde "posso cursar sem ser aluno regular?" —
-- duas perguntas frequentes que o portal institucional não responde.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS ppgc_disciplina CASCADE;
CREATE TABLE ppgc_disciplina (
    disciplina_id   BIGSERIAL PRIMARY KEY,
    nome            TEXT NOT NULL,
    creditos        INTEGER,
    ano             INTEGER,
    semestre        SMALLINT,
    aluno_especial  BOOLEAN,
    responsavel     TEXT,
    ementa          TEXT,
    horario         TEXT,
    modalidade      TEXT,                     -- presencial | remoto | híbrida
    pagina_id       BIGINT REFERENCES ppgc_pagina(pagina_id) ON DELETE SET NULL,
    url             TEXT,
    UNIQUE (nome, ano, semestre)
);

-- -----------------------------------------------------------------------------
-- ppgc_calendario_evento — o calendário público do Programa (Google Agenda)
--
-- A página do calendário é só um `<iframe>` — nenhum texto. Os eventos vêm do
-- `.ics` público do mesmo calendário, que é o único lugar onde as datas
-- existem em forma consultável.
--
-- `data_fim` já vem CORRIGIDA: no iCalendar o `DTEND` de evento de dia inteiro
-- é exclusivo, e usá-lo cru faria todo prazo aparecer valendo um dia a mais.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS ppgc_calendario_evento CASCADE;
CREATE TABLE ppgc_calendario_evento (
    uid               TEXT PRIMARY KEY,       -- UID do iCalendar (estável)
    titulo            TEXT,
    descricao         TEXT,
    local             TEXT,
    data_inicio       DATE NOT NULL,
    data_fim          DATE,
    hora_inicio       TIME,                   -- NULL quando é dia inteiro
    dia_inteiro       BOOLEAN,
    ano               INTEGER,
    mes               INTEGER,
    semestre          SMALLINT,
    data_por_extenso  TEXT,
    situacao          TEXT,                   -- confirmed | cancelled | tentative
    url_calendario    TEXT,
    pagina_id         BIGINT REFERENCES ppgc_pagina(pagina_id) ON DELETE SET NULL,
    atualizado_em     TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- ppgc_defesa — defesas de dissertação e tese anunciadas no site
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS ppgc_defesa CASCADE;
CREATE TABLE ppgc_defesa (
    defesa_id  BIGSERIAL PRIMARY KEY,
    tipo       TEXT NOT NULL,                 -- dissertacao | tese
    discente   TEXT NOT NULL,
    data       DATE NOT NULL,
    ano        INTEGER,
    hora       TIME,
    local      TEXT,
    link       TEXT,
    pagina_id  BIGINT REFERENCES ppgc_pagina(pagina_id) ON DELETE SET NULL,
    url        TEXT,
    UNIQUE (tipo, discente, data)
);

-- -----------------------------------------------------------------------------
-- ppgc_link — todos os links do conteúdo, para rastreabilidade
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS ppgc_link CASCADE;
CREATE TABLE ppgc_link (
    origem_tipo  TEXT NOT NULL,               -- pagina | post
    origem_id    BIGINT NOT NULL,
    ordem        INTEGER NOT NULL,
    url          TEXT NOT NULL,
    texto        TEXT,
    host         TEXT,
    tipo         TEXT,
    extensao     TEXT,
    PRIMARY KEY (origem_tipo, origem_id, ordem)
);


-- #############################################################################
-- CAMADA VETORIAL
-- #############################################################################

DROP TABLE IF EXISTS emb_ppgc_pagina CASCADE;
CREATE TABLE emb_ppgc_pagina (
    emb_id         BIGSERIAL PRIMARY KEY,
    pagina_id      BIGINT NOT NULL REFERENCES ppgc_pagina(pagina_id) ON DELETE CASCADE,
    escopo         TEXT NOT NULL,
    titulo         TEXT NOT NULL,
    url            TEXT,
    texto          TEXT NOT NULL,
    embedding      vector(2048) NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{}',
    atualizado_em  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (pagina_id, escopo)
);

DROP TABLE IF EXISTS emb_ppgc_post CASCADE;
CREATE TABLE emb_ppgc_post (
    emb_id         BIGSERIAL PRIMARY KEY,
    post_id        BIGINT NOT NULL REFERENCES ppgc_post(post_id) ON DELETE CASCADE,
    escopo         TEXT NOT NULL,
    titulo         TEXT NOT NULL,
    url            TEXT,
    texto          TEXT NOT NULL,
    embedding      vector(2048) NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{}',
    atualizado_em  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (post_id, escopo)
);

DROP TABLE IF EXISTS emb_ppgc_edital CASCADE;
CREATE TABLE emb_ppgc_edital (
    emb_id         BIGSERIAL PRIMARY KEY,
    edital_id      TEXT NOT NULL REFERENCES ppgc_edital(edital_id) ON DELETE CASCADE,
    escopo         TEXT NOT NULL,
    titulo         TEXT NOT NULL,
    url            TEXT,
    texto          TEXT NOT NULL,
    embedding      vector(2048) NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{}',
    atualizado_em  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (edital_id, escopo)
);

DROP TABLE IF EXISTS emb_ppgc_normativo CASCADE;
CREATE TABLE emb_ppgc_normativo (
    emb_id         BIGSERIAL PRIMARY KEY,
    normativo_id   TEXT NOT NULL REFERENCES ppgc_normativo(normativo_id) ON DELETE CASCADE,
    escopo         TEXT NOT NULL,
    titulo         TEXT NOT NULL,
    url            TEXT,
    texto          TEXT NOT NULL,
    embedding      vector(2048) NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{}',
    atualizado_em  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (normativo_id, escopo)
);

DROP TABLE IF EXISTS emb_ppgc_faq CASCADE;
CREATE TABLE emb_ppgc_faq (
    emb_id         BIGSERIAL PRIMARY KEY,
    faq_id         BIGINT NOT NULL REFERENCES ppgc_faq(faq_id) ON DELETE CASCADE,
    escopo         TEXT NOT NULL,
    titulo         TEXT NOT NULL,
    url            TEXT,
    texto          TEXT NOT NULL,
    embedding      vector(2048) NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{}',
    atualizado_em  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (faq_id, escopo)
);

DROP TABLE IF EXISTS emb_ppgc_linha_pesquisa CASCADE;
CREATE TABLE emb_ppgc_linha_pesquisa (
    emb_id         BIGSERIAL PRIMARY KEY,
    linha_slug     TEXT NOT NULL REFERENCES ppgc_linha_pesquisa(slug) ON DELETE CASCADE,
    escopo         TEXT NOT NULL,
    titulo         TEXT NOT NULL,
    url            TEXT,
    texto          TEXT NOT NULL,
    embedding      vector(2048) NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{}',
    atualizado_em  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (linha_slug, escopo)
);

DROP TABLE IF EXISTS emb_ppgc_disciplina CASCADE;
CREATE TABLE emb_ppgc_disciplina (
    emb_id         BIGSERIAL PRIMARY KEY,
    disciplina_id  BIGINT NOT NULL REFERENCES ppgc_disciplina(disciplina_id) ON DELETE CASCADE,
    escopo         TEXT NOT NULL,
    titulo         TEXT NOT NULL,
    url            TEXT,
    texto          TEXT NOT NULL,
    embedding      vector(2048) NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{}',
    atualizado_em  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (disciplina_id, escopo)
);

-- -----------------------------------------------------------------------------
-- emb_ppgc_documento — os chunks dos PDFs
--
-- Vazia até a ingestão dos editais rodar. Existe desde já para que a etapa
-- seguinte seja "preencher `ppgc_documento_chunk` e rodar
-- `load_ppgc.py --only-embeddings`", e não uma migração de schema.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS emb_ppgc_documento CASCADE;
CREATE TABLE emb_ppgc_documento (
    emb_id         BIGSERIAL PRIMARY KEY,
    documento_url  TEXT NOT NULL REFERENCES ppgc_documento(url) ON DELETE CASCADE,
    escopo         TEXT NOT NULL,             -- 'pdf#0', 'pdf#1', …
    titulo         TEXT NOT NULL,
    url            TEXT,
    texto          TEXT NOT NULL,
    embedding      vector(2048) NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{}',
    atualizado_em  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (documento_url, escopo)
);


-- #############################################################################
-- VIEWS
-- #############################################################################

-- Edital com a contagem de PDFs e a URL do edital propriamente dito.
CREATE OR REPLACE VIEW vw_ppgc_edital AS
SELECT e.edital_id,
       e.titulo,
       e.tipo,
       e.nivel,
       e.ano,
       e.semestre,
       e.periodo_letivo,
       e.numero_oficial,
       e.url,
       e.data_publicacao,
       e.resumo,
       d.n_documentos,
       d.url_edital_pdf
  FROM ppgc_edital e
  LEFT JOIN LATERAL (
        SELECT count(*)::INT AS n_documentos,
               max(url) FILTER (WHERE tipo_documento = 'edital') AS url_edital_pdf
          FROM ppgc_edital_documento ed
         WHERE ed.edital_id = e.edital_id
  ) d ON TRUE;

-- O edital MAIS RECENTE de cada tipo/nível. É o que "está aberto?" quer dizer
-- na prática — o site não publica data de encerramento em campo nenhum, então
-- a melhor aproximação honesta é o mais recente de cada linha.
CREATE OR REPLACE VIEW vw_ppgc_edital_mais_recente AS
SELECT DISTINCT ON (tipo, COALESCE(nivel, '-'))
       edital_id, titulo, tipo, nivel, ano, semestre, periodo_letivo,
       numero_oficial, url, data_publicacao
  FROM ppgc_edital
 WHERE ano IS NOT NULL
 ORDER BY tipo, COALESCE(nivel, '-'), ano DESC, semestre DESC NULLS LAST,
          data_publicacao DESC NULLS LAST;

-- Documentos de edital com o contexto do edital, prontos para citar.
CREATE OR REPLACE VIEW vw_ppgc_edital_documento AS
SELECT ed.edital_id,
       e.titulo            AS edital,
       e.tipo,
       e.nivel,
       e.ano,
       e.semestre,
       ed.tipo_documento,
       ed.titulo           AS documento,
       ed.secao,
       ed.url,
       d.extensao,
       d.data_upload,
       (d.texto_extraido IS NOT NULL) AS texto_disponivel
  FROM ppgc_edital_documento ed
  JOIN ppgc_edital e   ON e.edital_id = ed.edital_id
  JOIN ppgc_documento d ON d.url = ed.url;

-- Normas seguras de citar. Use SEMPRE esta view para responder sobre regra:
-- as de fora dela ou estão revogadas ou não constam do índice vigente.
CREATE OR REPLACE VIEW vw_ppgc_normativo_vigente AS
SELECT normativo_id, tipo, numero, ano, titulo, ementa,
       COALESCE(url_pagina, url_pdf) AS url, url_pdf,
       (texto IS NOT NULL) AS tem_texto_integral
  FROM ppgc_normativo
 WHERE vigente IS TRUE;

-- Agenda do Programa a partir de hoje.
CREATE OR REPLACE VIEW vw_ppgc_agenda AS
SELECT uid, titulo, descricao, data_inicio, data_fim, hora_inicio,
       dia_inteiro, ano, mes, semestre, data_por_extenso, url_calendario
  FROM ppgc_calendario_evento
 WHERE situacao IS DISTINCT FROM 'cancelled'
 ORDER BY data_inicio;

-- Requisitos em forma de tabela de conferência.
CREATE OR REPLACE VIEW vw_ppgc_requisito AS
SELECT nivel, obrigatorio, requisito, prazo, url
  FROM ppgc_requisito
 ORDER BY nivel, obrigatorio DESC, requisito;

-- -----------------------------------------------------------------------------
-- vw_computacao_noticia — a linha do tempo COMPLETA (portal + PPGC)
--
-- Existe porque a partição em dois acervos é boa para busca dirigida e ruim
-- para "quais as últimas notícias?": um estudante que faz essa pergunta quer
-- as duas coisas. Criada condicionalmente — este schema não depende do outro.
-- -----------------------------------------------------------------------------
DO $$
BEGIN
    IF to_regclass('public.port_post') IS NOT NULL THEN
        EXECUTE $view$
            CREATE OR REPLACE VIEW vw_computacao_noticia AS
            SELECT 'portal'::TEXT AS acervo, post_id, titulo, url,
                   data_publicacao, data_por_extenso, ano, mes, resumo
              FROM port_post
            UNION ALL
            SELECT 'ppgc'::TEXT AS acervo, post_id, titulo, url,
                   data_publicacao, data_por_extenso, ano, mes, resumo
              FROM ppgc_post
        $view$;
    ELSE
        RAISE NOTICE 'port_post ausente — vw_computacao_noticia não criada. '
                     'Carregue schema_portal_computacao.sql e reexecute este arquivo.';
    END IF;
END $$;


-- #############################################################################
-- ÍNDICES
-- #############################################################################

CREATE INDEX IF NOT EXISTS idx_ppgc_post_data      ON ppgc_post (data_publicacao DESC);
CREATE INDEX IF NOT EXISTS idx_ppgc_post_ano       ON ppgc_post (ano);
CREATE INDEX IF NOT EXISTS idx_ppgc_post_assunto   ON ppgc_post (assunto);
CREATE INDEX IF NOT EXISTS idx_ppgc_pagina_cat     ON ppgc_pagina (categoria);
CREATE INDEX IF NOT EXISTS idx_ppgc_pagina_caminho ON ppgc_pagina (caminho);
CREATE INDEX IF NOT EXISTS idx_ppgc_edital_tipo    ON ppgc_edital (tipo, ano DESC, semestre DESC);
CREATE INDEX IF NOT EXISTS idx_ppgc_edital_nivel   ON ppgc_edital (nivel);
CREATE INDEX IF NOT EXISTS idx_ppgc_edital_num     ON ppgc_edital (numero_oficial);
CREATE INDEX IF NOT EXISTS idx_ppgc_ed_doc_tipo    ON ppgc_edital_documento (tipo_documento);
CREATE INDEX IF NOT EXISTS idx_ppgc_doc_extraido   ON ppgc_documento ((texto_extraido IS NOT NULL));
CREATE INDEX IF NOT EXISTS idx_ppgc_doc_origem     ON ppgc_documento (origem_tipo, origem_id);
CREATE INDEX IF NOT EXISTS idx_ppgc_chunk_url      ON ppgc_documento_chunk (url);
CREATE INDEX IF NOT EXISTS idx_ppgc_norm_tipo      ON ppgc_normativo (tipo, ano DESC);
CREATE INDEX IF NOT EXISTS idx_ppgc_norm_vigente   ON ppgc_normativo (vigente);
CREATE INDEX IF NOT EXISTS idx_ppgc_cal_inicio     ON ppgc_calendario_evento (data_inicio);
CREATE INDEX IF NOT EXISTS idx_ppgc_cal_ano_mes    ON ppgc_calendario_evento (ano, mes);
CREATE INDEX IF NOT EXISTS idx_ppgc_req_nivel      ON ppgc_requisito (nivel);
CREATE INDEX IF NOT EXISTS idx_ppgc_defesa_data    ON ppgc_defesa (data DESC);
CREATE INDEX IF NOT EXISTS idx_ppgc_link_origem    ON ppgc_link (origem_tipo, origem_id);

-- ── Trigram ────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_ppgc_edital_titulo_trgm
    ON ppgc_edital USING gin (titulo gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_ppgc_post_titulo_trgm
    ON ppgc_post USING gin (titulo gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_ppgc_docente_nome_trgm
    ON ppgc_docente USING gin (nome gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_ppgc_disc_nome_trgm
    ON ppgc_disciplina USING gin (nome gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_ppgc_cal_titulo_trgm
    ON ppgc_calendario_evento USING gin (titulo gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_ppgc_norm_ementa_trgm
    ON ppgc_normativo USING gin (COALESCE(ementa, '') gin_trgm_ops);

-- ── Full-text em português ─────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_ppgc_post_fts
    ON ppgc_post USING gin (to_tsvector('portuguese',
        COALESCE(titulo, '') || ' ' || COALESCE(texto, '')));
CREATE INDEX IF NOT EXISTS idx_ppgc_pagina_secao_fts
    ON ppgc_pagina_secao USING gin (to_tsvector('portuguese', texto));
CREATE INDEX IF NOT EXISTS idx_ppgc_faq_fts
    ON ppgc_faq USING gin (to_tsvector('portuguese', pergunta || ' ' || resposta));
CREATE INDEX IF NOT EXISTS idx_ppgc_edital_fts
    ON ppgc_edital USING gin (to_tsvector('portuguese',
        COALESCE(titulo, '') || ' ' || COALESCE(texto, '')));
CREATE INDEX IF NOT EXISTS idx_ppgc_chunk_fts
    ON ppgc_documento_chunk USING gin (to_tsvector('portuguese', texto));

-- ── HNSW cosseno (ver nota sobre halfvec em schema_portal_computacao.sql) ──
CREATE INDEX IF NOT EXISTS idx_emb_ppgc_pagina_hnsw
    ON emb_ppgc_pagina    USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 128);
CREATE INDEX IF NOT EXISTS idx_emb_ppgc_post_hnsw
    ON emb_ppgc_post      USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 128);
CREATE INDEX IF NOT EXISTS idx_emb_ppgc_edital_hnsw
    ON emb_ppgc_edital    USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 128);
CREATE INDEX IF NOT EXISTS idx_emb_ppgc_normativo_hnsw
    ON emb_ppgc_normativo USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 128);
CREATE INDEX IF NOT EXISTS idx_emb_ppgc_faq_hnsw
    ON emb_ppgc_faq       USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 128);
CREATE INDEX IF NOT EXISTS idx_emb_ppgc_linha_hnsw
    ON emb_ppgc_linha_pesquisa USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 128);
CREATE INDEX IF NOT EXISTS idx_emb_ppgc_disciplina_hnsw
    ON emb_ppgc_disciplina USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 128);
CREATE INDEX IF NOT EXISTS idx_emb_ppgc_documento_hnsw
    ON emb_ppgc_documento  USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- Pré-filtros
CREATE INDEX IF NOT EXISTS idx_emb_ppgc_pagina_escopo ON emb_ppgc_pagina (escopo);
CREATE INDEX IF NOT EXISTS idx_emb_ppgc_post_escopo   ON emb_ppgc_post   (escopo);
CREATE INDEX IF NOT EXISTS idx_emb_ppgc_edital_meta
    ON emb_ppgc_edital    USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_emb_ppgc_post_meta
    ON emb_ppgc_post      USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_emb_ppgc_pagina_meta
    ON emb_ppgc_pagina    USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_emb_ppgc_documento_meta
    ON emb_ppgc_documento USING gin (metadata jsonb_path_ops);
