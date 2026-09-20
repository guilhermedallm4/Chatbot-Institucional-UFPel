-- =============================================================================
-- Schema Relacional + Vetorial — Cursos de COMPUTAÇÃO da UFPel
-- =============================================================================
-- Executar:  psql "$DATABASE_URL" -f schema_computacao.sql
--
-- Duas camadas sobre o mesmo crawl, ligadas pelas MESMAS chaves naturais:
--
--   1. CAMADA RELACIONAL  (tabelas normalizadas)
--      Para perguntas factuais, agregações e joins. O LLM monta o SQL.
--      Ex.: "todos os professores da computação e suas áreas"
--           "disciplinas do 4º semestre da versão 1028 de Ciência da Computação"
--           "quantas vagas o SISU 2026 ofereceu para Engenharia de Computação"
--
--   2. CAMADA VETORIAL  (tabelas emb_*, pgvector)
--      Para perguntas semânticas/abertas. Cada linha é um trecho vetorizado
--      com FK para a entidade relacional correspondente.
--      Ex.: "quais professores têm projetos ligados a inteligência artificial"
--           → busca em emb_projeto/emb_servidor → JOIN de volta nos fatos
--
-- Regra de ouro: TODA tabela emb_* carrega a chave da entidade. Um acerto
-- semântico sempre pode virar um JOIN relacional, e vice-versa. É isso que
-- permite ao roteador escolher SQL, semântico, ou os dois em sequência.
--
-- Convenções
--   * Chaves naturais do portal (codigo_ufpel, codigo da disciplina,
--     servidor_id, projeto_id) em vez de UUID → carga idempotente (UPSERT)
--     e joins legíveis para o LLM.
--   * Campo ausente no portal = NULL (não a string "Não há informações
--     disponíveis"). Sentinela em texto quebra COUNT/WHERE/agregação; a
--     renderização de "não informado" é responsabilidade da camada de síntese.
--   * Todo texto longo destinado a busca semântica vive numa tabela
--     *_secao / *_conteudo (uma linha por seção) — nunca numa coluna larga.
--     Isso permite 1 embedding por seção, que é a granularidade certa.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;


-- #############################################################################
-- DOMÍNIO 1 — CURSO
-- #############################################################################

-- -----------------------------------------------------------------------------
-- curso — um registro por curso de Computação (aba superior / div.ficha-dados)
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS curso CASCADE;
CREATE TABLE curso (
    codigo_ufpel        TEXT PRIMARY KEY,          -- 3900, 3910, 7057, 8102, 9130
    nome                TEXT NOT NULL,
    nivel               TEXT,                      -- Graduação | Pós-Graduação
    grau                TEXT,                      -- Bacharelado | Mestrado Acadêmico | Doutorado | Especialização
    modalidade          TEXT,                      -- Presencial | A distância
    turno               TEXT,
    codigo_emec         TEXT,
    codigo_capes        TEXT,
    unidade_nome        TEXT,
    unidade_id          TEXT,                      -- id do portal em /unidades/id/N
    programa            TEXT,                      -- ex.: PPGC
    coordenador_nome    TEXT,
    coordenador_id      TEXT,                      -- FK lógica p/ servidor (pode não ter sido crawleado)
    criacao_reconhecimento TEXT,
    url                 TEXT NOT NULL,
    crawled_at          TIMESTAMPTZ NOT NULL
);

COMMENT ON TABLE  curso IS 'Cursos de Computação da UFPel (graduação, mestrado, doutorado, especialização).';
COMMENT ON COLUMN curso.coordenador_id IS 'servidor_id do coordenador; NULL quando o portal não publica o vínculo.';

-- -----------------------------------------------------------------------------
-- curso_conceito — indicadores de qualidade (span.curso-conceito)
--   "Enade (2021) = 4", "CPC (2021) = 3"
-- -----------------------------------------------------------------------------
-- PK surrogada porque `ano` é anulável (o portal às vezes publica o indicador
-- sem o ano) e coluna de PRIMARY KEY não aceita NULL. A unicidade real é
-- garantida pelo índice com COALESCE, que trata NULL como valor comparável —
-- em UNIQUE comum dois NULLs seriam considerados distintos e duplicariam.
DROP TABLE IF EXISTS curso_conceito CASCADE;
CREATE TABLE curso_conceito (
    conceito_id    BIGSERIAL PRIMARY KEY,
    curso_codigo   TEXT NOT NULL REFERENCES curso(codigo_ufpel) ON DELETE CASCADE,
    indicador      TEXT NOT NULL,                  -- Enade | CPC | CC | IGC
    ano            INT,
    nota           TEXT NOT NULL                   -- TEXT: o portal publica "4", "SC", "-"
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_curso_conceito
    ON curso_conceito (curso_codigo, indicador, COALESCE(ano, -1));

-- -----------------------------------------------------------------------------
-- curso_info_secao — aba "Informações" (accordions), uma linha por seção
--   Graduação : Contextualização, Objetivos, Perfil do Egresso, Competências
--               e habilidades, Organização Curricular, Procedimentos e
--               metodologias de ensino, Avaliação, Integração com a Pesquisa...
--   Pós       : Apresentação, Área de Concentração, Linhas de Pesquisa,
--               Organização Curricular, Pesquisadores, Créditos necessários
--
-- Modelada como (curso, secao, texto) — e não como colunas fixas — porque o
-- conjunto de seções varia por nível de curso e muda a cada revisão do PPC.
-- Cada linha vira um embedding em emb_curso.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS curso_info_secao CASCADE;
CREATE TABLE curso_info_secao (
    info_id        BIGSERIAL PRIMARY KEY,
    curso_codigo   TEXT NOT NULL REFERENCES curso(codigo_ufpel) ON DELETE CASCADE,
    secao          TEXT NOT NULL,                  -- rótulo exibido no accordion
    secao_slug     TEXT NOT NULL,                  -- contextualizacao, objetivos, perfil_egresso...
    ordem          INT  NOT NULL,                  -- ordem de exibição na página
    texto          TEXT NOT NULL,
    UNIQUE (curso_codigo, secao_slug)
);

-- -----------------------------------------------------------------------------
-- curso_forma_ingresso — legenda das siglas de cota (rodapé "(**)")
--   AC = Ampla concorrência, LB_EP = ..., LI_PPI = ...
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS curso_forma_ingresso CASCADE;
CREATE TABLE curso_forma_ingresso (
    curso_codigo   TEXT NOT NULL REFERENCES curso(codigo_ufpel) ON DELETE CASCADE,
    sigla          TEXT NOT NULL,
    descricao      TEXT NOT NULL,
    PRIMARY KEY (curso_codigo, sigla)
);

-- -----------------------------------------------------------------------------
-- curso_vaga — vagas por processo seletivo × cota (formato LONGO)
--   A ficha publica uma matriz (processo × cota). Normalizada para linhas,
--   agregações ficam triviais:
--     SELECT processo, SUM(vagas) FROM curso_vaga
--      WHERE curso_codigo='3900' GROUP BY processo;
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS curso_vaga CASCADE;
CREATE TABLE curso_vaga (
    vaga_id        BIGSERIAL PRIMARY KEY,
    curso_codigo   TEXT NOT NULL REFERENCES curso(codigo_ufpel) ON DELETE CASCADE,
    processo       TEXT NOT NULL,                  -- SISU | PAVE | Transferência...
    ano            INT,
    semestre       INT,
    cota           TEXT NOT NULL,                  -- AC, LB_EP, ..., VR, TOTAL
    vagas          INT  NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_curso_vaga
    ON curso_vaga (curso_codigo, processo, COALESCE(ano, -1), COALESCE(semestre, -1), cota);

COMMENT ON COLUMN curso_vaga.cota IS 'Sigla da forma de ingresso; a linha TOTAL é o total publicado pelo portal para o processo.';

-- -----------------------------------------------------------------------------
-- curriculo_versao — versões de currículo do curso
--   O portal expõe as versões na aba "Turmas Ofertadas"
--   (div.versao > h4 "Versão do Currículo: 1028 (ATUAL)").
--   É a versão que determina em QUAL semestre uma disciplina é ofertada —
--   por isso ela é entidade própria e entra na chave da matriz e das turmas.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS curriculo_versao CASCADE;
CREATE TABLE curriculo_versao (
    curso_codigo   TEXT NOT NULL REFERENCES curso(codigo_ufpel) ON DELETE CASCADE,
    versao         TEXT NOT NULL,                  -- 1027, 1028, 1029
    is_atual       BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (curso_codigo, versao)
);

COMMENT ON TABLE curriculo_versao IS
    'Versao de curriculo equivale ao ano de ingresso do estudante. Turmas de anos diferentes seguem matrizes diferentes, entao toda consulta de grade deve filtrar por versao (ou usar is_atual quando o usuario nao especificar).';


-- #############################################################################
-- DOMÍNIO 2 — DISCIPLINA e MATRIZ CURRICULAR
-- #############################################################################

-- -----------------------------------------------------------------------------
-- disciplina — ficha da disciplina (/disciplinas/cod/N)
--   Entidade INDEPENDENTE de curso: a mesma disciplina aparece em várias
--   matrizes (ALGORITMOS E PROGRAMAÇÃO serve CC, EC e engenharias).
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS disciplina CASCADE;
CREATE TABLE disciplina (
    codigo              TEXT PRIMARY KEY,          -- 22000294
    nome                TEXT NOT NULL,
    tipo_atividade      TEXT,                      -- DISCIPLINA | ESTÁGIO | TCC...
    periodicidade       TEXT,                      -- Semestral | Anual
    creditos            INT,
    carga_horaria       INT,                       -- em horas
    ch_teorica          INT,
    ch_pratica          INT,
    ch_obrigatoria      INT,
    freq_aprovacao      TEXT,                      -- "75%"
    unidade_nome        TEXT,
    unidade_id          TEXT,
    url                 TEXT NOT NULL,
    crawled_at          TIMESTAMPTZ NOT NULL
);

-- -----------------------------------------------------------------------------
-- disciplina_conteudo — Ementa / Objetivos / Conteúdo Programático / Bibliografia
--   Mesma modelagem de curso_info_secao: uma linha por seção → um embedding
--   por seção. É o que permite "qual disciplina ensina árvores B?" acertar a
--   ementa certa em vez de diluir o sinal num texto gigante.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS disciplina_conteudo CASCADE;
CREATE TABLE disciplina_conteudo (
    conteudo_id        BIGSERIAL PRIMARY KEY,
    disciplina_codigo  TEXT NOT NULL REFERENCES disciplina(codigo) ON DELETE CASCADE,
    secao              TEXT NOT NULL,              -- Ementa | Objetivos | ...
    secao_slug         TEXT NOT NULL,              -- ementa | objetivos | conteudo_programatico | bibliografia
    ordem              INT  NOT NULL,
    texto              TEXT NOT NULL,
    UNIQUE (disciplina_codigo, secao_slug)
);

-- -----------------------------------------------------------------------------
-- disciplina_bibliografia — referências individuais (um <li> = uma linha)
--   Separada de disciplina_conteudo porque referência bibliográfica é dado
--   enumerável ("quantos títulos de Cormen a computação usa?").
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS disciplina_bibliografia CASCADE;
CREATE TABLE disciplina_bibliografia (
    biblio_id          BIGSERIAL PRIMARY KEY,
    disciplina_codigo  TEXT NOT NULL REFERENCES disciplina(codigo) ON DELETE CASCADE,
    tipo               TEXT NOT NULL,              -- basica | complementar | nao_classificada
    ordem              INT  NOT NULL,
    referencia         TEXT NOT NULL,
    UNIQUE (disciplina_codigo, tipo, ordem)
);

-- -----------------------------------------------------------------------------
-- disciplina_equivalencia — aba "Disciplinas Equivalentes"
--   Ajuda o estudante: "posso aproveitar X que fiz em outro curso?"
--   A equivalência é sempre relativa a um curso de destino.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS disciplina_equivalencia CASCADE;
CREATE TABLE disciplina_equivalencia (
    equiv_id             BIGSERIAL PRIMARY KEY,
    disciplina_codigo    TEXT NOT NULL REFERENCES disciplina(codigo) ON DELETE CASCADE,
    equivalente_nome     TEXT NOT NULL,
    equivalente_ref      TEXT,                     -- id interno do portal (/disciplinas/id/N)
    equivalente_url      TEXT,
    curso_codigo         TEXT,                     -- curso onde a equivalência vale
    curso_nome           TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_disciplina_equivalencia
    ON disciplina_equivalencia (disciplina_codigo, equivalente_nome, COALESCE(curso_codigo, ''));

COMMENT ON COLUMN disciplina_equivalencia.equivalente_ref IS
    'O portal referencia equivalentes por /disciplinas/id/N (id interno), não por código. Por isso não há FK para disciplina(codigo): a equivalente pode ser de outro curso e nunca ter sido crawleada.';

-- -----------------------------------------------------------------------------
-- curso_matriz — a MATRIZ CURRICULAR: qual disciplina, em qual semestre,
--                de qual curso, em qual versão de currículo.
--
--   Tabela associativa curso × versão × disciplina. A versão está na PK
--   justamente porque a mesma disciplina muda de semestre entre versões.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS curso_matriz CASCADE;
CREATE TABLE curso_matriz (
    curso_codigo       TEXT NOT NULL,
    versao             TEXT NOT NULL,
    disciplina_codigo  TEXT NOT NULL REFERENCES disciplina(codigo) ON DELETE CASCADE,
    bloco              TEXT NOT NULL,              -- semestre | optativas | complementares
    semestre_rotulo    TEXT NOT NULL,              -- "1º Semestre", "Optativas"
    semestre_num       INT,                        -- 1..10; NULL p/ optativas/complementares
    carater            TEXT,                       -- Obrigatória | Optativa | Complementar
    creditos           INT,
    horas              INT,
    ordem              INT NOT NULL,               -- ordem de exibição dentro do bloco
    PRIMARY KEY (curso_codigo, versao, disciplina_codigo),
    FOREIGN KEY (curso_codigo, versao)
        REFERENCES curriculo_versao(curso_codigo, versao) ON DELETE CASCADE
);

COMMENT ON TABLE curso_matriz IS
    'Grade curricular. IMPORTANTE: a aba Matriz Curricular do portal publica apenas a versao vigente. Versoes anteriores aparecem somente de forma indireta, via turmas ofertadas (ver turma_curriculo). Consultas de grade sem versao explicita devem usar curriculo_versao.is_atual.';

-- -----------------------------------------------------------------------------
-- matriz_prerequisito — pré-requisitos (span.tabela-detalhe-info na matriz)
--   Pré-requisito é propriedade do par (curso, versão, disciplina), não da
--   disciplina isolada: EC e CC podem exigir pré-requisitos distintos.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS matriz_prerequisito CASCADE;
CREATE TABLE matriz_prerequisito (
    curso_codigo         TEXT NOT NULL,
    versao               TEXT NOT NULL,
    disciplina_codigo    TEXT NOT NULL,
    prereq_codigo        TEXT NOT NULL,
    prereq_nome          TEXT,
    PRIMARY KEY (curso_codigo, versao, disciplina_codigo, prereq_codigo),
    FOREIGN KEY (curso_codigo, versao, disciplina_codigo)
        REFERENCES curso_matriz(curso_codigo, versao, disciplina_codigo) ON DELETE CASCADE
);


-- #############################################################################
-- DOMÍNIO 3 — PROFESSOR (SERVIDOR)
-- #############################################################################

-- -----------------------------------------------------------------------------
-- servidor — vínculo ATIVO do professor/servidor (/servidores/id/N)
--
--   Atenção de parsing: o portal empilha vínculos encerrados no MESMO
--   div.ficha-dados, dentro de div.vinculo.oculta-exibe-conteudo. Ler a ficha
--   inteira num dict achatado faz o vínculo encerrado sobrescrever o ativo
--   (cargo/titulação errados). O crawler ignora essas subárvores.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS servidor CASCADE;
CREATE TABLE servidor (
    servidor_id           TEXT PRIMARY KEY,        -- id do portal (/servidores/id/132456)
    nome                  TEXT NOT NULL,
    matricula_siape       TEXT,
    categoria             TEXT,                    -- Docente | Técnico-Administrativo
    cargo                 TEXT,
    classe_nivel          TEXT,
    titulacao             TEXT,                    -- Doutorado | Mestrado | ...
    lotacao_nome          TEXT,
    lotacao_id            TEXT,
    regime_jornada        TEXT,
    situacao              TEXT,                    -- Ativo Permanente | Contr. Prof. Substituto | NULL
    vinculo_ativo         BOOLEAN NOT NULL DEFAULT TRUE,
    data_ingresso_servico DATE,
    data_ingresso_ufpel   DATE,
    data_ingresso_cargo   DATE,
    data_saida_cargo      DATE,                    -- preenchida = vínculo com término previsto/ocorrido
    email                 TEXT,
    lattes_url            TEXT,
    curriculo_resumo      TEXT,                    -- Resumo do Lattes (#lattes > Resumo)
    url                   TEXT NOT NULL,
    crawled_at            TIMESTAMPTZ NOT NULL
);

COMMENT ON COLUMN servidor.curriculo_resumo IS
    'Texto livre do Lattes. Fica aqui (e nao em tabela de secoes) por ser unico por servidor. E vetorizado em emb_servidor com escopo curriculo_resumo.';

COMMENT ON COLUMN servidor.vinculo_ativo IS
    'FALSE quando o vinculo lido nao esta corrente na data do crawl: Situacao indica encerramento (Aposentado, Exonerado...) OU Data de saida do Cargo ja passou. A linha e mantida de proposito, porque o professor pode ter ministrado turmas nos ultimos semestres e remove-lo quebraria turma_professor e servidor_disciplina_ministrada. Consultas sobre o corpo docente ATUAL devem filtrar por vinculo_ativo.';

-- -----------------------------------------------------------------------------
-- servidor_funcao — Função/Unidade + gratificação (quem coordena o quê)
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS servidor_funcao CASCADE;
CREATE TABLE servidor_funcao (
    funcao_id      BIGSERIAL PRIMARY KEY,
    servidor_id    TEXT NOT NULL REFERENCES servidor(servidor_id) ON DELETE CASCADE,
    funcao         TEXT NOT NULL,                  -- "Coordenador de Curso de Graduação"
    unidade        TEXT,                           -- "Colegiado do Curso de Ciência da Computação"
    gratificacao   TEXT,                           -- "FUC-01"
    data_inicio    DATE
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_servidor_funcao
    ON servidor_funcao (servidor_id, funcao, COALESCE(unidade, ''));

-- -----------------------------------------------------------------------------
-- servidor_formacao — titulações (#lattes > Formação acadêmica)
--   "Doutorado em Oceanografia (Universidade Federal do Rio Grande, 2019)"
--   é quebrado em nivel / area / instituicao / ano.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS servidor_formacao CASCADE;
CREATE TABLE servidor_formacao (
    formacao_id    BIGSERIAL PRIMARY KEY,
    servidor_id    TEXT NOT NULL REFERENCES servidor(servidor_id) ON DELETE CASCADE,
    ordem          INT  NOT NULL,
    nivel          TEXT,                           -- Doutorado | Mestrado | Graduação | Pós-Doutorado
    area           TEXT,
    instituicao    TEXT,
    ano            INT,
    texto_original TEXT NOT NULL,
    UNIQUE (servidor_id, ordem)
);

-- -----------------------------------------------------------------------------
-- servidor_area_atuacao — áreas de atuação declaradas no Lattes
--
--   Tabela-chave para o caso de uso citado: "buscar todos os professores da
--   computação e o LLM sintetizar a área de cada um" resolve-se com um único
--   SELECT + string_agg, sem embedding nenhum.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS servidor_area_atuacao CASCADE;
CREATE TABLE servidor_area_atuacao (
    area_id        BIGSERIAL PRIMARY KEY,
    servidor_id    TEXT NOT NULL REFERENCES servidor(servidor_id) ON DELETE CASCADE,
    ordem          INT  NOT NULL,
    area           TEXT NOT NULL,                  -- "Ciência da Computação - Sistemas de Computação"
    area_geral     TEXT,                           -- "Ciência da Computação"  (antes do hífen)
    subarea        TEXT,                           -- "Sistemas de Computação" (depois do hífen)
    UNIQUE (servidor_id, ordem)
);

-- -----------------------------------------------------------------------------
-- servidor_curso — ponte professor × curso de Computação
--   Origem: aba "Professores" do curso (quem ministrou nos últimos 3 semestres).
--   É o filtro que define "professor da computação".
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS servidor_curso CASCADE;
CREATE TABLE servidor_curso (
    servidor_id    TEXT NOT NULL REFERENCES servidor(servidor_id) ON DELETE CASCADE,
    curso_codigo   TEXT NOT NULL REFERENCES curso(codigo_ufpel)   ON DELETE CASCADE,
    origem         TEXT NOT NULL DEFAULT 'aba_professores',       -- aba_professores | coordenador | turma
    PRIMARY KEY (servidor_id, curso_codigo, origem)
);


-- #############################################################################
-- DOMÍNIO 4 — PROJETO (apenas vigentes)
-- #############################################################################

-- -----------------------------------------------------------------------------
-- projeto — projetos ATIVOS (data_fim >= hoje) dos professores da computação
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS projeto CASCADE;
CREATE TABLE projeto (
    projeto_id        TEXT PRIMARY KEY,            -- id do portal (/projetos/id/u9876)
    titulo            TEXT NOT NULL,
    enfase            TEXT,                        -- Pesquisa | Extensão | Ensino
    resumo            TEXT,
    coordenador_nome  TEXT,
    coordenador_id    TEXT,
    unidade_origem    TEXT,
    area_cnpq         TEXT,
    eixo_tematico     TEXT,
    linha_extensao    TEXT,
    data_inicio       DATE,
    data_fim          DATE,
    url               TEXT NOT NULL,
    crawled_at        TIMESTAMPTZ NOT NULL
);

COMMENT ON TABLE projeto IS
    'Apenas projetos vigentes na data do crawl (data_fim IS NULL OR data_fim >= CURRENT_DATE). Projetos marcados como finalizados no portal (tr.finalizado) são descartados na coleta.';

-- -----------------------------------------------------------------------------
-- projeto_info_secao — aba Informações do projeto
--   Objetivo Geral, Justificativa, Metodologia, Indicadores/Metas/Resultados
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS projeto_info_secao CASCADE;
CREATE TABLE projeto_info_secao (
    info_id        BIGSERIAL PRIMARY KEY,
    projeto_id     TEXT NOT NULL REFERENCES projeto(projeto_id) ON DELETE CASCADE,
    secao          TEXT NOT NULL,
    secao_slug     TEXT NOT NULL,
    ordem          INT  NOT NULL,
    texto          TEXT NOT NULL,
    UNIQUE (projeto_id, secao_slug)
);

-- -----------------------------------------------------------------------------
-- projeto_equipe — equipe do projeto (aba Equipe)
--   Inclui discentes (sem página no portal → servidor_id NULL).
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS projeto_equipe CASCADE;
CREATE TABLE projeto_equipe (
    equipe_id      BIGSERIAL PRIMARY KEY,
    projeto_id     TEXT NOT NULL REFERENCES projeto(projeto_id) ON DELETE CASCADE,
    nome           TEXT NOT NULL,
    servidor_id    TEXT,                           -- NULL para discentes/externos
    is_servidor    BOOLEAN NOT NULL DEFAULT FALSE,
    ch_semanal     INT,
    data_inicio    DATE,
    data_fim       DATE,
    UNIQUE (projeto_id, nome)
);

COMMENT ON COLUMN projeto_equipe.servidor_id IS
    'Sem FK rígida: a equipe pode conter servidores de fora do escopo do crawl (que não estão na tabela servidor). Use LEFT JOIN.';

-- -----------------------------------------------------------------------------
-- servidor_projeto — participação do professor no projeto
--   Origem: aba "Projetos" da página do professor. Traz papel e CH que a aba
--   Equipe do projeto não traz.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS servidor_projeto CASCADE;
CREATE TABLE servidor_projeto (
    servidor_id    TEXT NOT NULL REFERENCES servidor(servidor_id) ON DELETE CASCADE,
    projeto_id     TEXT NOT NULL REFERENCES projeto(projeto_id)   ON DELETE CASCADE,
    papel          TEXT,                           -- COORDENADOR DO PROJETO | NULL (membro)
    enfase         TEXT,                           -- Ensino | Extensão | Pesquisa
    ch_semanal     INT,
    data_inicio    DATE,
    data_fim       DATE,
    PRIMARY KEY (servidor_id, projeto_id)
);


-- #############################################################################
-- DOMÍNIO 5 — TURMAS OFERTADAS
-- #############################################################################

-- -----------------------------------------------------------------------------
-- turma — a oferta concreta: disciplina × período letivo × código de turma
--
--   Modelagem crítica: a MESMA turma (ex.: T1 de Cálculo 1, 2026/2) aparece
--   sob VÁRIAS versões de currículo na aba do curso. Turma é entidade única;
--   o vínculo com curso/versão/semestre é a tabela turma_curriculo. Sem essa
--   separação a turma seria duplicada 3× e todo COUNT sairia inflado.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS turma CASCADE;
CREATE TABLE turma (
    turma_id           TEXT PRIMARY KEY,           -- <disciplina>-<ano>-<sem>-<codigo>  ex: 22000294-2026-2-M11
    disciplina_codigo  TEXT NOT NULL REFERENCES disciplina(codigo) ON DELETE CASCADE,
    ano                INT  NOT NULL,
    semestre           INT  NOT NULL,              -- 1 | 2
    codigo_turma       TEXT NOT NULL,              -- T1 | M11 | P2
    vagas              INT,
    matriculados       INT,
    UNIQUE (disciplina_codigo, ano, semestre, codigo_turma)
);

COMMENT ON COLUMN turma.turma_id IS
    'Chave determinística derivada dos dados → carga idempotente e joins legíveis sem consultar sequences.';

-- -----------------------------------------------------------------------------
-- turma_curriculo — em qual curso/versão/semestre a turma é ofertada
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS turma_curriculo CASCADE;
CREATE TABLE turma_curriculo (
    turma_id         TEXT NOT NULL REFERENCES turma(turma_id) ON DELETE CASCADE,
    curso_codigo     TEXT NOT NULL,
    versao           TEXT NOT NULL,
    semestre_rotulo  TEXT NOT NULL,                -- "3º Semestre" | "Optativas"
    semestre_num     INT,
    PRIMARY KEY (turma_id, curso_codigo, versao),
    FOREIGN KEY (curso_codigo, versao)
        REFERENCES curriculo_versao(curso_codigo, versao) ON DELETE CASCADE
);

-- -----------------------------------------------------------------------------
-- turma_professor — quem ministra a turma
--   A aba do curso publica só o NOME; o servidor_id vem do cruzamento com a
--   aba "Disciplinas ministradas" da página do professor (que traz
--   ano/semestre + código de turma + disciplina). Quando o cruzamento falha,
--   servidor_id fica NULL e o nome é preservado.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS turma_professor CASCADE;
CREATE TABLE turma_professor (
    turma_id       TEXT NOT NULL REFERENCES turma(turma_id) ON DELETE CASCADE,
    nome           TEXT NOT NULL,
    servidor_id    TEXT REFERENCES servidor(servidor_id) ON DELETE SET NULL,
    papel          TEXT,                           -- responsavel | regente
    PRIMARY KEY (turma_id, nome)
);

-- -----------------------------------------------------------------------------
-- turma_horario — grade de horários (tabela aninhada .grade-horarios)
--   Colunas do portal = período (Manhã/Tarde/Noite); dentro de cada célula,
--   <span.grade-horarios-dia> marca o dia e os horários seguintes herdam o
--   último dia visto.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS turma_horario CASCADE;
CREATE TABLE turma_horario (
    horario_id     BIGSERIAL PRIMARY KEY,
    turma_id       TEXT NOT NULL REFERENCES turma(turma_id) ON DELETE CASCADE,
    ordem          INT  NOT NULL,
    periodo        TEXT,                           -- Manhã | Tarde | Noite
    dia_semana     TEXT,                           -- SEG | TER | QUA | QUI | SEX | SAB
    hora_inicio    TIME,
    hora_fim       TIME,
    UNIQUE (turma_id, ordem)
);

-- -----------------------------------------------------------------------------
-- servidor_disciplina_ministrada — histórico dos 3 últimos semestres
--   Origem: aba "Disciplinas ministradas" da página do professor. Cobre
--   semestres anteriores que já não aparecem na aba de turmas do curso.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS servidor_disciplina_ministrada CASCADE;
CREATE TABLE servidor_disciplina_ministrada (
    ministrada_id      BIGSERIAL PRIMARY KEY,
    servidor_id        TEXT NOT NULL REFERENCES servidor(servidor_id) ON DELETE CASCADE,
    disciplina_codigo  TEXT NOT NULL,
    ano                INT,
    semestre           INT,
    codigo_turma       TEXT,
    carga_horaria      TEXT,                       -- "2+0" (teórica+prática)
    curso_codigo       TEXT,
    curso_nome         TEXT,
    turma_id           TEXT REFERENCES turma(turma_id) ON DELETE SET NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_servidor_disciplina_ministrada
    ON servidor_disciplina_ministrada (
        servidor_id, disciplina_codigo,
        COALESCE(ano, -1), COALESCE(semestre, -1), COALESCE(codigo_turma, ''));


-- #############################################################################
-- CAMADA VETORIAL — pgvector (nvidia/nemotron-3-embed-1b → 2048 dims)
-- #############################################################################
-- Uma tabela por categoria, todas com a MESMA forma de colunas para que o
-- código de retrieval seja genérico:
--
--   <chave da entidade> | escopo | titulo | url | texto | embedding | metadata
--
--   escopo   : qual faceta gerou o texto (ficha, ementa, objetivos, ...).
--              Permite busca dirigida: WHERE escopo = 'ementa'.
--   titulo   : rótulo pronto para citação na resposta do LLM.
--   metadata : filtros pré-busca (curso, unidade, semestre) sem JOIN.
--
-- A FK é o que fecha o ciclo: acerto semântico → chave → JOIN relacional.
-- #############################################################################

-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS emb_curso CASCADE;
CREATE TABLE emb_curso (
    emb_id         BIGSERIAL PRIMARY KEY,
    curso_codigo   TEXT NOT NULL REFERENCES curso(codigo_ufpel) ON DELETE CASCADE,
    escopo         TEXT NOT NULL,                  -- ficha | contextualizacao | objetivos | perfil_egresso | ...
    titulo         TEXT NOT NULL,
    url            TEXT,
    texto          TEXT NOT NULL,
    embedding      vector(2048) NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{}',
    atualizado_em  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (curso_codigo, escopo)
);

-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS emb_disciplina CASCADE;
CREATE TABLE emb_disciplina (
    emb_id             BIGSERIAL PRIMARY KEY,
    disciplina_codigo  TEXT NOT NULL REFERENCES disciplina(codigo) ON DELETE CASCADE,
    escopo             TEXT NOT NULL,              -- ficha | ementa | objetivos | conteudo_programatico | bibliografia
    titulo             TEXT NOT NULL,
    url                TEXT,
    texto              TEXT NOT NULL,
    embedding          vector(2048) NOT NULL,
    metadata           JSONB NOT NULL DEFAULT '{}',
    atualizado_em      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (disciplina_codigo, escopo)
);

-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS emb_servidor CASCADE;
CREATE TABLE emb_servidor (
    emb_id         BIGSERIAL PRIMARY KEY,
    servidor_id    TEXT NOT NULL REFERENCES servidor(servidor_id) ON DELETE CASCADE,
    escopo         TEXT NOT NULL,                  -- ficha | curriculo_resumo | areas_atuacao | formacao
    titulo         TEXT NOT NULL,
    url            TEXT,
    texto          TEXT NOT NULL,
    embedding      vector(2048) NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{}',
    atualizado_em  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (servidor_id, escopo)
);

-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS emb_projeto CASCADE;
CREATE TABLE emb_projeto (
    emb_id         BIGSERIAL PRIMARY KEY,
    projeto_id     TEXT NOT NULL REFERENCES projeto(projeto_id) ON DELETE CASCADE,
    escopo         TEXT NOT NULL,                  -- ficha | resumo | objetivo_geral | justificativa | metodologia
    titulo         TEXT NOT NULL,
    url            TEXT,
    texto          TEXT NOT NULL,
    embedding      vector(2048) NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{}',
    atualizado_em  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (projeto_id, escopo)
);

-- -----------------------------------------------------------------------------
-- emb_turma — oferta descrita em linguagem natural
--   Perguntas de horário costumam vir soltas ("tem aula de banco de dados de
--   manhã?"), então vale um embedding por turma. Consultas exatas
--   (contagens, vagas, matriculados) devem usar as tabelas relacionais.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS emb_turma CASCADE;
CREATE TABLE emb_turma (
    emb_id         BIGSERIAL PRIMARY KEY,
    turma_id       TEXT NOT NULL REFERENCES turma(turma_id) ON DELETE CASCADE,
    escopo         TEXT NOT NULL DEFAULT 'oferta',
    titulo         TEXT NOT NULL,
    url            TEXT,
    texto          TEXT NOT NULL,
    embedding      vector(2048) NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{}',
    atualizado_em  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (turma_id, escopo)
);


-- #############################################################################
-- VIEWS — superfície simplificada para o LLM gerar SQL
-- #############################################################################
-- O roteador do RAG decide entre "montar SQL" e "buscar semanticamente".
-- Quando escolhe SQL, escrever contra estas views é muito mais confiável do
-- que pedir ao LLM para acertar 5 JOINs. Exponha as views no prompt do
-- gerador de SQL e mantenha as tabelas para uso interno.
-- #############################################################################

-- Professores da computação com áreas e cursos já agregados.
-- Responde "todos os professores da computação e sua área" em 1 query.
CREATE OR REPLACE VIEW vw_professor_computacao AS
SELECT
    s.servidor_id,
    s.nome,
    s.cargo,
    s.titulacao,
    s.lotacao_nome,
    s.situacao,
    s.vinculo_ativo,
    s.data_saida_cargo,
    s.email,
    s.lattes_url,
    s.curriculo_resumo,
    s.url,
    ARRAY_REMOVE(ARRAY_AGG(DISTINCT c.nome), NULL)                  AS cursos,
    ARRAY_REMOVE(ARRAY_AGG(DISTINCT c.codigo_ufpel), NULL)          AS cursos_codigos,
    ARRAY_REMOVE(ARRAY_AGG(DISTINCT a.area), NULL)                  AS areas_atuacao,
    ARRAY_REMOVE(ARRAY_AGG(DISTINCT a.area_geral), NULL)            AS areas_gerais
FROM servidor s
LEFT JOIN servidor_curso        sc ON sc.servidor_id = s.servidor_id
LEFT JOIN curso                 c  ON c.codigo_ufpel = sc.curso_codigo
LEFT JOIN servidor_area_atuacao a  ON a.servidor_id  = s.servidor_id
GROUP BY s.servidor_id;

-- Matriz curricular achatada (curso + versão + disciplina + ementa).
CREATE OR REPLACE VIEW vw_disciplina_curso AS
SELECT
    m.curso_codigo,
    c.nome                AS curso_nome,
    m.versao,
    v.is_atual            AS versao_atual,
    m.bloco,
    m.semestre_rotulo,
    m.semestre_num,
    m.carater,
    d.codigo              AS disciplina_codigo,
    d.nome                AS disciplina_nome,
    COALESCE(m.creditos, d.creditos)      AS creditos,
    COALESCE(m.horas, d.carga_horaria)    AS horas,
    d.unidade_nome,
    d.url                 AS disciplina_url,
    (SELECT texto FROM disciplina_conteudo dc
      WHERE dc.disciplina_codigo = d.codigo AND dc.secao_slug = 'ementa')    AS ementa,
    (SELECT texto FROM disciplina_conteudo dc
      WHERE dc.disciplina_codigo = d.codigo AND dc.secao_slug = 'objetivos') AS objetivos
FROM curso_matriz m
JOIN curso            c ON c.codigo_ufpel = m.curso_codigo
JOIN curriculo_versao v ON v.curso_codigo = m.curso_codigo AND v.versao = m.versao
JOIN disciplina       d ON d.codigo       = m.disciplina_codigo;

-- Turmas com disciplina, curso, versão, professores e horários agregados.
CREATE OR REPLACE VIEW vw_turma_completa AS
SELECT
    t.turma_id,
    t.ano,
    t.semestre,
    t.codigo_turma,
    t.vagas,
    t.matriculados,
    d.codigo  AS disciplina_codigo,
    d.nome    AS disciplina_nome,
    tc.curso_codigo,
    c.nome    AS curso_nome,
    tc.versao,
    tc.semestre_rotulo,
    tc.semestre_num,
    ARRAY_REMOVE(ARRAY_AGG(DISTINCT tp.nome), NULL)        AS professores,
    ARRAY_REMOVE(ARRAY_AGG(DISTINCT tp.servidor_id), NULL) AS professores_ids,
    (SELECT STRING_AGG(
                th.dia_semana || ' ' || TO_CHAR(th.hora_inicio,'HH24:MI')
                             || '-'  || TO_CHAR(th.hora_fim,'HH24:MI'),
                ', ' ORDER BY th.ordem)
       FROM turma_horario th WHERE th.turma_id = t.turma_id)  AS horarios
FROM turma t
JOIN disciplina        d  ON d.codigo = t.disciplina_codigo
LEFT JOIN turma_curriculo tc ON tc.turma_id = t.turma_id
LEFT JOIN curso         c  ON c.codigo_ufpel = tc.curso_codigo
LEFT JOIN turma_professor tp ON tp.turma_id = t.turma_id
GROUP BY t.turma_id, d.codigo, d.nome, tc.curso_codigo, c.nome,
         tc.versao, tc.semestre_rotulo, tc.semestre_num;

-- Uma linha POR TURMA — versões, cursos, professores e horários agregados.
--
-- Existe porque vw_turma_completa (uma linha por turma x curso x versao) faz
-- o LLM duplicar a resposta em perguntas sobre TURMAS: "quais turmas têm mais
-- matriculados que vagas" devolvia 160 linhas para 58 turmas. Avisar no prompt
-- não resolveu (o modelo ignora a instrução em 2 de 3 tentativas), então a
-- correção é estrutural: nesta view a duplicação é impossível, porque a
-- granularidade é a turma.
--
-- Use vw_turma para perguntas sobre a turma; vw_turma_completa apenas quando a
-- pergunta for sobre a grade de uma VERSÃO específica de currículo.
CREATE OR REPLACE VIEW vw_turma AS
SELECT
    t.turma_id,
    t.ano,
    t.semestre,
    t.codigo_turma,
    t.vagas,
    t.matriculados,
    d.codigo                                                  AS disciplina_codigo,
    d.nome                                                    AS disciplina_nome,
    d.creditos,
    d.carga_horaria,
    ARRAY_REMOVE(ARRAY_AGG(DISTINCT c.nome), NULL)            AS cursos,
    ARRAY_REMOVE(ARRAY_AGG(DISTINCT tc.curso_codigo), NULL)   AS cursos_codigos,
    ARRAY_REMOVE(ARRAY_AGG(DISTINCT tc.versao), NULL)         AS versoes,
    ARRAY_REMOVE(ARRAY_AGG(DISTINCT tc.semestre_rotulo), NULL) AS semestres_rotulo,
    MIN(tc.semestre_num)                                      AS semestre_num,
    ARRAY_REMOVE(ARRAY_AGG(DISTINCT tp.nome), NULL)           AS professores,
    ARRAY_REMOVE(ARRAY_AGG(DISTINCT tp.servidor_id), NULL)    AS professores_ids,
    (SELECT STRING_AGG(
                th.dia_semana || ' ' || TO_CHAR(th.hora_inicio,'HH24:MI')
                             || '-'  || TO_CHAR(th.hora_fim,'HH24:MI'),
                ', ' ORDER BY th.ordem)
       FROM turma_horario th WHERE th.turma_id = t.turma_id)  AS horarios
FROM turma t
JOIN disciplina d           ON d.codigo       = t.disciplina_codigo
LEFT JOIN turma_curriculo tc ON tc.turma_id   = t.turma_id
LEFT JOIN curso c            ON c.codigo_ufpel = tc.curso_codigo
LEFT JOIN turma_professor tp ON tp.turma_id   = t.turma_id
GROUP BY t.turma_id, d.codigo, d.nome, d.creditos, d.carga_horaria;

COMMENT ON VIEW vw_turma IS
    'Uma linha por turma, com cursos/versoes/professores agregados em arrays. Preferir esta view para perguntas sobre turmas; vw_turma_completa devolve uma linha por (turma x curso x versao) e duplica a resposta.';

COMMENT ON VIEW vw_turma_completa IS
    'Uma linha por (turma x curso x versao de curriculo). Uma mesma turma ofertada em duas versoes aparece duas vezes: correto para consultas de grade por versao, mas parece duplicata quando a pergunta e sobre a turma em si. Nesse caso agregue as versoes (array_agg(DISTINCT versao)) ou filtre uma versao especifica.';

-- Projetos vigentes com os professores envolvidos.
-- Base para "quais professores têm projetos de inteligência artificial":
-- busca semântica em emb_projeto devolve projeto_id → JOIN aqui.
CREATE OR REPLACE VIEW vw_projeto_professor AS
SELECT
    p.projeto_id,
    p.titulo,
    p.enfase,
    p.area_cnpq,
    p.resumo,
    p.coordenador_nome,
    p.unidade_origem,
    p.data_inicio,
    p.data_fim,
    p.url,
    sp.servidor_id,
    s.nome        AS professor_nome,
    s.lotacao_nome AS professor_lotacao,
    sp.papel,
    sp.ch_semanal
FROM projeto p
LEFT JOIN servidor_projeto sp ON sp.projeto_id  = p.projeto_id
LEFT JOIN servidor         s  ON s.servidor_id  = sp.servidor_id;

-- Grade completa por versão — atende "matriz do ano de ingresso X".
CREATE OR REPLACE VIEW vw_matriz_versao AS
SELECT
    v.curso_codigo,
    c.nome AS curso_nome,
    v.versao,
    v.is_atual,
    COUNT(m.disciplina_codigo)                                  AS n_disciplinas,
    SUM(COALESCE(m.creditos, 0))                                AS creditos_totais,
    SUM(COALESCE(m.horas, 0))                                   AS horas_totais,
    COUNT(*) FILTER (WHERE m.bloco = 'semestre')                AS n_obrigatorias,
    COUNT(*) FILTER (WHERE m.bloco = 'optativas')               AS n_optativas
FROM curriculo_versao v
JOIN curso c ON c.codigo_ufpel = v.curso_codigo
LEFT JOIN curso_matriz m ON m.curso_codigo = v.curso_codigo AND m.versao = v.versao
GROUP BY v.curso_codigo, c.nome, v.versao, v.is_atual;


-- #############################################################################
-- ÍNDICES
-- #############################################################################

-- ── Relacionais: colunas de JOIN e de filtro ────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_curso_matriz_disc      ON curso_matriz (disciplina_codigo);
CREATE INDEX IF NOT EXISTS idx_curso_matriz_sem       ON curso_matriz (curso_codigo, versao, semestre_num);
CREATE INDEX IF NOT EXISTS idx_matriz_prereq_disc     ON matriz_prerequisito (disciplina_codigo);
CREATE INDEX IF NOT EXISTS idx_disc_conteudo_disc     ON disciplina_conteudo (disciplina_codigo);
CREATE INDEX IF NOT EXISTS idx_disc_equiv_disc        ON disciplina_equivalencia (disciplina_codigo);
CREATE INDEX IF NOT EXISTS idx_servidor_curso_curso   ON servidor_curso (curso_codigo);
CREATE INDEX IF NOT EXISTS idx_servidor_area_serv     ON servidor_area_atuacao (servidor_id);
CREATE INDEX IF NOT EXISTS idx_servidor_projeto_proj  ON servidor_projeto (projeto_id);
CREATE INDEX IF NOT EXISTS idx_projeto_equipe_serv    ON projeto_equipe (servidor_id);
CREATE INDEX IF NOT EXISTS idx_turma_disc             ON turma (disciplina_codigo);
CREATE INDEX IF NOT EXISTS idx_turma_periodo          ON turma (ano, semestre);
CREATE INDEX IF NOT EXISTS idx_turma_curriculo_curso  ON turma_curriculo (curso_codigo, versao);
CREATE INDEX IF NOT EXISTS idx_turma_prof_serv        ON turma_professor (servidor_id);
CREATE INDEX IF NOT EXISTS idx_turma_horario_turma    ON turma_horario (turma_id);
CREATE INDEX IF NOT EXISTS idx_sdm_disc               ON servidor_disciplina_ministrada (disciplina_codigo);
CREATE INDEX IF NOT EXISTS idx_projeto_vigente        ON projeto (data_fim) WHERE data_fim IS NOT NULL;

-- ── Trigram: casamento de nome vindo do LLM ("prof. Ana Marilza") ───────────
CREATE INDEX IF NOT EXISTS idx_servidor_nome_trgm     ON servidor    USING gin (nome         gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_disciplina_nome_trgm   ON disciplina  USING gin (nome         gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_curso_nome_trgm        ON curso       USING gin (nome         gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_projeto_titulo_trgm    ON projeto     USING gin (titulo       gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_servidor_area_trgm     ON servidor_area_atuacao USING gin (area gin_trgm_ops);

-- ── Full-text em português: filtro léxico barato antes do vetorial ──────────
CREATE INDEX IF NOT EXISTS idx_disc_conteudo_fts
    ON disciplina_conteudo USING gin (to_tsvector('portuguese', texto));
CREATE INDEX IF NOT EXISTS idx_projeto_info_fts
    ON projeto_info_secao  USING gin (to_tsvector('portuguese', texto));
CREATE INDEX IF NOT EXISTS idx_curso_info_fts
    ON curso_info_secao    USING gin (to_tsvector('portuguese', texto));
CREATE INDEX IF NOT EXISTS idx_servidor_resumo_fts
    ON servidor            USING gin (to_tsvector('portuguese', COALESCE(curriculo_resumo, '')));

-- ── HNSW cosseno nas tabelas vetoriais ──────────────────────────────────────
-- ATENÇÃO à forma do índice: o nemotron-3-embed-1b produz 2048 dimensões e o
-- pgvector NÃO indexa `vector` com mais de 2000 dims em HNSW
--   ERROR: column cannot have more than 2000 dimensions for hnsw index
-- A saída canônica é indexar a projeção em `halfvec` (meia precisão), que o
-- HNSW suporta até 4000 dims. A coluna continua `vector(2048)` — precisão
-- integral no armazenamento — e só o índice é aproximado.
--
-- Consequência para quem consulta: o ORDER BY precisa repetir exatamente a
-- expressão do índice, senão o planner ignora o HNSW e faz seq scan:
--   ORDER BY embedding::halfvec(2048) <=> $1::halfvec(2048)
-- (ver busca_semantica.py, que já monta a consulta nessa forma)
-- m=16 / ef_construction=128 é o trade-off padrão qualidade × memória.
-- Criar DEPOIS da carga dos embeddings é mais rápido; são idempotentes.
CREATE INDEX IF NOT EXISTS idx_emb_curso_hnsw
    ON emb_curso      USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 128);
CREATE INDEX IF NOT EXISTS idx_emb_disciplina_hnsw
    ON emb_disciplina USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 128);
CREATE INDEX IF NOT EXISTS idx_emb_servidor_hnsw
    ON emb_servidor   USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 128);
CREATE INDEX IF NOT EXISTS idx_emb_projeto_hnsw
    ON emb_projeto    USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 128);
CREATE INDEX IF NOT EXISTS idx_emb_turma_hnsw
    ON emb_turma      USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 128);

-- Filtro por escopo antes do vetorial (WHERE escopo = 'ementa')
CREATE INDEX IF NOT EXISTS idx_emb_curso_escopo      ON emb_curso      (escopo);
CREATE INDEX IF NOT EXISTS idx_emb_disciplina_escopo ON emb_disciplina (escopo);
CREATE INDEX IF NOT EXISTS idx_emb_servidor_escopo   ON emb_servidor   (escopo);
CREATE INDEX IF NOT EXISTS idx_emb_projeto_escopo    ON emb_projeto    (escopo);

-- Filtros pré-busca vindos do metadata JSONB
CREATE INDEX IF NOT EXISTS idx_emb_disciplina_meta   ON emb_disciplina USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_emb_servidor_meta     ON emb_servidor   USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_emb_projeto_meta      ON emb_projeto    USING gin (metadata jsonb_path_ops);
