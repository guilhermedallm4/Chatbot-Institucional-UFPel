-- =============================================================================
-- Índices pós-ingestão (executar após crawler/ingest_ufpel.py)
--   docker exec -i rag_postgres psql -U gdlima -d semanticdb < docker/sql/pos_ingestao.sql
-- Versão do create_indexes.sql sem UUIDs fixos e sem CONCURRENTLY (é rápido
-- no volume do minicurso e pode rodar dentro de transação).
-- =============================================================================
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Coleções LangChain: dimensão fixa + HNSW + trigram + metadados
ALTER TABLE langchain_pg_embedding ALTER COLUMN embedding TYPE vector(1024);
CREATE INDEX IF NOT EXISTS idx_lpe_collection_id ON langchain_pg_embedding (collection_id);
CREATE INDEX IF NOT EXISTS idx_lpe_embedding_hnsw ON langchain_pg_embedding
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);
CREATE INDEX IF NOT EXISTS idx_lpe_document_trgm ON langchain_pg_embedding USING gin (document gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_lpe_meta_titulo ON langchain_pg_embedding ((cmetadata->>'titulo'));
CREATE INDEX IF NOT EXISTS idx_lpe_meta_tipo   ON langchain_pg_embedding ((cmetadata->>'tipo'));

-- doc_completos
CREATE INDEX IF NOT EXISTS idx_dc_url ON doc_completos ((dados->>'url')) WHERE dados->>'url' IS NOT NULL;

-- Tabelas físicas por tipo: trigram no conteúdo (busca léxica ILIKE rápida)
CREATE INDEX IF NOT EXISTS disciplinas_conteudo_trgm ON disciplinas USING gin (conteudo gin_trgm_ops);
CREATE INDEX IF NOT EXISTS projetos_conteudo_trgm    ON projetos    USING gin (conteudo gin_trgm_ops);
CREATE INDEX IF NOT EXISTS servidores_conteudo_trgm  ON servidores  USING gin (conteudo gin_trgm_ops);
CREATE INDEX IF NOT EXISTS cursos_conteudo_trgm      ON cursos      USING gin (conteudo gin_trgm_ops);

-- Visão unificada: facilita o SQL do agente (um só lugar para tipo/título/URL)
CREATE OR REPLACE VIEW documentos AS
SELECT d.doc_id, d.tipo, d.titulo, d.dados->>'url' AS url, d.dados
FROM doc_completos d;

-- Garante leitura para o papel do agente em tudo que foi criado
GRANT SELECT ON ALL TABLES IN SCHEMA public TO rag_leitor;

ANALYZE langchain_pg_embedding;
ANALYZE doc_completos;
ANALYZE disciplinas; ANALYZE projetos; ANALYZE servidores; ANALYZE cursos;
ANALYZE disciplinas_info; ANALYZE projetos_info; ANALYZE servidores_info; ANALYZE cursos_info;
