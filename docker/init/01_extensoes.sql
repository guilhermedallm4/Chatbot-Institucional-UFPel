-- Executado automaticamente na primeira inicialização do container
-- (docker-entrypoint-initdb.d). Idempotente.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Papel somente leitura para o agente executar SQL com segurança.
-- A senha pode ser trocada depois: ALTER ROLE rag_leitor PASSWORD '...';
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_leitor') THEN
        CREATE ROLE rag_leitor LOGIN PASSWORD 'leitor';
    END IF;
END
$$;
GRANT CONNECT ON DATABASE semanticdb TO rag_leitor;
GRANT USAGE ON SCHEMA public TO rag_leitor;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO rag_leitor;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO rag_leitor;
