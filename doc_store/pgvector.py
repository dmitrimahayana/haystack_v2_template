from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from dotenv import load_dotenv
import os

# Load env
load_dotenv()

# Define db constants
DB_HOST = os.getenv("db_host")
DB_NAME = os.getenv("db_name")
DB_USERNAME = os.getenv("db_username")
DB_PASSWORD = os.getenv("db_password")
DB_PORT = os.getenv("db_port")


def load_doc_store(table_name: str):
    os.environ["PG_CONN_STR"] = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    document_store = PgvectorDocumentStore(
        schema_name="public",
        table_name=table_name,
        embedding_dimension=768,
        vector_function="cosine_similarity",
        search_strategy="hnsw",
        hnsw_index_name=f"haystack_hnsw_index_{table_name}",
        keyword_index_name=f"haystack_keyword_index_{table_name}"
        # hnsw_recreate_index_if_exists=False,
        # recreate_table=True,
    )
    return document_store


def filter_doc_store(document_store, filters):
    document_store.filter_documents(filters=filters)
    return document_store
