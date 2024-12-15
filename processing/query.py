from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever


def run_query(query_string, top_k, document_store, progress_bar: bool = True):
    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder_node", SentenceTransformersTextEmbedder(progress_bar=progress_bar))
    query_pipeline.add_component("retriever_node", PgvectorEmbeddingRetriever(document_store=document_store))
    query_pipeline.connect("text_embedder_node.embedding", "retriever_node.query_embedding")

    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.created_at", "operator": ">=", "value": "2024-12-15"},
        ],
    }
    results = query_pipeline.run(
        data={
            "text_embedder_node": {"text": query_string},
            "retriever_node": {"filters": filters, "top_k": top_k},
        }
    )
    document_result = results["retriever_node"]["documents"]
    return document_result
