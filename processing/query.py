from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever


def run_query_search(query_string, meta_filters, top_k, document_store, progress_bar: bool = True):
    pipe = Pipeline()
    pipe.add_component("text_embedder_node", SentenceTransformersTextEmbedder(progress_bar=progress_bar))
    pipe.add_component("retriever_node", PgvectorEmbeddingRetriever(document_store=document_store))
    pipe.connect("text_embedder_node.embedding", "retriever_node.query_embedding")

    results = pipe.run(
        data={
            "text_embedder_node": {"text": query_string},
            "retriever_node": {"filters": meta_filters, "top_k": top_k},
        }
    )
    document_result = results["retriever_node"]["documents"]
    return document_result
