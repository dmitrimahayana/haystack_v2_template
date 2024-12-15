from haystack.document_stores.types import DuplicatePolicy
from haystack.components.embedders import SentenceTransformersDocumentEmbedder


def run_embedding(list_doc, document_store, progress_bar: bool = True):
    document_embedder = SentenceTransformersDocumentEmbedder(progress_bar=progress_bar)
    document_embedder.warm_up()
    documents_with_embeddings = document_embedder.run(list_doc)
    document_store.write_documents(documents_with_embeddings.get("documents"), policy=DuplicatePolicy.OVERWRITE)
