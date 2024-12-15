from utility.load_data import *
from doc_store.pgvector import *
from processing.embedding import *
from processing.query import *

TABLE_NAME = "digimon_documents"


def perform_indexing():
    list_doc = load_data()
    doc_store = load_doc_store(TABLE_NAME)
    run_embedding(list_doc, doc_store)


def perform_similarity_search(query_string):
    doc_store = load_doc_store(TABLE_NAME)
    meta_filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.created_at", "operator": ">=", "value": "2024-12-15"},
        ],
    }
    results = run_query(query_string, meta_filters, 10, doc_store)
    for result in results:
        print(result.content, result.score)


if __name__ == '__main__':
    print("demo haystack v2")
    # perform_indexing()
    perform_similarity_search("agumon")
