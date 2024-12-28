from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator, OllamaGenerator
from haystack.components.embedders import SentenceTransformersTextEmbedder
# from haystack.dataclasses import ChatMessage
from haystack import Pipeline

# no parameter init, we don't use any runtime template variables
prompt_builder = ChatPromptBuilder()
# chat_generator = OllamaChatGenerator(model="mistral:latest",
#                                      url="http://localhost:11434",
#                                      generation_kwargs={
#                                          "temperature": 0.9,
#                                      })
text_generator = OllamaGenerator(model="mistral:latest",
                                 url="http://localhost:11434")


def run_chatbot(document_store, question_text, progress_bar: bool = True):
    # query_pipeline = Pipeline()
    # query_pipeline.add_component("text_embedder_node", SentenceTransformersTextEmbedder(progress_bar=progress_bar))
    # query_pipeline.add_component("retriever_node", PgvectorEmbeddingRetriever(document_store=document_store))
    # query_pipeline.add_component("prompt_builder_node", prompt_builder)
    # query_pipeline.add_component("llm_node", generator)
    # query_pipeline.connect("text_embedder_node.embedding", "retriever_node.query_embedding")
    # query_pipeline.connect("retriever_node.query_embedding", "prompt_builder_node.documents")
    # query_pipeline.connect("prompt_builder_node.prompt", "llm_node.messages")

    # template_messages = [
    #     ChatMessage.from_system("You are expert in digimon world, you know only digimon from previous context."),
    #     ChatMessage.from_user("Tell me about {{question_text}}")]
    # result = query_pipeline.run(
    #     data={
    #         "text_embedder_node": {"text": question_text},
    #         # "prompt_builder": {"template_variables": {"question_text": question_text}, "template": template_messages}
    #     })

    template = """
    Given only the following information, answer the question.
    Ignore your own knowledge.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{ query }}?
    """

    pipe = Pipeline()
    pipe.add_component("text_embedder_node", SentenceTransformersTextEmbedder(progress_bar=progress_bar))
    pipe.add_component("retriever_node", PgvectorEmbeddingRetriever(document_store=document_store))
    pipe.add_component("prompt_builder_node", PromptBuilder(template=template))
    pipe.add_component("llm_node", text_generator)
    pipe.connect("text_embedder_node.embedding", "retriever_node.query_embedding")
    pipe.connect("retriever_node.documents", "prompt_builder_node.documents")
    pipe.connect("prompt_builder_node", "llm_node")

    result = pipe.run(
        data={
            "text_embedder_node": {"text": question_text},
            "retriever_node": {"top_k": 100},
            "prompt_builder_node": {"query": question_text},
        }
    )
    return result["llm_node"]
