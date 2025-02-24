import argparse
import os

from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

import basic_gpt
from embedding_function import get_embedding_function
import util


PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---

Answer the question based on the above context: {question}
"""

WHITE = "\033[97m"
ORANGE = "\033[38;5;208m"
GREEN = "\033[32m"
RESET = "\033[0m"





def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    config = util.load_config()
    data_topics = config['data_topics']
    default_topic = config['default_topic']
    parser.add_argument("--query_text", type=str, help="The query text.")
    parser.add_argument("--debug", action="store_true", help="Additional print statements")
    parser.add_argument("--topic", choices=data_topics.keys(), help="Select the data topic.")
    args = parser.parse_args()
    if args.debug:
        print(f"{ORANGE}⭕  DEBUG Mode Active{RESET}")

    selected_topic = args.topic if args.topic else default_topic
    topic_config = data_topics[selected_topic]
    topic_dir = topic_config['topic_dir']
    chroma_dir = f"{topic_dir}/chroma"
    # data_dir = f"{topic_dir}/documents"

    query_text = args.query_text
    response_text, _, _ = query_rag(query_text, chroma_dir, debug=args.debug)

    print(f"{WHITE}{response_text}{RESET}")
    print()


def query_rag(query_text: str, chroma_dir: str, unique_role: str=None, unique_prompt_template: str=None,
              debug: bool = False):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_dir, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=4)

    context_texts = []
    metadata_list = []
    for doc, _score in results:
        metadata_list.append(doc.metadata)
        pdf_name = doc.metadata.get("pdf_name", None)
        title = doc.metadata.get("title", None)
        # doc_name = doc.metadata.get("doc_name", None)
        page_content = doc.page_content
        # if type == "image":
        context_texts.append(f"[source: {pdf_name}, {title}]\n{page_content}")

    context_text = "\n\n---\n\n".join(context_texts)
    # sources = [doc.metadata.get("id", None) for doc, _score in results]
    if debug:
        print("Prompt:\n", query_text)
        # print("Retrieved Summarize:\n", results)
        print("Context:\n", context_text)
        print("Metadata:\n", metadata_list)
        print("\n")

    prompt_template = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE if not unique_prompt_template else unique_prompt_template)

    prompt = prompt_template.format(context=context_text, question=query_text)

    role = "Provide accurate and concise answers based solely on the given context." if not unique_role else unique_role
    response_text = basic_gpt.ask_mini_gpt(prompt, role)

    return response_text, context_text, metadata_list


def load_raw_document_content(doc_name: str, data_dir: str):
    file_path = os.path.join(data_dir, doc_name)
    if file_path.endswith('.txt') or file_path.endswith('.csv'):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    return "Content not available"


if __name__ == "__main__":
    main()
