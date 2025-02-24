import argparse
import json
import os
import shutil
import yaml
from langchain.schema.document import Document
from sqlalchemy.testing.suite.test_reflection import metadata
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz

from embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma

WHITE = "\033[97m"
BLUE = "\033[34m"
GREEN = "\033[32m"
ORANGE = "\033[38;5;208m"
PINK = "\033[38;5;205m"
RESET = "\033[0m"

def load_config(config_file=None):
    if config_file is None:
        config_file = 'config.yaml'

    base_path = os.path.dirname(__file__)  # Get current file directory
    config_file = os.path.join(base_path, config_file)

    with open(config_file, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def populate_db():
    config = load_config()
    data_topics = config['data_topics']
    default_topic = config['default_topic']

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--debug", action="store_true", help="Additional print statements")
    parser.add_argument("--topic", choices=data_topics.keys(), help="Select the data topic.")
    args = parser.parse_args()

    selected_topic = args.topic if args.topic else default_topic
    topic_config = data_topics[selected_topic]
    topic_dir = topic_config['topic_dir']

    populater = DatabaseManager(topic_dir, args.reset, args.debug)
    populater.save_data()


class DatabaseManager:
    def __init__(self, topic_dir: str, reset: bool = False, debug: bool = False, separate_in_chunks: bool = True,
                 chunk_separation_warning: bool = True, chunk_size: int = 8192):
        self.topic_dir = topic_dir
        self.chroma_dir = f"{topic_dir}/chroma"
        self.data_dir = f"{topic_dir}/documents"
        self.url_mapping_file = f"{topic_dir}/url_mapping.yml"
        self.context_file = f"{topic_dir}/context_data.yaml"
        self.predefined_ids_file = f"{topic_dir}/ids.yaml"
        self.metadata_file = f"{topic_dir}/metadata.json"

        self.context_data = self.open_context_data()
        self.metadata_entries = self.load_metadata_entries()

        self.separate_in_chunks = separate_in_chunks
        self.chunk_separation_warning = chunk_separation_warning
        self.chunk_size = chunk_size
        self.debug = debug
        if debug:
            print(f"{ORANGE}⭕  DEBUG Mode Active{RESET}")
            print("Topic Dir:", topic_dir)
        if reset:
            print(f"{WHITE}✨  Clearing Database{RESET}")
            self.clear_database()
        self.url_mapping = self.load_url_mapping()

        self.db = Chroma(persist_directory=self.chroma_dir, embedding_function=get_embedding_function())

        self.predefined_ids = self.get_predefined_ids()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size)

    def load_metadata_entries(self):
        with open(self.metadata_file, 'r', encoding='utf-8') as file:
            return json.load(file)

    def find_metadata_for_txt(self, txt_file):
        for entry in self.metadata_entries:
            if entry['txt_file'] == txt_file:
                return {key: value for key, value in entry.items() if key != 'txt_file'}
        return {}

    def clear_database(self):
        if os.path.exists(self.chroma_dir):
            shutil.rmtree(self.chroma_dir)

    def find_pdfs_in_folder(self):
        pdf_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pdf')]
        return pdf_files

    def extract_txt_from_pdf(self):
        pdf_files = self.find_pdfs_in_folder()
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.data_dir, pdf_file)
            txt_path = pdf_path.replace('.pdf', '.txt')
            if os.path.exists(txt_path):
                continue
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            self.save_text_to_file(text, txt_path)
        pass

    def save_text_to_file(self, text, txt_path):
        with open(txt_path, 'w', encoding='utf-8') as file:
            file.write(text)


    def save_data(self):
        self.extract_txt_from_pdf()

        txt_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.txt')]

        # Custom bar format with color codes
        bar_format_txt = f"{WHITE}⌛  Adding Text    {{l_bar}}{BLUE}{{bar}}{WHITE}{{r_bar}}{RESET}"

        ncols = shutil.get_terminal_size((80, 20)).columns - 10

        with tqdm(total=len(txt_files), bar_format=bar_format_txt, unit="document", ncols=ncols) as pbar:
            for txt_file in txt_files:
                doc_chunks = self.process_txt(txt_file)
                for chunk in doc_chunks:
                    self.add_to_chroma(chunk)
                pbar.update(1)

    def load_url_mapping(self):
        try:
            with open(self.url_mapping_file, 'r') as file:
                return yaml.safe_load(file).get('documents', {})
        except FileNotFoundError:
            return {}

    def unique(self, metadata: dict):
        # Load the existing database.
        self.db = Chroma(persist_directory=self.chroma_dir, embedding_function=get_embedding_function())

        # Calculate Page IDs.
        metadata = self.calculate_chunk_id(metadata)

        # Add or Update the documents.
        existing_items = self.db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids.yaml"])

        return metadata["id"] not in existing_ids

    def calculate_chunk_id(self, metadata: dict):
        source = metadata.get("doc_name", None)
        url = metadata.get("url", None)
        chunk_number = metadata.get("chunk_number", None)
        if source in self.predefined_ids:
            chunk_id = self.predefined_ids[source]
        elif url in self.predefined_ids:
            chunk_id = self.predefined_ids[url]
        else:
            chunk_id = f"{chunk_number}|{url}|{source}"
        metadata["id"] = chunk_id
        return metadata

    def add_to_chroma(self, doc: Document):
        self.db.add_documents([doc], ids=[doc.metadata["id"]], metadata=[doc.metadata])

    def gather_context(self, file_path, base_url=None):
        context = [self.get_context_from_filename(file_path)]

        if base_url is not None:
            other_docs = self.filter_non_image_documents_for_url(base_url)

            for doc in other_docs:
                with open(doc, 'r', encoding='utf-8') as file:
                    content = file.read()
                context.append(content)
        return "\n".join(context)

    def get_context_from_filename(self, filename):
        basename = os.path.basename(filename)

        if basename in self.context_data['files']:
            return self.context_data['files'][basename].get('context', None)
        return None

    def filter_non_image_documents_for_url(self, specific_url):
        file_names = []
        for file_name, url in self.url_mapping.items():
            if specific_url in url and not url.lower().endswith(
                    ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg')):
                file_names.append(file_name)
        return file_names

    def get_base_url_from_filename(self, filename):
        basename = os.path.basename(filename)

        if basename in self.context_data['files']:
            return self.context_data['files'][basename].get('base_url', None)
        return None

    def open_context_data(self):
        try:
            with open(self.context_file, 'r') as file:
                context_data = yaml.safe_load(file) or {'files': {}}
        except FileNotFoundError:
            context_data = None
        return context_data

    def load_txt_metadata(self, file_path, chunk_number):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        doc_name = os.path.basename(file_path)
        url = self.url_mapping.get(doc_name, None)
        lines = content.split('\n')
        has_filename = len(lines) > 0 and lines[0].startswith("Filename:")
        has_branch = len(lines) > 2 and lines[2].startswith("Branch:")
        has_code_marker = "```" in content
        if has_filename and has_branch and has_code_marker:
            _type = "code"
        else:
            _type = "text"

        metadata = self.find_metadata_for_txt(doc_name)
        metadata.update({"url": url, "doc_name": doc_name, "type": _type, "chunk_number": chunk_number})
        return self.calculate_chunk_id(metadata)

    def clean_metadata(self, metadata):
        return {k: v for k, v in metadata.items() if v is not None}

    def process_txt(self, file_path: str):
        content = self.get_text(file_path)

        if self.separate_in_chunks:
            content_chunks = self.split_text_into_chunks(content)
        else:
            content_chunks = [content]
        documents = []
        for i, chunk in enumerate(content_chunks):
            metadata = self.load_txt_metadata(file_path, i)
            metadata = self.clean_metadata(metadata)
            if metadata is None:
                print(f"{PINK}⚠️  Error: Metadata empty (line 247){RESET}")
            elif self.debug:
                print()
                print(f"{GREEN}📄  Processing: {metadata['doc_name']} {RESET}")
                print("Metadata:", metadata)

            doc = Document(page_content=chunk, id=metadata["id"], metadata=metadata)
            documents.append(doc)

        return documents

    def split_text_into_chunks(self, text):
        chunks = self.text_splitter.split_text(text)
        if len(chunks) > 1 and self.chunk_separation_warning:
            print(f"\n{PINK}⚠️  Warning: A text was split into {len(chunks)} chunks.{RESET}")
        for chunk in chunks:
            yield chunk

    def get_text(self, txt_file):
        with open(txt_file, 'r', encoding='utf-8') as file:
            return file.read()

    def get_predefined_ids(self):
        try:
            with open(self.predefined_ids_file, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            return {}



if __name__ == "__main__":
    populate_