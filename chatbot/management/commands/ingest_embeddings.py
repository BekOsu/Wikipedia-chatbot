import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
from django.core.management.base import BaseCommand
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from ...models import Article

load_dotenv()

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class Command(BaseCommand):
    help = "Ingest Wikipedia articles as embeddings into a FAISS vector database."

    def add_arguments(self, parser):
        parser.add_argument("--delete", action="store_true", help="Delete the existing vector database.")
        parser.add_argument("--limit", type=int, default=1, help="Number of articles to process.")
        parser.add_argument("--path", type=str, default="./wiki_embeddings.pkl",
                            help="Path to save the vector database.")

    def handle(self, *args, **options):
        limit = options.get("limit")
        path = options.get("path")

        if options["delete"]:
            self.stdout.write(f"Are you sure you want to delete {path}? (yes/no)")
            response = input().strip().lower()
            if response == "yes":
                self._prompt_delete_vector_database(path)
            return

        # Fetch articles
        article_qs = Article.objects.all()[:limit]
        if not article_qs.exists():
            self.stdout.write(self.style.ERROR("No articles found in the database."))
            return

        # Process and save embeddings
        vectorstore = self._populate_vectorstore(article_qs)
        self._save_vectorstore(path, vectorstore)

    def _prompt_delete_vector_database(self, path):
        """
        Delete the existing vector database file.
        """
        Path(path).unlink(missing_ok=True)
        self.stdout.write(self.style.SUCCESS("Vector database deleted."))

    def _populate_vectorstore(self, article_qs):
        """
        Generate embeddings and populate the FAISS vector store.
        """
        documents = []
        for article in article_qs:
            self.stdout.write(f"Processing article: {article.title}")
            try:
                # Create title and text documents
                title_doc = Document(
                    page_content=article.title,
                    metadata={"source": article.url, "title": article.title, "field": "title"}
                )
                text_doc = Document(
                    page_content=article.text,
                    metadata={"source": article.url, "title": article.title, "field": "text"}
                )

                # Split text into smaller chunks
                text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=600, chunk_overlap=100)
                documents.extend(text_splitter.split_documents([title_doc, text_doc]))
            except Exception as err:
                self.stdout.write(self.style.ERROR(f"Error processing {article.title}: {err}"))

        # Generate embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        return FAISS.from_documents(documents, embeddings)

    def _save_vectorstore(self, path: str, vectorstore):
        """
        Save the FAISS vector store to a file.
        """
        with open(path, "wb") as f:
            pickle.dump(vectorstore, f)  # type: ignore
        self.stdout.write(self.style.SUCCESS(f"Vector store saved to {path}."))
