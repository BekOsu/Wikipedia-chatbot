from django.core.management.base import BaseCommand, CommandError
from datasets import load_dataset
from ...models import Article

class Command(BaseCommand):
    help = "Manage Wikipedia articles: load, list, or delete articles."

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=10, help="Number of articles to load")
        parser.add_argument("--subset", type=str, default="20220301.simple", help="Dataset subset to load")
        parser.add_argument("--list", action="store_true", help="List all articles in the database")
        parser.add_argument("--delete", action="store_true", help="Delete all articles from the database")

    def handle(self, *args, **options):
        if options["list"]:
            self._list_articles()
        elif options["delete"]:
            self._delete_articles()
        elif options["limit"] and options["subset"]:
            self._load_articles(options["limit"], options["subset"])
        else:
            self.stdout.write(self.style.ERROR("Invalid arguments. Use --help for guidance."))

    def _list_articles(self):
        articles = Article.objects.all()
        if not articles:
            self.stdout.write("No articles found in the database.")
        for article in articles:
            self.stdout.write(f"{article.id}: {article.title} ({article.url})")

    def _delete_articles(self):
        count, _ = Article.objects.all().delete()
        self.stdout.write(self.style.SUCCESS(f"Deleted {count} articles."))

    def _load_articles(self, limit, subset):
        try:
            self.stdout.write(f"Loading dataset with subset '{subset}' and limit {limit}...")
            wikipedia_data = load_dataset("wikipedia", subset, split=f"train[:{limit}]")

            # Use _prepare_articles to prepare article objects
            articles_to_create = self._prepare_articles(wikipedia_data)

            # Save articles in bulk
            self._save_articles(articles_to_create)

        except Exception as e:
            raise CommandError(f"An error occurred: {e}")

    def _prepare_articles(self, wikipedia_data):
        """
        Convert dataset rows into Article model instances.
        """
        articles = []
        for row in wikipedia_data:
            try:
                articles.append(Article(
                    title=row.get("title", ""),
                    url=row.get("url", ""),
                    text=row.get("text", ""),
                ))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error processing row {row.get('id', 'unknown')}: {e}"))
        return articles

    def _save_articles(self, articles):
        """
        Bulk save Article instances to the database.
        """
        try:
            Article.objects.bulk_create(articles)
            self.stdout.write(self.style.SUCCESS(f"Successfully saved {len(articles)} articles."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"An error occurred while saving articles: {e}"))
