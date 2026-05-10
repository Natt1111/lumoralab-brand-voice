import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self):
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.voyage_api_key = os.getenv("VOYAGE_API_KEY", "")
        self.chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        self.collection_name = os.getenv("COLLECTION_NAME", "lumoralab_brand_voice")
        self._validate()

    def _validate(self):
        missing = [
            name
            for name, val in [
                ("ANTHROPIC_API_KEY", self.anthropic_api_key),
                ("VOYAGE_API_KEY", self.voyage_api_key),
            ]
            if not val
        ]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Copy .env.example to .env and fill in your API keys."
            )


settings = Settings()
