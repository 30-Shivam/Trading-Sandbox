"""MongoDB connection handling.

Reads connection details from the environment (MONGODB_URI, optionally
MONGODB_DB_NAME), loading a local `.env` file if present via python-dotenv.
The client is created once per process and reused (pymongo's MongoClient is
itself a connection pool -- it should not be re-instantiated per call).
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()

MONGODB_URI_ENV = "MONGODB_URI"
MONGODB_DB_NAME_ENV = "MONGODB_DB_NAME"
DEFAULT_DB_NAME = "swingtrade"


class MongoNotConfigured(RuntimeError):
    """Raised when MONGODB_URI isn't set. Callers should treat this as
    "storage is unavailable" and degrade gracefully, not crash."""


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    uri = os.environ.get(MONGODB_URI_ENV)
    if not uri:
        raise MongoNotConfigured(
            f"{MONGODB_URI_ENV} environment variable is not set. "
            "Copy .env.example to .env and fill in your MongoDB Atlas connection string."
        )
    return MongoClient(uri, server_api=ServerApi("1"))


def get_db():
    client = get_client()
    db_name = os.environ.get(MONGODB_DB_NAME_ENV, DEFAULT_DB_NAME)
    return client[db_name]
