#database.py
import os
from urllib.parse import quote_plus

from fastapi import Depends
from sqlalchemy import create_engine, inspect, text
from typing import Annotated
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from dotenv import load_dotenv
load_dotenv()

MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

configured_database_url = os.getenv("DATABASE_URL")
mysql_is_configured = all((MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_DATABASE))
SQLALCHEMY_DATABASE_URL = configured_database_url or (
    f"mysql+pymysql://{quote_plus(MYSQL_USER)}:{quote_plus(MYSQL_PASSWORD)}@{MYSQL_HOST}/{MYSQL_DATABASE}"
    if mysql_is_configured
    else "sqlite:///./database.db"
)

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def sync_users_table_schema():
    inspector = inspect(engine)
    if not inspector.has_table("users"):
        return

    existing_columns = {column["name"] for column in inspector.get_columns("users")}
    column_definitions = {
        "email": "VARCHAR(50) NULL",
        "hashed_password": "VARCHAR(255) NULL",
        "name": "VARCHAR(255) NULL",
        "surname": "VARCHAR(255) NULL",
        "is_active": "BOOLEAN DEFAULT TRUE",
        "created_at": "DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6)",
        "updated_at": "DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)",
    }

    if engine.dialect.name == "sqlite":
        column_definitions = {
            "email": "VARCHAR(50)",
            "hashed_password": "VARCHAR(255)",
            "name": "VARCHAR(255)",
            "surname": "VARCHAR(255)",
            "is_active": "BOOLEAN",
            "created_at": "DATETIME",
            "updated_at": "DATETIME",
        }

    with engine.begin() as connection:
        for column_name, definition in column_definitions.items():
            if column_name not in existing_columns:
                connection.execute(text(f"ALTER TABLE users ADD COLUMN {column_name} {definition}"))


def sync_laptop_specs_schema():
    """Bring the ``laptop_specs`` table up to date with the ORM models.

    ``Base.metadata.create_all`` only creates missing tables, it never alters
    existing ones. ``image_url`` used to be ``VARCHAR(500)`` which cannot hold
    the base64 data URIs returned by the image generator, so widen it in place.
    """
    inspector = inspect(engine)
    if not inspector.has_table("laptop_specs"):
        return

    columns = {column["name"]: str(column["type"]) for column in inspector.get_columns("laptop_specs")}
    image_url_type = columns.get("image_url")
    if not image_url_type:
        return

    if engine.dialect.name == "mysql" and image_url_type.upper().startswith("VARCHAR"):
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE laptop_specs MODIFY image_url TEXT NULL"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]