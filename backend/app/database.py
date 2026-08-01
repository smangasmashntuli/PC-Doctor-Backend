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



SQLALCHEMY_DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}"
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

    with engine.begin() as connection:
        if "username" in existing_columns:
            connection.execute(text("ALTER TABLE users DROP COLUMN username"))

        for column_name, definition in column_definitions.items():
            if column_name not in existing_columns:
                connection.execute(text(f"ALTER TABLE users ADD COLUMN {column_name} {definition}"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]