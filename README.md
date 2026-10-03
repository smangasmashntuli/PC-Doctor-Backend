# PC Doctor AI Backend

This repository contains the FastAPI backend for PC Doctor AI. It provides authentication, laptop setup and specification ingestion, retrieval-augmented troubleshooting, Gemini-powered responses, safety guardrails, component explanations, notifications, and cached YouTube repair tutorials for the mobile frontend.

## Features

- JWT authentication with bcrypt password hashing.
- MySQL persistence through SQLAlchemy.
- Laptop setup and specification storage.
- Web-based laptop specification ingestion and ChromaDB vector search.
- Gemini 1.5 Flash troubleshooting responses and component explanations.
- Gemini Imagen laptop image generation when configured.
- Safety checks for high-risk repair questions.
- YouTube Data API tutorial search with database caching.
- Maintenance notifications and read/unread status.
- OpenAPI documentation through FastAPI.

## Technology

- FastAPI and Uvicorn
- SQLAlchemy with MySQL/PyMySQL
- Pydantic
- JWT (`python-jose`) and bcrypt (`passlib`)
- ChromaDB, BeautifulSoup, Requests, and lxml for RAG ingestion
- Google Gemini and YouTube Data API integrations
- Pytest and pytest-asyncio

## Prerequisites

- Python 3.10 or newer
- A running MySQL database and a database user with permission to create/update the application tables
- Gemini API credentials for AI-backed endpoints
- A YouTube Data API key if tutorial search is required

## Install on Windows

From this repository root (`backend/LearningPython`):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents activation, use the Python executable in `.venv\Scripts\python.exe` directly or adjust the local execution policy according to your machine's policy.

## Environment variables

Create a `.env` file in `backend/LearningPython`:

```env
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_HOST=127.0.0.1:3306
MYSQL_DATABASE=native_app

SECRET_KEY=replace_with_a_long_random_secret
GEMINI_FLASH_API_KEY=your_gemini_api_key

# Required only for generated laptop images
GEMINI_IMAGEN_API_KEY=your_gemini_image_api_key
# Optional; defaults to imagen-3.0-generate-002
GEMINI_IMAGEN_MODEL=imagen-3.0-generate-002

# Optional; YouTube search returns no live results without it
YOUTUBE_API_KEY=your_youtube_data_api_key

# Optional; defaults to ./chroma_db
CHROMADB_PATH=./chroma_db
```

The database module uses `DATABASE_URL` when it is set, otherwise it uses the four `MYSQL_*` variables. When neither is configured, local development falls back to `sqlite:///./database.db`. Keep `.env` out of source control and never commit credentials.

`GEMINI_FLASH_API_KEY` is needed for chat and component explanations. `GEMINI_IMAGEN_API_KEY` is only needed for image generation. `YOUTUBE_API_KEY` is optional; without it, tutorial search is disabled gracefully.

## Run the API

Run these commands from `backend/LearningPython` so the `backend.app` package imports resolve correctly:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API is available at `http://localhost:8000`.

Useful URLs:

- Health check: `http://localhost:8000/ping`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

The application creates missing SQLAlchemy tables when `main.py` is imported. It also performs a small users-table schema synchronization at startup.

## API areas

All paths also accept the trailing-slash variant where implemented.

| Area           | Routes                                                                                                         |
| -------------- | -------------------------------------------------------------------------------------------------------------- |
| Health         | `GET /`, `GET /ping`                                                                                           |
| Authentication | `POST /signup`, `POST /login`, `GET /me`                                                                       |
| Laptop setup   | `POST /setup`, `GET /setup/status`                                                                             |
| Specifications | `POST /laptop/specs/ingest`                                                                                    |
| Device visuals | `GET /laptop/{laptop_id}/3d-model`                                                                             |
| AI assistance  | `POST /chat/message`, `GET /chat/history/{session_id}`, `POST /chat/explain-component`                         |
| Tutorials      | `GET /videos/search`                                                                                           |
| Notifications  | `GET /notifications`, `PATCH /notifications/{notification_id}/read`, `POST /notifications/trigger-maintenance` |

Except for health, sign-up, and login, the application expects a valid `Authorization: Bearer <token>` header.

## Test

Run the complete test suite from this repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests/ -v
```

Useful focused commands:

```powershell
python -m pytest tests/test_chat_safety.py -v
python -m pytest tests/test_rag_engine.py -v
python -m pytest tests/test_youtube_notifications.py -v
python -m pytest tests/test_phase4_visuals.py -v
```

Many tests mock external services such as Gemini, ChromaDB, and YouTube. Integration behavior still depends on the configured database and environment when tests exercise application startup.

## Project structure

```text
main.py                         FastAPI application and route definitions
backend/app/auth.py             JWT and password authentication
backend/app/database.py         SQLAlchemy engine and database session
backend/app/models.py           Database models
backend/app/rag_engine.py       Specification retrieval and ingestion
backend/app/vector_store.py     ChromaDB integration
backend/app/gemini_service.py   Gemini chat and image generation
backend/app/youtube_service.py  YouTube search and caching
backend/app/notifications.py    Maintenance notification logic
tests/                          Unit and integration-style tests
docs/                           RAG design documentation
chroma_db/                      Local ChromaDB persistence
```

## Frontend connection

The Expo client defaults to port `8000`. Set `EXPO_PUBLIC_API_URL` in the frontend repository when the backend runs on another host or when testing from a physical device. The backend CORS configuration currently includes the local Expo origins used during development.
