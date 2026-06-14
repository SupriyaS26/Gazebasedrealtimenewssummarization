# Gaze-Based Real-Time Summarization

## Overview

This repo contains two main parts:

- `gaze_backend/`: Flask backend for gaze model training, prediction, RSS news fetching, and summary pipeline execution.
- `news_project/`: Django frontend for user authentication and summary display.

> The Django app uses PostgreSQL by default. The Flask app uses local JSON files and XGBoost models.

## Prerequisites

- Python 3.10+ installed.
- PostgreSQL installed and running if you want the Django app to use the default database configuration.
- Recommended: create and activate a Python virtual environment.

## Recommended Dependencies

Install required Python packages in the virtual environment.

```powershell
cd <path-to-repo-root>
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install django flask flask-cors xgboost pandas joblib feedparser psycopg2-binary
```

If you prefer a single install line:

```powershell
pip install django flask flask-cors xgboost pandas joblib feedparser psycopg2-binary
```

## Database Setup for Django

The Django settings file is located at `news_project/news_project/settings.py`. It is configured for PostgreSQL, but you should replace the example values with your own database credentials.

Example placeholders:

- NAME: `<your_database_name>`
- USER: `<your_database_user>`
- PASSWORD: `<your_database_password>`
- HOST: `localhost`
- PORT: `5432`

If you are using PostgreSQL, either:

1. Create a database and user for this app, or
2. Edit `news_project/news_project/settings.py` to use your local database credentials.

## Django App Setup

Run the following from the repo root or from the `news_project/` folder.

```powershell
cd <path-to-repo-root>\news_project
..\.venv\Scripts\Activate.ps1  # if you created the virtual environment in the root
python manage.py migrate
```

Optional: create a superuser for Django admin access.

```powershell
python manage.py createsuperuser
```

## Running the Django Frontend

From the `news_project/` folder:

```powershell
python manage.py runserver
```

Then open:

- `http://127.0.0.1:8000/login/`
- `http://127.0.0.1:8000/signup/`
- `http://127.0.0.1:8000/dashboard/`

## Running the Flask Backend

From the repo root or `gaze_backend/` folder:

```powershell
cd <path-to-repo-root>\gaze_backend
python server.py
```

This starts the Flask backend on the default port `5000`.

### Flask Endpoints

- `/upload` (POST): train gaze correction models from JSON payload.
- `/predict` (POST): predict corrected gaze coordinates.
- `/gaze_scores` (POST): save gaze score payloads to `gaze_scores.json`.
- `/run_pipeline` (GET): execute `pipeline.py`.
- `/summarized` (GET): return summarized output from `output.json`.
- `/rss_news` (GET): fetch RSS articles from configured feeds.
- `/save_articles` (POST): save article data to `news_raw.json`.

## Important Notes

- The Django view `news_project/accounts/views.py` currently uses a placeholder path for `output.json`:

  ```python
  output_file = os.path.join('path_to_output_json', 'output.json')
  ```

  If the frontend cannot find summary output, update that path to the actual file location, e.g.:

  ```python
  output_file = os.path.join('..', 'gaze_backend', 'output.json')
  ```

- The Flask backend writes model files `x_model.pkl` and `y_model.pkl` in `gaze_backend/`.

- If you want to run both servers at once, start the Flask backend first, then start the Django frontend in a second terminal.

## Summary Flow

1. Start Flask backend (`gaze_backend/server.py`).
2. Optionally use `/upload` to train gaze models.
3. Use `/run_pipeline` to create or refresh `output.json`.
4. Start Django frontend and visit `/login/` or `/signup/`.
5. Access `/dashboard/` after login.

## Troubleshooting

- If Django fails on database migration, verify PostgreSQL is running and credentials match `news_project/news_project/settings.py`.
- If Flask fails because of missing packages, install the packages listed above.
- If summary data does not display in Django, fix the `output_file` path in `news_project/accounts/views.py`.

## Contact

If you need help, check the source files:

- `gaze_backend/server.py`
- `news_project/news_project/settings.py`
- `news_project/accounts/views.py`
