# Analytics Bot with BigQuery Session Management

A system that processes analytical queries using Google BigQuery and Vertex AI (Gemini), with a robust session management system that tracks query state and results.

## Features

- Natural language processing of analytical queries
- Automatic SQL query generation
- BigQuery integration for data retrieval
- Advanced session management with append-only pattern
- Intelligent summarization of results

## Session Management System

The session management system uses BigQuery for storing and retrieving session data with an append-only pattern:

- **Append-Only Pattern**: Due to BigQuery's streaming buffer limitations (which prevent updating recently inserted rows), we use an append-only pattern for session updates.
- **Session History**: Each update to a session creates a new row, allowing us to maintain a complete history of session updates.
- **Latest State Retrieval**: When querying a session, we get the most recent row for that session ID.

## Prerequisites

- Python 3.8+
- Google Cloud Platform account with:
  - BigQuery API enabled
  - Vertex AI API enabled
  - Service account with appropriate permissions

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd analytics-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```env
PROJECT_ID=your-gcp-project-id
DATASET_ID=your-bigquery-dataset
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
```

4. Configure Google Cloud credentials:
- Download your service account key file
- Set the path in GOOGLE_APPLICATION_CREDENTIALS

## Testing the System

Run the full session system test:
```bash
python test_full_session_system.py
```

Test the strategy generation:
```bash
python test_strategy.py
```

## Architecture

The application consists of several components:

- `session_manager_v2.py`: Manages query session state with append-only pattern
- `bigquery_client.py`: Handles BigQuery interactions
- `gemini_client.py`: Manages Vertex AI/Gemini interactions
- `query_processor.py`: Processes queries and generates SQL
- `main.py`: Main application coordinator

## GitHub Workflow

This repository uses GitHub Actions for CI/CD. The workflow includes:

1. Running tests on pull requests
2. Linting code for quality
3. Deploying to development/production environments

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 