#!/bin/bash
set -e

# Run database population script if requested
if [ "$POPULATE_DB" = "true" ]; then
    echo "Populating database..."
    python -m app.populate_tables
fi

# Start the FastAPI application
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
