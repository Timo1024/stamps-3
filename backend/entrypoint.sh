#!/bin/bash
set -e

# Force database reset if requested
if [ "$RESET_DB" = "true" ]; then
    echo "Resetting database schema..."
    python -c "from app.database import Base, engine; Base.metadata.drop_all(bind=engine); Base.metadata.create_all(bind=engine)"
fi

# Run database population script if requested
if [ "$POPULATE_DB" = "true" ]; then
    echo "Populating database..."
    python -m app.populate_tables
fi

# Start the FastAPI application
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
