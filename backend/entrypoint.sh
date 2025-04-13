#!/bin/bash
set -e

# Force database reset if requested
if [ "$RESET_DB" = "true" ]; then
    echo "Resetting database schema..."
    
    # Print diagnostic information
    echo "Checking database connection..."
    python -c "from app.database import engine; print(f'Connected to database: {engine.url}')"
    
    # Drop and recreate all tables
    echo "Dropping all tables..."
    python -c "from app.database import Base, engine; Base.metadata.drop_all(bind=engine)"
    
    echo "Creating new tables according to models..."
    # Import all models explicitly to ensure they're registered with Base
    python -c "from app.database import Base, engine; from app.models import Set, Stamp, User, UserStamp, Theme, StampTheme, Image, StampImage, Color, StampColor; Base.metadata.create_all(bind=engine)"
    
    # Add a brief pause to ensure tables are created
    sleep 2
    
    echo "Verifying tables in database..."
    python -c "from app.database import engine; import pandas as pd; print(pd.read_sql('SHOW TABLES', engine))"
    
    # Only check sets table if it exists
    python -c "from app.database import engine; import pandas as pd; tables = pd.read_sql('SHOW TABLES', engine); print('Sets table exists:' if 'sets' in [t[0].lower() for t in tables.values] else 'Sets table does not exist yet')"
fi

# Run database population script if requested
if [ "$POPULATE_DB" = "true" ]; then
    echo "Populating database..."
    python -m app.populate_tables
    
    # Force a final verification that data exists and is accessible
    echo "Verifying data was populated correctly..."
    python -c "from app.database import engine, SessionLocal; from app.models import Set, Stamp, Theme, Image, Color; db = SessionLocal(); print(f'Sets count: {db.query(Set).count()}'); print(f'Stamps count: {db.query(Stamp).count()}'); print(f'Themes count: {db.query(Theme).count()}'); print(f'Images count: {db.query(Image).count()}'); print(f'Colors count: {db.query(Color).count()}'); db.close()"
    
    # Ensure database connection is truly closed
    echo "Closing all database connections..."
    python -c "from app.database import engine; engine.dispose()"
fi

# Start the FastAPI application
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
