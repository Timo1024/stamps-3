import os
from fastapi import FastAPI
import subprocess
import sys

from . import models
from .database import engine
from .routes import base, sets, stamps

# Only reset database if explicitly requested and not during normal startup
reset_db = os.getenv("RESET_DB", "").lower() == "true"
populate_db = os.getenv("POPULATE_DB", "").lower() == "true"

if reset_db:
    print("RESET_DB=true detected. Dropping all tables and recreating schema...")
    models.Base.metadata.drop_all(bind=engine)
    models.Base.metadata.create_all(bind=engine)
    
    # When tables are reset, automatically set populate_db to True
    populate_db = True
    print("Database tables have been reset. Will populate data automatically.")
else:
    print("RESET_DB not set to true. Preserving existing database tables.")
    # Only create tables that don't exist, don't drop anything
    models.Base.metadata.create_all(bind=engine)

# Populate database if needed
if populate_db:
    print("Populating database with data...")
    try:
        # Import and run the populate_tables module directly
        from .populate_tables import (
            populate_sets, 
            populate_stamps, 
            populate_themes, 
            populate_images,
            populate_colors
        )
        
        print("Running population functions...")
        populate_sets()
        populate_stamps()
        populate_themes()
        populate_images()
        populate_colors()
        print("Database population completed!")
    except Exception as e:
        print(f"Error populating database: {e}", file=sys.stderr)
        print(f"Detailed error: {type(e).__name__}: {str(e)}", file=sys.stderr)

app = FastAPI(title="Stamps API", description="API for stamp collectors")

# Include routers from route modules
app.include_router(base.router)
app.include_router(sets.router)
app.include_router(stamps.router)
