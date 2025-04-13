import os
from fastapi import FastAPI

from . import models
from .database import engine
from .routes import base, sets, stamps

# Check if we should reset the database
if os.getenv("RESET_DB", "").lower() == "true":
    print("Dropping all tables and recreating schema...")
    models.Base.metadata.drop_all(bind=engine)

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Stamps API", description="API for stamp collectors")

# Include routers from route modules
app.include_router(base.router)
app.include_router(sets.router)
app.include_router(stamps.router)
