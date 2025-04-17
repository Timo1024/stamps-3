from fastapi import FastAPI
import os

from . import models
from .database import engine
from .routes import base, sets, stamps
from . import data_init

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Stamps API", description="API for stamp collectors")

@app.on_event("startup")
async def startup_db_client():
    """Initialize database on application startup"""
    print("Application startup: Initializing database")
    data_init.initialize_data()

# Include routers from route modules
app.include_router(base.router)
app.include_router(sets.router)
app.include_router(stamps.router)
