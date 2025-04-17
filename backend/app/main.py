from fastapi import FastAPI
import os

from . import models
from .database import engine
from .routes import base, sets, stamps
from . import data_init  # Import the new module

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Initialize data if needed
data_init.initialize_data()

app = FastAPI(title="Stamps API", description="API for stamp collectors")

# Include routers from route modules
app.include_router(base.router)
app.include_router(sets.router)
app.include_router(stamps.router)
