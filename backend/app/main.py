from fastapi import FastAPI
from app.routes.stamps import router as stamps_router

app = FastAPI()

app.include_router(stamps_router, prefix="/api")
