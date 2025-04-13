import os
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Improve connection pooling settings
engine = create_engine(
    DATABASE_URL, 
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    pool_pre_ping=True,
    pool_timeout=30,
    echo=True  # Enable SQL logging
)

# Debug: output the connection URL
print("Engine created with URL:", DATABASE_URL)

# Add connection event listeners
@event.listens_for(engine, "connect")
def on_connect(dbapi_con, connection_record):
    print(f"New database connection established: {dbapi_con}")

@event.listens_for(engine, "checkout")
def on_checkout(dbapi_con, connection_record, connection_proxy):
    print(f"Database connection checkout: {dbapi_con}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get DB session
def get_db():
    print("Creating new database session")
    db = SessionLocal()
    try:
        # Force validation of connection - using text() to wrap raw SQL
        db.execute(text("SELECT 1"))
        print("Database connection validated")
        yield db
    except Exception as e:
        print(f"Database connection error: {e}")
        raise
    finally:
        print("Closing database session")
        db.close()
