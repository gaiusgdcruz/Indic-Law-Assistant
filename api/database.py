import os
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from core.logging_utils import get_logger

logger = get_logger(__name__)

# Define the database URL. Default to a sqlite file in the `api` directory.
# For production, this should be a more robust database like PostgreSQL.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./api/sql_app.db")

# Create the SQLAlchemy engine
# The check_same_thread argument is needed only for SQLite.
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for our models
Base = declarative_base()


# --- User Model ---
class UserDB(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    disabled = Column(Boolean, default=False)


def create_db_and_tables():
    """
    Creates the database and all tables defined in the Base metadata.
    This should be called once on application startup.
    """
    logger.info("Checking and creating database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully (if they didn't exist).")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}", exc_info=True)
        raise


def get_db():
    """
    Dependency to get a database session.
    Ensures the database session is always closed after the request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
