import argparse
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.database import SessionLocal, UserDB, create_db_and_tables
from api.auth import get_password_hash
from core.logging_utils import get_logger

logger = get_logger(__name__)

def create_user(username: str, password: str, is_disabled: bool = False):
    """Creates a new user in the database."""
    logger.info(f"Attempting to create user: {username}")

    # Ensure the database and tables exist
    create_db_and_tables()

    db = SessionLocal()
    try:
        # Check if user already exists
        existing_user = db.query(UserDB).filter(UserDB.username == username).first()
        if existing_user:
            logger.warning(f"User '{username}' already exists. Aborting.")
            return

        # Hash the password
        hashed_password = get_password_hash(password)

        # Create new user instance
        new_user = UserDB(
            username=username,
            hashed_password=hashed_password,
            disabled=is_disabled
        )

        # Add to session and commit
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        logger.info(f"Successfully created user: {username}")
        print(f"User '{username}' created successfully.")

    except Exception as e:
        logger.error(f"Failed to create user '{username}': {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new user for the RAG API.")
    parser.add_argument("username", type=str, help="The username for the new user.")
    parser.add_argument("password", type=str, help="The password for the new user.")
    parser.add_argument("--disabled", action="store_true", help="Create the user in a disabled state.")

    args = parser.parse_args()

    create_user(args.username, args.password, args.disabled)
