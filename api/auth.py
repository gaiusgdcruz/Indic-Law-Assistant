import os
# hashlib is no longer used for password hashing directly in this scheme,
# it was for the previous client-side SHA256 hashing.
# import hashlib 
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from .schemas import TokenData, User, UserInDB
from . import config

load_dotenv()

# Security configuration
SECRET_KEY = config.SECRET_KEY
ALGORITHM = config.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = config.ACCESS_TOKEN_EXPIRE_MINUTES

# Password hashing using bcrypt, but note that the stored password is the bcrypt hash of the SHA-256 digest.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# In-memory user database (replace with proper database in production)
# Passwords are now hashed server-side using bcrypt directly on the raw password.
fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),  # bcrypt hash of the raw password "admin123"
        "disabled": False,
    }
}

# Client should send raw passwords. Server hashes them for storage (if applicable) and comparison.
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a stored hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def get_user(db, username: str):
    """Retrieve a user from the database by username."""
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(fake_db, username: str, plain_password: str):
    """
    Authenticate a user.
    Expects `plain_password` to be the raw password from the client.
    Server-side hashing (bcrypt) is used for comparison.
    """
    user = get_user(fake_db, username)
    if not user:
        return False
    # The stored password is user.hashed_password (already bcrypt hashed)
    # The password from the form is plain_password (raw)
    if not verify_password(plain_password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create an access token with an expiration time."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a refresh token with a longer expiration time (default 7 days)."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Retrieve the current user from the access token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Retrieve the current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
