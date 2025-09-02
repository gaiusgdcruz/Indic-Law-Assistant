from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .. import auth, config, schemas

router = APIRouter(
    prefix="/token",
    tags=["Authentication"],
)

@router.post("", response_model=schemas.Token)
async def login_for_access_token(
    form_data: dict, db: Session = Depends(auth.get_db)
):
    """
    Authenticate user and return an access token.
    Uses form data for username and password.
    """
    username = form_data.get("username")
    password = form_data.get("password")

    user = auth.authenticate_user(db, username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    refresh_token = auth.create_refresh_token(data={"sub": user.username})

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


@router.post("/refresh", response_model=schemas.Token)
async def refresh_access_token(
    payload: dict, db: Session = Depends(auth.get_db)
):
    """
    Refresh the access token using a valid refresh token.
    """
    from jose import JWTError, jwt

    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=400, detail="Missing refresh token")

    try:
        token_payload = jwt.decode(
            refresh_token, config.SECRET_KEY, algorithms=[config.ALGORITHM]
        )
        username: str = token_payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=401, detail="Invalid refresh token (no subject)"
            )
        user = auth.get_user(db, username)
        if user is None:
            raise HTTPException(status_code=401, detail="User from token not found")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    new_access_token = auth.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    new_refresh_token = auth.create_refresh_token(data={"sub": user.username})

    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
    }
