#crud.py
from sqlalchemy.orm import Session
from backend.app.models import Users
from backend.app.fetch import UserCreate
from backend.app.auth import get_password_hash

def get_user_by_email(db: Session, email: str):
    return db.query(Users).filter(Users.email == email).first()

def get_user_by_name(db: Session, name: str):
    return db.query(Users).filter(Users.name == name).first()

def create_user(db: Session, user: UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = Users(
        email=user.email,
        name=user.name,
        surname=user.surname,
        hashed_password=hashed_password
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Users).offset(skip).limit(limit).all()