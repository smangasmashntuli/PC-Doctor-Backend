#setup.py
from sqlalchemy.orm import Session

from backend.app.models import LaptopSetup, Users
from backend.app.fetch import LaptopSetupCreate


def get_laptop_setup_by_user_id(db: Session, user_id: int):
    return db.query(LaptopSetup).filter(LaptopSetup.user_id == user_id).first()


def create_laptop_setup(db: Session, user: Users, laptop: LaptopSetupCreate):
    existing_setup = get_laptop_setup_by_user_id(db, user.id)
    if existing_setup:
        existing_setup.brand = laptop.brand
        existing_setup.model = laptop.model
        db.commit()
        db.refresh(existing_setup)
        return existing_setup

    setup_record = LaptopSetup(
        user_id=user.id,
        brand=laptop.brand,
        model=laptop.model,
    )
    db.add(setup_record)
    db.commit()
    db.refresh(setup_record)
    return setup_record
