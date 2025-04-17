import os
import sys
import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Correct import path for Docker environment
from app.models import Base, Set, Stamp, Theme, Image, Color, User, UserStamp, StampTheme, StampImage, StampColor
from app.database import get_db

# Use an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def db():
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Create a session for testing
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
    
    # Drop tables after tests
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def sample_set(db):
    test_set = Set(
        setid=1,
        country="Test Country", 
        category="Test Category",
        year=2023,
        url="https://example.com/set",
        name="Test Set",
        set_description="A test set"
    )
    db.add(test_set)
    db.commit()
    return test_set

@pytest.fixture
def sample_stamp(db, sample_set):
    test_stamp = Stamp(
        stampid=1,
        number="TS-001",
        type="Regular",
        denomination="5.00",
        color="Blue",
        description="Test stamp",
        date_of_issue=date(2023, 1, 1),
        setid=sample_set.setid
    )
    db.add(test_stamp)
    db.commit()
    return test_stamp

@pytest.fixture
def sample_theme(db):
    test_theme = Theme(themeid=1, name="Test Theme")
    db.add(test_theme)
    db.commit()
    return test_theme

@pytest.fixture
def sample_image(db):
    test_image = Image(pathid=1, path="/images/test.jpg")
    db.add(test_image)
    db.commit()
    return test_image

@pytest.fixture
def sample_color(db):
    test_color = Color(colorid=1, name="Blue")
    db.add(test_color)
    db.commit()
    return test_color

@pytest.fixture
def sample_user(db):
    test_user = User(
        userid=1,
        username="testuser",
        email="test@example.com",
        password_hash="hashed_password"
    )
    db.add(test_user)
    db.commit()
    return test_user

def test_set_crud(db):
    # Create
    new_set = Set(
        setid=100,
        country="Sweden", 
        category="Commemorative",
        year=2022,
        name="Birds of Sweden"
    )
    db.add(new_set)
    db.commit()
    
    # Read
    retrieved_set = db.query(Set).filter(Set.setid == 100).first()
    assert retrieved_set is not None
    assert retrieved_set.country == "Sweden"
    assert retrieved_set.year == 2022
    
    # Update
    retrieved_set.year = 2023
    db.commit()
    updated_set = db.query(Set).filter(Set.setid == 100).first()
    assert updated_set.year == 2023
    
    # Delete
    db.delete(retrieved_set)
    db.commit()
    deleted_set = db.query(Set).filter(Set.setid == 100).first()
    assert deleted_set is None

def test_stamp_crud(db, sample_set):
    # Create
    new_stamp = Stamp(
        stampid=100,
        number="SE-001",
        type="Commemorative",
        denomination="10.00",
        color="Red",
        date_of_issue=date(2022, 6, 15),
        setid=sample_set.setid
    )
    db.add(new_stamp)
    db.commit()
    
    # Read
    retrieved_stamp = db.query(Stamp).filter(Stamp.stampid == 100).first()
    assert retrieved_stamp is not None
    assert retrieved_stamp.number == "SE-001"
    
    # Update
    retrieved_stamp.denomination = "12.50"
    db.commit()
    updated_stamp = db.query(Stamp).filter(Stamp.stampid == 100).first()
    assert updated_stamp.denomination == "12.50"
    
    # Delete
    db.delete(retrieved_stamp)
    db.commit()
    deleted_stamp = db.query(Stamp).filter(Stamp.stampid == 100).first()
    assert deleted_stamp is None

def test_theme_crud(db):
    # Create
    new_theme = Theme(themeid=100, name="Birds")
    db.add(new_theme)
    db.commit()
    
    # Read
    retrieved_theme = db.query(Theme).filter(Theme.themeid == 100).first()
    assert retrieved_theme is not None
    assert retrieved_theme.name == "Birds"
    
    # Update
    retrieved_theme.name = "Birds of Prey"
    db.commit()
    updated_theme = db.query(Theme).filter(Theme.themeid == 100).first()
    assert updated_theme.name == "Birds of Prey"
    
    # Delete
    db.delete(retrieved_theme)
    db.commit()
    deleted_theme = db.query(Theme).filter(Theme.themeid == 100).first()
    assert deleted_theme is None

def test_set_stamp_relationship(db, sample_set):
    # Create stamps within the set
    stamp1 = Stamp(
        stampid=201,
        number="TS-201",
        type="Regular",
        denomination="5.00",
        setid=sample_set.setid
    )
    
    stamp2 = Stamp(
        stampid=202,
        number="TS-202",
        type="Regular",
        denomination="10.00",
        setid=sample_set.setid
    )
    
    db.add_all([stamp1, stamp2])
    db.commit()
    
    # Test relationship
    retrieved_set = db.query(Set).filter(Set.setid == sample_set.setid).first()
    assert len(retrieved_set.stamps) == 2
    assert retrieved_set.stamps[0].number in ["TS-201", "TS-202"]
    assert retrieved_set.stamps[1].number in ["TS-201", "TS-202"]
    
    # Test cascade delete
    db.delete(retrieved_set)
    db.commit()
    
    # Check if stamps are also deleted
    stamps = db.query(Stamp).filter(
        Stamp.stampid.in_([201, 202])
    ).all()
    assert len(stamps) == 0

def test_stamp_theme_relationship(db, sample_stamp, sample_theme):
    # Create relationship
    stamp_theme = StampTheme(
        stampid=sample_stamp.stampid,
        themeid=sample_theme.themeid
    )
    db.add(stamp_theme)
    db.commit()
    
    # Test relationship from stamp
    retrieved_stamp = db.query(Stamp).filter(Stamp.stampid == sample_stamp.stampid).first()
    assert len(retrieved_stamp.themes) == 1
    assert retrieved_stamp.themes[0].name == sample_theme.name
    
    # Test relationship from theme
    retrieved_theme = db.query(Theme).filter(Theme.themeid == sample_theme.themeid).first()
    assert len(retrieved_theme.stamps) == 1
    assert retrieved_theme.stamps[0].stampid == sample_stamp.stampid
    
    # Test removing relationship
    db.delete(stamp_theme)
    db.commit()
    
    # Verify relationship is removed
    retrieved_stamp = db.query(Stamp).filter(Stamp.stampid == sample_stamp.stampid).first()
    assert len(retrieved_stamp.themes) == 0

def test_stamp_image_relationship(db, sample_stamp, sample_image):
    # Create relationship
    stamp_image = StampImage(
        stampid=sample_stamp.stampid,
        pathid=sample_image.pathid
    )
    db.add(stamp_image)
    db.commit()
    
    # Test relationship from stamp
    retrieved_stamp = db.query(Stamp).filter(Stamp.stampid == sample_stamp.stampid).first()
    assert len(retrieved_stamp.images) == 1
    assert retrieved_stamp.images[0].path == sample_image.path
    
    # Test relationship from image
    retrieved_image = db.query(Image).filter(Image.pathid == sample_image.pathid).first()
    assert len(retrieved_image.stamps) == 1
    assert retrieved_image.stamps[0].stampid == sample_stamp.stampid

def test_stamp_color_relationship(db, sample_stamp, sample_color):
    # Create relationship
    stamp_color = StampColor(
        stampid=sample_stamp.stampid,
        colorid=sample_color.colorid
    )
    db.add(stamp_color)
    db.commit()
    
    # Test relationship from stamp
    retrieved_stamp = db.query(Stamp).filter(Stamp.stampid == sample_stamp.stampid).first()
    assert len(retrieved_stamp.colors) == 1
    assert retrieved_stamp.colors[0].name == sample_color.name
    
    # Test relationship from color
    retrieved_color = db.query(Color).filter(Color.colorid == sample_color.colorid).first()
    assert len(retrieved_color.stamps) == 1
    assert retrieved_color.stamps[0].stampid == sample_stamp.stampid

def test_user_stamp_relationship(db, sample_user, sample_stamp):
    # Create relationship with collection data
    user_stamp = UserStamp(
        userid=sample_user.userid,
        stampid=sample_stamp.stampid,
        amount_unused=2,
        amount_minted=1,
        note="From my grandfather's collection"
    )
    db.add(user_stamp)
    db.commit()
    
    # Test retrieving the user's collection
    retrieved_user = db.query(User).filter(User.userid == sample_user.userid).first()
    assert len(retrieved_user.user_stamps) == 1
    assert retrieved_user.user_stamps[0].amount_unused == 2
    assert retrieved_user.user_stamps[0].stamp.stampid == sample_stamp.stampid
    
    # Test updating collection
    user_stamp.amount_unused = 3
    db.commit()
    
    # Verify update
    retrieved_user_stamp = db.query(UserStamp).filter(
        UserStamp.userid == sample_user.userid,
        UserStamp.stampid == sample_stamp.stampid
    ).first()
    assert retrieved_user_stamp.amount_unused == 3

def test_load_data(db):
    """Test loading sample data into database"""
    # Create sample data
    test_set = Set(setid=500, country="Finland", year=2020, name="Finnish Wildlife")
    db.add(test_set)
    
    test_stamp = Stamp(
        stampid=500,
        number="FI-001",
        denomination="2.50",
        setid=test_set.setid
    )
    db.add(test_stamp)
    
    test_theme = Theme(themeid=500, name="Wildlife")
    db.add(test_theme)
    
    # Create relationships
    stamp_theme = StampTheme(stampid=test_stamp.stampid, themeid=test_theme.themeid)
    db.add(stamp_theme)
    
    db.commit()
    
    # Verify data is loaded
    assert db.query(Set).count() >= 1
    assert db.query(Stamp).count() >= 1
    assert db.query(Theme).count() >= 1
    
    # Verify relationships
    stamp = db.query(Stamp).filter(Stamp.stampid == 500).first()
    assert len(stamp.themes) == 1
    assert stamp.themes[0].name == "Wildlife"
