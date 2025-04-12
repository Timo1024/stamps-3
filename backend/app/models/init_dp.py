from app.__init__ import engine
from app.models import Base

# Import all models to register them with the Base
from app.models.stamps import Stamp
from app.models.sets import Set
from app.models.users import User
from app.models.user_stamps import UserStamp
from app.models.themes import Theme
from app.models.stamp_themes import StampTheme
from app.models.images import Image
from app.models.stamp_images import StampImage
from app.models.colors import Color
from app.models.stamp_colors import StampColor

# Create the tables in the database
Base.metadata.create_all(bind=engine)
