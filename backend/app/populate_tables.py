import csv
import sys
import os

# Check if DATABASE_URL is set
database_url = os.getenv("DATABASE_URL")
if not database_url:
    print("ERROR: DATABASE_URL environment variable is not set.")
    sys.exit(1)

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import using the absolute path
from app.database import SessionLocal, Base
from app.models import Set, Stamp, Theme, Image, Color

def populate_sets():
    db = SessionLocal()
    try:
        # Open the CSV file
        with open("app/csv_data/sets.csv", "r") as file:
            reader = csv.DictReader(file)  # Automatically maps column names to dict keys
            for row in reader:
                new_set = Set(
                    setid=int(row["setid"]),
                    country=row["country"],
                    category=row["category"],
                    year=int(row["year"]),
                    url=row["url"],
                    name=row["name"],
                    description=row["description"],
                )
                db.add(new_set)  # Add the set to the database session

        db.commit()  # Commit all changes
        print("Sets table populated successfully!")
    except Exception as e:
        print(f"Error populating sets table: {e}")
        db.rollback()
    finally:
        db.close()

def populate_stamps():
    db = SessionLocal()
    try:
        with open("app/csv_data/stamps.csv", "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                new_stamp = Stamp(
                    number=row["number"],
                    type=row["type"],
                    denomination=row["denomination"],
                    color=row["color"],
                    description=row["description"],
                    stamps_issued=row["stamps_issued"],
                    mint_condition=row["mint_condition"],
                    unused=row["unused"],
                    used=row["used"],
                    letter_fdc=row["letter_fdc"],
                    date_of_issue=row["date_of_issue"],
                    perforations=row["perforations"],
                    sheet_size=row["sheet_size"],
                    designed=row["designed"],
                    engraved=row["engraved"],
                    height_width=row["height_width"],
                    image_accuracy=int(row["image_accuracy"]),
                    perforation_horizontal=float(row["perforation_horizontal"]),
                    perforation_vertical=float(row["perforation_vertical"]),
                    perforation_keyword=row["perforation_keyword"],
                    value_from=float(row["value_from"]),
                    value_to=float(row["value_to"]),
                    number_issued=int(row["number_issued"]),
                    mint_condition_float=float(row["mint_condition_float"]),
                    unused_float=float(row["unused_float"]),
                    used_float=float(row["used_float"]),
                    letter_fdc_float=float(row["letter_fdc_float"]),
                    sheet_size_amount=float(row["sheet_size_amount"]),
                    sheet_size_x=float(row["sheet_size_x"]),
                    sheet_size_y=float(row["sheet_size_y"]),
                    sheet_size_note=row["sheet_size_note"],
                    height=float(row["height"]),
                    width=float(row["width"]),
                    setid=int(row["setid"]),  # Foreign key reference to Set
                )
                db.add(new_stamp)

        db.commit()
        print("Stamps table populated successfully!")
    except Exception as e:
        print(f"Error populating stamps table: {e}")
        db.rollback()
    finally:
        db.close()

def populate_themes():
    db = SessionLocal()
    try:
        with open("app/csv_data/themes.csv", "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                new_theme = Theme(
                    name=row["name"],
                )
                db.add(new_theme)

        db.commit()
        print("Themes table populated successfully!")
    except Exception as e:
        print(f"Error populating themes table: {e}")
        db.rollback()
    finally:
        db.close()

def populate_images():
    db = SessionLocal()
    try:
        with open("app/csv_data/images.csv", "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                new_image = Image(
                    path=row["path"],
                )
                db.add(new_image)

        db.commit()
        print("Images table populated successfully!")
    except Exception as e:
        print(f"Error populating images table: {e}")
        db.rollback()
    finally:
        db.close()

def populate_colors():
    db = SessionLocal()
    try:
        with open("app/csv_data/colors.csv", "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                new_color = Color(
                    name=row["name"],
                )
                db.add(new_color)

        db.commit()
        print("Colors table populated successfully!")
    except Exception as e:
        print(f"Error populating colors table: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    populate_sets()
    populate_stamps()
    populate_themes()
    populate_images()
    populate_colors()
