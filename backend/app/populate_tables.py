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
    count = 0
    try:
        # Open the CSV file
        print(f"Reading sets from app/csv_data/sets.csv")
        with open("app/csv_data/sets.csv", "r", encoding='utf-8') as file:
            reader = csv.DictReader(file)  # Automatically maps column names to dict keys
            for row in reader:
                try:
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
                    count += 1
                    if count % 100 == 0:
                        # Commit in batches to avoid memory issues
                        db.commit()
                        print(f"  - {count} sets processed")
                except Exception as row_error:
                    print(f"Error processing set row: {row_error}")
                    print(f"Row data: {row}")

        db.commit()  # Commit all changes
        print(f"Sets table populated successfully with {count} records!")
        
        # Verify data was inserted
        set_count = db.query(Set).count()
        print(f"Total sets in database: {set_count}")
    except Exception as e:
        print(f"Error populating sets table: {e}")
        db.rollback()
    finally:
        db.close()

def populate_stamps():
    db = SessionLocal()
    count = 0
    try:
        print(f"Reading stamps from app/csv_data/stamps.csv")
        with open("app/csv_data/stamps.csv", "r", encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            # Helper functions (moved outside the loop for efficiency)
            def safe_float(value, default=0.0):
                try:
                    return float(value) if value and value.strip() else default
                except (ValueError, AttributeError):
                    return default
            
            def safe_int(value, default=0):
                try:
                    return int(value) if value and value.strip() else default
                except (ValueError, AttributeError):
                    return default
            
            for row in reader:
                try:
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
                        date_of_issue=row["date_of_issue"] if row["date_of_issue"] else None,
                        perforations=row["perforations"],
                        sheet_size=row["sheet_size"],
                        designed=row["designed"],
                        engraved=row["engraved"],
                        height_width=row["height_width"],
                        image_accuracy=safe_int(row["image_accuracy"]),
                        perforation_horizontal=safe_float(row["perforation_horizontal"]),
                        perforation_vertical=safe_float(row["perforation_vertical"]),
                        perforation_keyword=row["perforation_keyword"],
                        value_from=safe_float(row["value_from"]),
                        value_to=safe_float(row["value_to"]),
                        number_issued=safe_int(row["number_issued"]),
                        mint_condition_float=safe_float(row["mint_condition_float"]),
                        unused_float=safe_float(row["unused_float"]),
                        used_float=safe_float(row["used_float"]),
                        letter_fdc_float=safe_float(row["letter_fdc_float"]),
                        sheet_size_amount=safe_float(row["sheet_size_amount"]),
                        sheet_size_x=safe_float(row["sheet_size_x"]),
                        sheet_size_y=safe_float(row["sheet_size_y"]),
                        sheet_size_note=row["sheet_size_note"],
                        height=safe_float(row["height"]),
                        width=safe_float(row["width"]),
                        setid=safe_int(row["setid"]),  # Foreign key reference to Set
                    )
                    db.add(new_stamp)
                    count += 1
                    if count % 100 == 0:
                        # Commit in batches to avoid memory issues
                        db.commit()
                        print(f"  - {count} stamps processed")
                except Exception as row_error:
                    print(f"Error processing stamp row: {row_error}")
                    print(f"Row data: {row}")

        db.commit()
        print(f"Stamps table populated successfully with {count} records!")
        
        # Verify data was inserted
        stamp_count = db.query(Stamp).count()
        print(f"Total stamps in database: {stamp_count}")
    except Exception as e:
        print(f"Error populating stamps table: {e}")
        db.rollback()
    finally:
        db.close()

def populate_themes():
    db = SessionLocal()
    count = 0
    try:
        print(f"Reading themes from app/csv_data/themes.csv")
        with open("app/csv_data/themes.csv", "r", encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    new_theme = Theme(
                        name=row["name"],
                    )
                    db.add(new_theme)
                    count += 1
                except Exception as row_error:
                    print(f"Error processing theme row: {row_error}")

        db.commit()
        print(f"Themes table populated successfully with {count} records!")
        
        # Verify data was inserted
        theme_count = db.query(Theme).count()
        print(f"Total themes in database: {theme_count}")
    except Exception as e:
        print(f"Error populating themes table: {e}")
        db.rollback()
    finally:
        db.close()

def populate_images():
    db = SessionLocal()
    count = 0
    try:
        print(f"Reading images from app/csv_data/images.csv")
        with open("app/csv_data/images.csv", "r", encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    new_image = Image(
                        path=row["path"],
                    )
                    db.add(new_image)
                    count += 1
                    if count % 100 == 0:
                        # Commit in batches to avoid memory issues
                        db.commit()
                        print(f"  - {count} images processed")
                except Exception as row_error:
                    print(f"Error processing image row: {row_error}")

        db.commit()
        print(f"Images table populated successfully with {count} records!")
        
        # Verify data was inserted
        image_count = db.query(Image).count()
        print(f"Total images in database: {image_count}")
    except Exception as e:
        print(f"Error populating images table: {e}")
        db.rollback()
    finally:
        db.close()

def populate_colors():
    db = SessionLocal()
    count = 0
    try:
        print(f"Reading colors from app/csv_data/colors.csv")
        with open("app/csv_data/colors.csv", "r", encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    new_color = Color(
                        name=row["name"],
                    )
                    db.add(new_color)
                    count += 1
                except Exception as row_error:
                    print(f"Error processing color row: {row_error}")

        db.commit()
        print(f"Colors table populated successfully with {count} records!")
        
        # Verify data was inserted
        color_count = db.query(Color).count()
        print(f"Total colors in database: {color_count}")
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
