from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, distinct, and_, or_
from sqlalchemy.orm import Session, joinedload
from typing import List, Dict, Any, Optional
from datetime import date

from .. import models, schemas
from ..database import get_db

router = APIRouter(
    prefix="/stamps",
    tags=["stamps"]
)


def build_filtered_query(
    db: Session,
    # Stamp filters
    type: Optional[str] = None,
    color: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    denomination: Optional[str] = None,
    # Set filters
    setid: Optional[int] = None,
    country: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    category: Optional[str] = None,
    # Theme filter
    theme: Optional[str] = None,
    # User ownership filter
    user_id: Optional[int] = None,
    owned_only: bool = False
):
    """
    Build a query with filters applied based on parameters.
    Returns the filtered query object that can be further processed.
    """
    # Start with the base query
    query = db.query(models.Stamp)

    # Apply direct stamp filters
    if type:
        query = query.filter(models.Stamp.type == type)
    if color:
        query = query.filter(models.Stamp.color.ilike(f"%{color}%"))
    if date_from:
        query = query.filter(models.Stamp.date_of_issue >= date_from)
    if date_to:
        query = query.filter(models.Stamp.date_of_issue <= date_to)
    if denomination:
        query = query.filter(models.Stamp.denomination == denomination)

    # Apply set-related filters
    if any([setid, country, year_from, year_to, category]):
        # Join with sets table if any set filter is applied
        query = query.join(models.Set, models.Stamp.setid == models.Set.setid)

        if setid:
            query = query.filter(models.Stamp.setid == setid)
        if country:
            query = query.filter(models.Set.country == country)
        if year_from:
            query = query.filter(models.Set.year >= year_from)
        if year_to:
            query = query.filter(models.Set.year <= year_to)
        if category:
            query = query.filter(models.Set.category == category)

    # Apply theme filter
    if theme:
        query = query.join(
            models.StampTheme,
            models.Stamp.stampid == models.StampTheme.stampid
        ).join(
            models.Theme,
            models.StampTheme.themeid == models.Theme.themeid
        ).filter(models.Theme.name.ilike(f"%{theme}%"))

    # Apply user ownership filter
    if user_id:
        if owned_only:
            # Only return stamps owned by the user
            query = query.join(
                models.UserStamp,
                and_(
                    models.Stamp.stampid == models.UserStamp.stampid,
                    models.UserStamp.userid == user_id
                )
            )
        else:
            # Include ownership information but don't filter out unowned stamps
            query = query.outerjoin(
                models.UserStamp,
                and_(
                    models.Stamp.stampid == models.UserStamp.stampid,
                    models.UserStamp.userid == user_id
                )
            )

    return query


@router.get("/filter", response_model=List[schemas.Stamp])
def filter_stamps(
    # Stamp filters
    type: Optional[str] = None,
    color: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    denomination: Optional[str] = None,
    # Set filters
    setid: Optional[int] = None,
    country: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    category: Optional[str] = None,
    # Theme filter
    theme: Optional[str] = None,
    # User ownership filter
    user_id: Optional[int] = None,
    owned_only: bool = False,
    # Pagination
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Filter stamps based on query parameters"""
    query = build_filtered_query(
        db, type, color, date_from, date_to, denomination,
        setid, country, year_from, year_to, category,
        theme, user_id, owned_only
    )

    return query.offset(skip).limit(limit).all()


@router.get("/stats", response_model=Dict[str, Any])
def get_stats(
    # Stamp filters
    type: Optional[str] = None,
    color: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    denomination: Optional[str] = None,
    # Set filters
    setid: Optional[int] = None,
    country: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    category: Optional[str] = None,
    # Theme filter
    theme: Optional[str] = None,
    # User ownership filter
    user_id: Optional[int] = None,
    owned_only: bool = False,
    db: Session = Depends(get_db)
):
    """Get statistics on stamps matching the filter criteria"""
    # First, build the filtered query
    query = build_filtered_query(
        db, type, color, date_from, date_to, denomination,
        setid, country, year_from, year_to, category,
        theme, user_id, owned_only
    )

    # Count total stamps matching the filter
    total_stamps = query.count()

    # Get stats with joined set data for counting countries
    if total_stamps > 0:
        # Count unique countries
        countries_query = db.query(func.count(distinct(models.Set.country))).join(
            models.Stamp, models.Set.setid == models.Stamp.setid
        ).filter(models.Stamp.stampid.in_(
            query.with_entities(models.Stamp.stampid)
        ))
        unique_countries = countries_query.scalar() or 0

        # Count unique types
        types_query = db.query(func.count(distinct(models.Stamp.type))).filter(
            models.Stamp.stampid.in_(query.with_entities(models.Stamp.stampid))
        )
        unique_types = types_query.scalar() or 0

        # Get min and max dates
        date_query = db.query(
            func.min(models.Stamp.date_of_issue),
            func.max(models.Stamp.date_of_issue)
        ).filter(
            models.Stamp.stampid.in_(query.with_entities(models.Stamp.stampid))
        )
        min_date, max_date = date_query.first()

        # Count by theme (top 5)
        theme_counts = db.query(
            models.Theme.name, func.count(models.Stamp.stampid).label('count')
        ).join(
            models.StampTheme,
            models.Theme.themeid == models.StampTheme.themeid
        ).join(
            models.Stamp,
            models.StampTheme.stampid == models.Stamp.stampid
        ).filter(
            models.Stamp.stampid.in_(query.with_entities(models.Stamp.stampid))
        ).group_by(
            models.Theme.name
        ).order_by(
            func.count(models.Stamp.stampid).desc()
        ).limit(5).all()

        theme_stats = [{"name": name, "count": count}
                       for name, count in theme_counts]

    else:
        unique_countries = 0
        unique_types = 0
        min_date = None
        max_date = None
        theme_stats = []

    return {
        "total_stamps": total_stamps,
        "unique_countries": unique_countries,
        "unique_types": unique_types,
        "date_range": {
            "min_date": min_date,
            "max_date": max_date
        },
        "top_themes": theme_stats
    }


@router.get("/date_range", response_model=Dict[str, Any])
def get_date_range(db: Session = Depends(get_db)):
    # Get min and max date_of_issue for all stamps (no filters)
    date_query = db.query(
        func.min(models.Stamp.date_of_issue),
        func.max(models.Stamp.date_of_issue)
    )
    min_date, max_date = date_query.first()

    print(f"Min date: {min_date}, Max date: {max_date}")

    return {
        "min_date": min_date,
        "max_date": max_date
    }


@router.get("/", response_model=List[schemas.Stamp])
def read_stamps(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    stamps = db.query(models.Stamp).offset(skip).limit(limit).all()
    return stamps


@router.get("/{stamp_id}", response_model=schemas.Stamp)
def read_stamp(stamp_id: int, db: Session = Depends(get_db)):
    stamp = db.query(models.Stamp).filter(
        models.Stamp.stampid == stamp_id).first()
    if stamp is None:
        raise HTTPException(status_code=404, detail="Stamp not found")
    return stamp
