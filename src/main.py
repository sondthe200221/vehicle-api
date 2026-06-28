import os
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, status, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Vehicle Detection API")

# Lấy Database URL và API Key từ môi trường
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")
API_KEY = os.getenv("API_KEY", "SECRET_KEY_12345")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Could not validate credentials")
    return api_key_header

def get_db_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Database connection error: {str(e)}")

class VehicleEvent(BaseModel):
    detected_at: datetime
    license_plate: Optional[str] = None
    category_id: int
    confidence: float
    image_path: Optional[str] = None
    item_ids: List[int] = []

@app.get("/health")
def health_check():
    try:
        conn = get_db_connection()
        conn.close()
        return {"status": "ok", "message": "Connected to PostgreSQL successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/v1/vehicles", status_code=status.HTTP_201_CREATED)
def create_vehicle_event(event: VehicleEvent, api_key: str = Depends(get_api_key)):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO DetectionEvents (DetectedAt, LicensePlate, CategoryID, Confidence, ImagePath)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING EventID
        """, (event.detected_at, event.license_plate, event.category_id, event.confidence, event.image_path))
        
        row = cursor.fetchone()
        event_id = row[0] if row else None
        
        if event_id and event.item_ids:
            for item_id in event.item_ids:
                cursor.execute("""
                    INSERT INTO EventItems (EventID, ItemID)
                    VALUES (%s, %s)
                """, (event_id, item_id))
                
        conn.commit()
        return {"status": "success", "event_id": event_id}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error inserting data: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/api/v1/vehicles")
def get_recent_vehicles(limit: int = 50, api_key: str = Depends(get_api_key)):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("""
            SELECT e.EventID, e.DetectedAt, e.LicensePlate, c.Name as CategoryName, e.Confidence, e.ImagePath
            FROM DetectionEvents e
            LEFT JOIN Categories c ON e.CategoryID = c.CategoryID
            ORDER BY e.DetectedAt DESC
            LIMIT %s
        """, (limit,))
        
        results = cursor.fetchall()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
