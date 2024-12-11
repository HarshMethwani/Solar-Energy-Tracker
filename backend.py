import os
import numpy as np
import pandas as pd
import tensorflow as tf
import uvicorn
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, declarative_base
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import logging
from datetime import datetime
import uuid
import torch

# Configure logging for the application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='forecasting_service.log'
)
app_logger = logging.getLogger(__name__)

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))
# Load the machine learning model
try:
    custom_objects = {"mse": mse}  # Include your custom function here
    solar_model = tf.keras.models.load_model("solar_forecaster.h5", custom_objects=custom_objects)
except Exception as model_load_error:
    app_logger.error(f"Error loading the model: {model_load_error}")
    solar_model = None
# Setup for database interactions
Base = declarative_base()

class ForecastData(Base):
    """
    Represents forecast entries in the database
    """
    __tablename__ = 'forecast_data'
    id = sa.Column(sa.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = sa.Column(sa.DateTime, index=True)
    input_data = sa.Column(sa.JSON)
    forecast_values = sa.Column(sa.JSON)
    metrics = sa.Column(sa.JSON)

# Input schema for the API
class SolarForecastInput(BaseModel):
    """
    Schema for forecast request payload
    """
    input_sequences: list[list[float]]
    version: str = Field(default="v1.0")

# Output schema for the API
class ForecastResponse(BaseModel):
    """
    Schema for forecast response payload
    """
    forecast_values: list[float]
    version: str
    timestamp: datetime

class DBHandler:
    """
    Manages database operations for forecast storage and retrieval
    """
    def __init__(self, db_url: str = "sqlite:///solar_forecasts.db"):
        self.engine = sa.create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_forecast(self, input_payload, forecast_values, metrics=None):
        """
        Save forecast details to the database
        """
        session = self.Session()
        try:
            record = ForecastData(
                timestamp=datetime.now(),
                input_data=input_payload,
                forecast_values=forecast_values,
                metrics=metrics
            )
            session.add(record)
            session.commit()
            return record.id
        except Exception as db_error:
            session.rollback()
            app_logger.error(f"Database save error: {db_error}")
            raise
        finally:
            session.close()

# Create a database handler instance
db_handler = DBHandler()

# Initialize FastAPI application
app = FastAPI(
    title="Solar Power Forecasting API",
    description="API for real-time prediction of solar power generation",
    version="1.0.0"
)

def compute_error_metrics(actual, predicted):
    """
    Calculate performance metrics for the forecast
    """
    try:
        mse = np.mean((actual - predicted) ** 2)
        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        return {
            "MSE": float(mse),
            "MAE": float(mae),
            "MAPE": float(mape)
        }
    except Exception as metrics_error:
        app_logger.error(f"Error calculating metrics: {metrics_error}")
        return {}

@app.post("/forecast")
async def generate_forecast(data: SolarForecastInput, background_tasks: BackgroundTasks):
    """
    Generate solar power forecast
    """
    try:
        if solar_model is None:
            raise HTTPException(status_code=500, detail="Model is not loaded")

        # Process input data
        inputs = np.array(data.input_sequences)
        predictions = solar_model.predict(inputs).tolist()

        # Store forecast asynchronously
        forecast_id = db_handler.save_forecast(data.dict(), predictions)
        background_tasks.add_task(app_logger.info, f"Forecast saved: ID {forecast_id}")

        return ForecastResponse(
            forecast_values=predictions,
            version=data.version,
            timestamp=datetime.now()
        )
    except Exception as forecast_error:
        app_logger.error(f"Error generating forecast: {forecast_error}")
        raise HTTPException(status_code=500, detail="Forecasting failed")

@app.get("/forecasts/recent")
async def fetch_recent_forecasts(limit: int = 10):
    """
    Retrieve recent forecast entries from the database
    """
    session = db_handler.Session()
    try:
        recent_entries = session.query(ForecastData).order_by(
            ForecastData.timestamp.desc()
        ).limit(limit).all()
        return [{"id": record.id, "timestamp": record.timestamp, "forecast_values": record.forecast_values}
                for record in recent_entries]
    except Exception as fetch_error:
        app_logger.error(f"Error fetching forecasts: {fetch_error}")
        raise HTTPException(status_code=500, detail="Could not retrieve forecasts")
    finally:
        session.close()

@app.get("/health")
async def health_check():
    """
    Endpoint to verify service health
    """
    return {
        "status": "operational",
        "timestamp": datetime.now(),
        "model_loaded": solar_model is not None,
        "db_connected": db_handler.engine is not None
    }

if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 8000))
    HOST = os.getenv("HOST", "0.0.0.0")

    uvicorn.run("solar_forecasting_api:app", host=HOST, port=PORT, reload=True)
