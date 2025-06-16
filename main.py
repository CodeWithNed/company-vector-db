from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.vector_service import VectorService
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Company Data Vector Database", version="1.0.0")

# Initialize vector service
vector_service = VectorService()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    relevant_employees: list


@app.post("/load-data")
async def load_data():
    """Load company data from JSON file into vector database"""
    try:
        with open("company_data.json", "r") as file:
            data = json.load(file)

        employees = data.get("results", [])
        if not employees:
            raise HTTPException(status_code=400, detail="No employee data found")

        result = vector_service.load_employees(employees)
        logger.info(f"Loaded {len(employees)} employees into vector database")

        return {
            "message": f"Successfully loaded {len(employees)} employees",
            "details": result
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="company_data.json file not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_data(request: QueryRequest):
    """Query employee data using natural language"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        result = vector_service.query_employees(request.query)
        logger.info(f"Query processed: {request.query}")

        return QueryResponse(
            answer=result["answer"],
            relevant_employees=result["relevant_employees"]
        )

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Company Data Vector Database API",
        "endpoints": {
            "load_data": "/load-data (POST)",
            "query": "/query (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "vector-db"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)