# Company Data Vector Database

A simple vector database solution for querying company employee data using natural language queries.

## Features

- **Local Vector Database**: Uses ChromaDB for persistent local storage
- **Natural Language Queries**: Ask questions about employees in plain English
- **Relationship Mapping**: Understands manager-employee hierarchies
- **RESTful API**: Clean FastAPI endpoints with automatic documentation
- **Minimal Setup**: No external services required

## Quick Start

1. **Install Dependencies**
```bash
   python -m venv venv
   ```

2. **Install Dependencies**
```bash
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   uvicorn main:app --reload
   ```

6. **Test APIs**
   ```bash
   curl -X POST "http://localhost:8000/docs"
   ```

## API Endpoints

### POST /load-data
Loads employee data from `company_data.json` into the vector database.

**Response:**
```json
{
  "message": "Successfully loaded 3 employees",
  "details": {
    "loaded_count": 3,
    "status": "success"
  }
}
```

### POST /query
Queries employee data using natural language.

**Request:**
```json
{
  "query": "Who is the manager of employee 001?"
}
```

**Response:**
```json
{
  "answer": "The manager of Alice Johnson is Carol Baker.",
  "relevant_employees": [
    {
      "id": "emp_001",
      "name": "Alice Johnson",
      "employment_type": "FULL_TIME",
      "manager_name": "Carol Baker"
    }
  ]
}
```

## Example Queries

- "Who is the manager of employee 001?"
- "Who is the manager of the manager of employee 001?"
- "Show all full-time employees"
- "Show all part-time employees"
- "Who works at Acme Corp?"
- "Find employees with no manager"

## Project Structure

```
company-vector-db/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── company_data.json      # Employee data
├── services/
│   └── vector_service.py  # Vector database operations
├── models/
│   └── __init__.py        # Data models
└── chroma_db/            # ChromaDB storage (created automatically)
```

## How It Works

1. **Data Loading**: Employee records are converted to text embeddings using sentence transformers
2. **Vector Storage**: ChromaDB stores embeddings with metadata for fast similarity search
3. **Query Processing**: Natural language queries are matched against stored embeddings
4. **Smart Answers**: The system understands relationships and generates contextual responses

## Technologies Used

- **FastAPI**: Modern web framework for building APIs
- **FaissCPU**: Open-source vector database
- **Sentence Transformers**: For creating text embeddings
- **Pydantic**: Data validation and serialization

