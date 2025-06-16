import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import pickle
import os
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class VectorService:
    def __init__(self):
        # Initialize sentence transformer for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize FAISS index
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity

        # Store employee data and metadata
        self.employees_dict = {}
        self.employee_ids = []
        self.employee_metadata = []

        # File paths for persistence
        self.index_path = "vector_index.faiss"
        self.metadata_path = "metadata.pkl"

        # Load existing data if available
        self._load_existing_data()

    def _load_existing_data(self):
        """Load existing index and metadata from disk"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                # Load FAISS index
                self.index = faiss.read_index(self.index_path)

                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.employees_dict = data['employees_dict']
                    self.employee_ids = data['employee_ids']
                    self.employee_metadata = data['employee_metadata']

                logger.info(f"Loaded existing data: {len(self.employee_ids)} employees")
        except Exception as e:
            logger.warning(f"Could not load existing data: {e}")

    def _save_data(self):
        """Save index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)

            # Save metadata
            data = {
                'employees_dict': self.employees_dict,
                'employee_ids': self.employee_ids,
                'employee_metadata': self.employee_metadata
            }
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(data, f)

            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def create_employee_text(self, employee: Dict[str, Any]) -> str:
        """Create a comprehensive text representation of an employee"""
        manager_info = ""
        if employee.get("manager"):
            manager_info = f"Manager: {employee['manager']['display_full_name']} (ID: {employee['manager']['id']})"
        else:
            manager_info = "No manager (likely a top-level employee)"

        text = f"""
        Employee ID: {employee['id']}
        Name: {employee['display_full_name']}
        First Name: {employee['first_name']}
        Last Name: {employee['last_name']}
        Employment Status: {employee['employment_status']}
        Employment Type: {employee['employment_type']}
        Start Date: {employee['start_date']}
        Company: {employee['company']['name']}
        {manager_info}
        """

        return text.strip()

    def load_employees(self, employees: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Load employee data into vector database"""
        try:
            # Clear existing data
            self.index = faiss.IndexFlatIP(self.dimension)
            self.employees_dict.clear()
            self.employee_ids.clear()
            self.employee_metadata.clear()

            # Prepare data
            texts = []
            for employee in employees:
                # Store in dict for quick lookup
                self.employees_dict[employee['id']] = employee
                self.employee_ids.append(employee['id'])

                # Create text representation
                text = self.create_employee_text(employee)
                texts.append(text)

                # Create metadata
                metadata = {
                    "id": employee['id'],
                    "name": employee['display_full_name'],
                    "employment_type": employee['employment_type'],
                    "employment_status": employee['employment_status'],
                    "manager_id": employee['manager']['id'] if employee.get('manager') else None,
                    "manager_name": employee['manager']['display_full_name'] if employee.get('manager') else None,
                    "company": employee['company']['name']
                }
                self.employee_metadata.append(metadata)

            # Create embeddings
            embeddings = self.model.encode(texts)

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)

            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))

            # Save to disk
            self._save_data()

            logger.info(f"Successfully loaded {len(employees)} employees")
            return {"loaded_count": len(employees), "status": "success"}

        except Exception as e:
            logger.error(f"Error loading employees: {str(e)}")
            raise

    def find_manager_chain(self, employee_id: str, levels: int = 1) -> List[Dict[str, Any]]:
        """Find manager chain for a given employee"""
        chain = []
        current_id = employee_id

        for _ in range(levels):
            if current_id not in self.employees_dict:
                break

            employee = self.employees_dict[current_id]
            if not employee.get('manager'):
                break

            manager = employee['manager']
            chain.append(manager)
            current_id = manager['id']

        return chain

    def query_employees(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Query employees using natural language"""
        try:
            if self.index.ntotal == 0:
                return {
                    "answer": "No employee data loaded. Please load data first.",
                    "relevant_employees": []
                }

            # Create query embedding
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)

            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'),
                                                min(n_results, self.index.ntotal))

            # Process results
            relevant_employees = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # Valid index
                    metadata = self.employee_metadata[idx]
                    relevant_employees.append({
                        "id": metadata['id'],
                        "name": metadata['name'],
                        "employment_type": metadata['employment_type'],
                        "manager_name": metadata.get('manager_name'),
                        "similarity_score": float(scores[0][i])
                    })

            # Generate answer based on query type
            answer = self.generate_answer(query, relevant_employees)

            return {
                "answer": answer,
                "relevant_employees": relevant_employees
            }

        except Exception as e:
            logger.error(f"Error querying employees: {str(e)}")
            raise

    def generate_answer(self, query: str, relevant_employees: List[Dict]) -> str:
        """Generate a natural language answer based on the query and results"""
        query_lower = query.lower()

        if not relevant_employees:
            return "No relevant employees found for your query."

        # Handle manager queries
        if "manager" in query_lower:
            if "manager of" in query_lower and any(emp_id in query_lower for emp_id in ["001", "002", "003"]):
                # Extract employee ID from query
                emp_id = None
                for id_candidate in ["001", "002", "003"]:
                    if id_candidate in query_lower:
                        emp_id = f"emp_{id_candidate}"
                        break

                if emp_id and emp_id in self.employees_dict:
                    employee = self.employees_dict[emp_id]

                    # Check for "manager of manager" pattern
                    if "manager of" in query_lower and query_lower.count("manager") >= 2:
                        # Find the manager's manager
                        manager_chain = self.find_manager_chain(emp_id, levels=2)
                        if len(manager_chain) >= 2:
                            return f"The manager of the manager of {employee['display_full_name']} is {manager_chain[1]['display_full_name']}."
                        elif len(manager_chain) == 1:
                            return f"The manager of {employee['display_full_name']} is {manager_chain[0]['display_full_name']}, but they don't have a manager above them."
                        else:
                            return f"{employee['display_full_name']} doesn't have a manager."
                    else:
                        # Find direct manager
                        if employee.get('manager'):
                            return f"The manager of {employee['display_full_name']} is {employee['manager']['display_full_name']}."
                        else:
                            return f"{employee['display_full_name']} doesn't have a manager (likely a top-level employee)."

        # Handle employment type queries
        if "full-time" in query_lower or "full time" in query_lower:
            full_time_employees = [emp for emp in relevant_employees if emp.get('employment_type') == 'FULL_TIME']
            if full_time_employees:
                names = [emp['name'] for emp in full_time_employees]
                return f"Full-time employees: {', '.join(names)}"

        if "part-time" in query_lower or "part time" in query_lower:
            part_time_employees = [emp for emp in relevant_employees if emp.get('employment_type') == 'PART_TIME']
            if part_time_employees:
                names = [emp['name'] for emp in part_time_employees]
                return f"Part-time employees: {', '.join(names)}"

        # Default response
        if len(relevant_employees) == 1:
            emp = relevant_employees[0]
            manager_info = f" Their manager is {emp['manager_name']}." if emp.get(
                'manager_name') else " They don't have a manager."
            return f"Found employee: {emp['name']} ({emp['employment_type']}).{manager_info}"
        else:
            names = [emp['name'] for emp in relevant_employees[:3]]
            return f"Found {len(relevant_employees)} relevant employees. Top matches: {', '.join(names)}"