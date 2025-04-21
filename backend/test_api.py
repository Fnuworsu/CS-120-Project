import requests
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def create_test_patient():
    # Create a patient
    patient_data = {
        "first_name": "John",
        "last_name": "Doe",
        "date_of_birth": datetime.now().isoformat(),
        "address": "123 Test St",
        "email": "john.doe@example.com"
    }
    
    response = requests.post(f"{BASE_URL}/patients/", json=patient_data)
    print("Create Patient Response:", response.status_code)
    print("Response Data:", response.json())
    return response.json()

if __name__ == "__main__":
    patient = create_test_patient()
    print("\nCreated Patient ID:", patient.get("id")) 