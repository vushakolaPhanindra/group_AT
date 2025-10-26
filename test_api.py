"""
Test script for the Credit Score Intelligence API.
"""

import requests
import json
import time

def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:8000"
    
    # Test data
    test_data = {
        "Age": 35,
        "Gender": "Male",
        "Income": 75000.0,
        "Education": "Bachelor",
        "Marital_Status": "Married",
        "Number_of_Dependents": 2,
        "Credit_History_Length": 8,
        "Number_of_Credit_Cards": 3,
        "Number_of_Bank_Accounts": 2,
        "Interest_Rate": 12.5,
        "Number_of_Loans": 1,
        "Delay_from_due_date": 5,
        "Outstanding_Debt": 15000.0,
        "Credit_Utilization_Ratio": 0.25
    }
    
    print("Testing Credit Score Intelligence API...")
    print("=" * 50)
    
    # Test health endpoint
    try:
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return
    
    # Test predict endpoint
    try:
        print("\n2. Testing predict endpoint...")
        response = requests.post(f"{base_url}/predict", json=test_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful")
            print(f"   Category: {result['category']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Top features: {list(result['feature_importance'].keys())[:3]}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
    
    # Test explain endpoint
    try:
        print("\n3. Testing explain endpoint...")
        response = requests.post(f"{base_url}/explain", json=test_data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            print("✅ Explanation successful")
            print(f"   Category: {result['category']}")
            print(f"   Rationale length: {len(result['rationale'])} characters")
            print(f"   SHAP plot: {result['shap_plot']}")
            print(f"   Rationale preview: {result['rationale'][:200]}...")
        else:
            print(f"❌ Explanation failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Explanation failed: {str(e)}")
    
    print("\n" + "=" * 50)
    print("API testing completed!")

if __name__ == "__main__":
    test_api()
