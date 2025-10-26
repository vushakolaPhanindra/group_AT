#!/usr/bin/env python3
"""
Simple test script to debug preprocessing module.
"""

import sys
import os
sys.path.append('src')

print("Testing preprocessing module...")

try:
    print("1. Importing preprocess module...")
    import preprocess
    print("   ✅ Module imported successfully")
    
    print("2. Checking available functions...")
    functions = [f for f in dir(preprocess) if not f.startswith('_')]
    print(f"   Available functions: {functions}")
    
    print("3. Testing preprocess_pipeline function...")
    if hasattr(preprocess, 'preprocess_pipeline'):
        print("   ✅ preprocess_pipeline function found")
        
        print("4. Running preprocessing pipeline...")
        X_train, X_test, y_train, y_test, label_encoders = preprocess.preprocess_pipeline()
        
        print("   ✅ Pipeline executed successfully!")
        print(f"   Training set shape: {X_train.shape}")
        print(f"   Test set shape: {X_test.shape}")
        print(f"   Feature columns: {list(X_train.columns)}")
        print(f"   Target distribution: {y_train.value_counts().to_dict()}")
        
    else:
        print("   ❌ preprocess_pipeline function not found")
        
except Exception as e:
    print(f"   ❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("Test completed.")
