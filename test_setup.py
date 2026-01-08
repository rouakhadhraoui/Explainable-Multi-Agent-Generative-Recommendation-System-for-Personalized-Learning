# test_setup.py
import sys
print(f"✓ Python version: {sys.version}")

try:
    import numpy as np
    print("✓ Numpy OK")
except:
    print("✗ Numpy manquant")

try:
    import pandas as pd
    print("✓ Pandas OK")
except:
    print("✗ Pandas manquant")

try:
    from sentence_transformers import SentenceTransformer
    print("✓ Sentence-Transformers OK")
except:
    print("✗ Sentence-Transformers manquant")

try:
    import ollama
    print("✓ Ollama OK")
    # Test simple
    response = ollama.chat(model='llama3.2:3b', messages=[
        {'role': 'user', 'content': 'Say hi in one word'}
    ])
    print(f"✓ LLM Test: {response['message']['content']}")
except Exception as e:
    print(f"✗ Ollama error: {e}")