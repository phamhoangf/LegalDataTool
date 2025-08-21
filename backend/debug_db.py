#!/usr/bin/env python3

from app import app, db
from models import LegalDocument
import json

def debug_documents():
    with app.app_context():
        docs = LegalDocument.query.all()
        
        for doc in docs:
            print(f"\n=== Document: {doc.title[:50]}... ===")
            print(f"ID: {doc.id}")
            print(f"Has parsed_structure: {bool(doc.parsed_structure)}")
            
            if doc.parsed_structure:
                try:
                    structure = json.loads(doc.parsed_structure)
                    articles = structure.get('articles', [])
                    print(f"Articles count: {len(articles)}")
                    print(f"Structure keys: {list(structure.keys())}")
                    
                    # Show first article as sample
                    if articles:
                        print(f"First article sample: {str(articles[0])[:100]}...")
                    
                except Exception as e:
                    print(f"ERROR parsing JSON: {e}")
                    print(f"Raw parsed_structure (first 200 chars): {doc.parsed_structure[:200]}")
            else:
                print("No parsed_structure found")

if __name__ == "__main__":
    debug_documents()
