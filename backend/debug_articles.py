#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug articles in database
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import db, LegalDocument
from app import app
import json

def debug_articles():
    """Debug articles trong database"""
    with app.app_context():
        docs = LegalDocument.query.all()
        
        for doc in docs:
            print(f"\n📄 Document ID {doc.id}: {doc.title}")
            
            if doc.parsed_structure:
                try:
                    structure = json.loads(doc.parsed_structure)
                    print(f"   Structure keys: {list(structure.keys())}")
                    
                    if 'articles' in structure:
                        articles = structure['articles']
                        print(f"   ✅ Has articles key with {len(articles)} articles")
                        
                        if articles:
                            print(f"   Sample article: {articles[0].keys()}")
                    else:
                        print(f"   ❌ No articles key")
                        
                except Exception as e:
                    print(f"   ❌ Error parsing JSON: {e}")
            else:
                print(f"   ❌ No parsed_structure")

if __name__ == "__main__":
    debug_articles()
