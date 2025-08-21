#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kiểm tra parsed structure thực tế
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import db, LegalDocument
from app import app
import json

def check_parsed_structure():
    """Kiểm tra cấu trúc parsed thực tế"""
    with app.app_context():
        docs = LegalDocument.query.filter(LegalDocument.parsed_structure.isnot(None)).all()
        
        for doc in docs[:2]:  # Chỉ check 2 docs đầu
            print(f"\n📄 Document: {doc.title}")
            print(f"ID: {doc.id}")
            
            if doc.parsed_structure:
                try:
                    structure = json.loads(doc.parsed_structure)
                    print(f"✅ Parsed structure keys: {structure.keys()}")
                    
                    if 'articles' in structure:
                        articles = structure['articles']
                        print(f"📜 Articles count: {len(articles)}")
                        
                        if articles:
                            # Show first few articles
                            for i, article in enumerate(articles[:3]):
                                print(f"   Article {i+1}: {article.get('path', 'No path')} - {article.get('content', 'No content')[:100]}...")
                    else:
                        print("❌ No 'articles' key in parsed structure")
                        print(f"Available keys: {list(structure.keys())}")
                        
                except Exception as e:
                    print(f"❌ Error parsing JSON: {e}")
            else:
                print("❌ No parsed structure")

if __name__ == "__main__":
    check_parsed_structure()
