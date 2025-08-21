#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test parser với document cụ thể
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import db, LegalDocument
from legal_parser import LegalDocumentParser
from app import app
import json

def test_single_document():
    """Test parser với document cụ thể"""
    with app.app_context():
        # Test với document ID 5 (document nhỏ)
        doc = LegalDocument.query.get(5)
        if not doc:
            print("❌ Document ID 5 not found!")
            return
            
        print(f"📄 Testing with document: {doc.title}")
        print(f"📝 Content preview: {doc.content[:300]}...")
        
        parser = LegalDocumentParser()
        
        # Parse document
        parsed_data = parser.parse_document(doc.title, doc.content)
        
        if parsed_data:
            print(f"\n✅ Parse successful!")
            print(f"Structure keys: {list(parsed_data.keys())}")
            
            # Check if articles exist
            if 'articles' in parsed_data:
                articles = parsed_data['articles']
                print(f"📜 Found {len(articles)} articles:")
                for i, article in enumerate(articles[:3]):
                    print(f"   {i+1}. Path: {article.get('path', 'No path')}")
                    print(f"      Content: {article.get('content', 'No content')[:100]}...")
            else:
                print("❌ No 'articles' key in parsed data")
                
                # Check get_all_articles directly
                print("\n🔍 Testing get_all_articles directly:")
                articles = parser.get_all_articles(parsed_data)
                print(f"Direct get_all_articles returned {len(articles)} articles")
                
                if articles:
                    for i, article in enumerate(articles[:3]):
                        print(f"   {i+1}. Path: {article.get('path', 'No path')}")
                        print(f"      Content: {article.get('content', 'No content')[:100]}...")
        else:
            print("❌ Parse failed!")

if __name__ == "__main__":
    test_single_document()
