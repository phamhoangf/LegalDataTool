#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug get_all_articles function
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import db, LegalDocument
from legal_parser import LegalDocumentParser
from app import app
import json

def debug_get_all_articles():
    """Debug get_all_articles function"""
    with app.app_context():
        # Test với document ID 5
        doc = LegalDocument.query.filter_by(id=5).first()
        if not doc:
            print("❌ Document ID 5 not found!")
            return
            
        print(f"📄 Testing with: {doc.title}")
        
        parser = LegalDocumentParser()
        
        # Parse document trực tiếp
        parsed_data = parser.parse_document(doc.title, doc.content)
        
        print(f"\n📊 Parsed data keys: {list(parsed_data.keys())}")
        print(f"Total articles reported: {parsed_data.get('total_articles', 0)}")
        
        # Check chapters structure
        chapters = parsed_data.get('chapters', [])
        print(f"📚 Chapters: {len(chapters)}")
        
        for i, chapter in enumerate(chapters):
            chapter_articles = chapter.get('articles', [])
            print(f"   Chapter {i+1}: {len(chapter_articles)} articles")
            
            # Check sections in chapter
            sections = chapter.get('sections', [])
            for j, section in enumerate(sections):
                section_articles = section.get('articles', [])
                print(f"     Section {j+1}: {len(section_articles)} articles")
        
        # Check independent articles
        independent = parsed_data.get('independent_articles', [])
        print(f"🔗 Independent articles: {len(independent)}")
        
        # Test get_all_articles directly
        print(f"\n🧪 Testing get_all_articles():")
        all_articles = parser.get_all_articles(parsed_data)
        print(f"get_all_articles returned: {len(all_articles)} articles")
        
        if all_articles:
            for i, article in enumerate(all_articles[:3]):
                print(f"   {i+1}. Path: {article.get('path', 'No path')}")

if __name__ == "__main__":
    debug_get_all_articles()
