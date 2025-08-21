#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manual update document ID 5
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import db, LegalDocument
from legal_parser import LegalDocumentParser
from app import app
import json

def update_doc_5():
    """Manual update document ID 5"""
    with app.app_context():
        doc = LegalDocument.query.filter_by(id=5).first()
        if not doc:
            print("❌ Document ID 5 not found!")
            return
            
        print(f"📄 Updating: {doc.title}")
        
        parser = LegalDocumentParser()
        
        # Parse và lưu
        parsed_data = parser.parse_document(doc.title, doc.content)
        
        if parsed_data:
            articles = parsed_data.get('articles', [])
            print(f"✅ Generated {len(articles)} articles")
            
            # Save to database
            doc.parsed_structure = json.dumps(parsed_data, ensure_ascii=False)
            db.session.commit()
            print("✅ Saved to database")
            
            # Verify
            reloaded = LegalDocument.query.filter_by(id=5).first()
            if reloaded.parsed_structure:
                structure = json.loads(reloaded.parsed_structure)
                saved_articles = structure.get('articles', [])
                print(f"✅ Verified: {len(saved_articles)} articles in database")
                
                if saved_articles:
                    # Test Monte Carlo
                    selected = parser.monte_carlo_sample_articles(saved_articles, sample_size=2)
                    
                    if selected:
                        print(f"\n🎯 Monte Carlo test - selected {len(selected)} articles:")
                        for i, article in enumerate(selected):
                            print(f"   {i+1}. Path: {article.get('path', 'No path')}")
                    else:
                        print("❌ Monte Carlo failed!")
            else:
                print("❌ No parsed structure after save!")
        else:
            print("❌ Parse failed!")

if __name__ == "__main__":
    update_doc_5()
