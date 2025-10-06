#!/usr/bin/env python3
"""
Script to update articles_count field for existing documents
Run this once to migrate existing data
"""

import json
from app import app, db, LegalDocument

def update_articles_count():
    """Update articles_count field for all existing documents"""
    with app.app_context():
        print("🔄 Updating articles_count for existing documents...")
        
        documents = LegalDocument.query.all()
        updated_count = 0
        
        for doc in documents:
            articles_count = 0
            
            if doc.parsed_structure:
                try:
                    structure = json.loads(doc.parsed_structure)
                    articles = structure.get('articles', [])
                    articles_count = len(articles)
                except (json.JSONDecodeError, TypeError):
                    articles_count = 0
            
            # Update the field
            doc.articles_count = articles_count
            updated_count += 1
            
            print(f"📄 {doc.title}: {articles_count} điều")
        
        # Commit all changes
        db.session.commit()
        print(f"✅ Updated {updated_count} documents successfully!")

if __name__ == '__main__':
    update_articles_count()
