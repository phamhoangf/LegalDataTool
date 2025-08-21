#!/usr/bin/env python3

from app import app, db
from models import LegalDocument
from legal_parser import LegalDocumentParser
import json

def fix_all_documents():
    with app.app_context():
        docs = LegalDocument.query.all()
        parser = LegalDocumentParser()
        
        print(f'🔧 Found {len(docs)} documents to fix')
        
        for doc in docs:
            try:
                print(f'\n📄 Processing: {doc.title}')
                
                # Check current state
                current_articles = 0
                if doc.parsed_structure:
                    current_structure = json.loads(doc.parsed_structure)
                    current_articles = current_structure.get('total_articles', 0)
                
                print(f'  Current articles: {current_articles}')
                
                # Parse with correct parser
                parsed = parser.parse_document(doc.title, doc.content)
                found_articles = parsed.get('total_articles', 0)
                
                print(f'  Parser found: {found_articles} articles')
                
                # Update database
                doc.parsed_structure = json.dumps(parsed, ensure_ascii=False)
                db.session.commit()
                
                print(f'  ✅ Updated successfully')
                
            except Exception as e:
                print(f'  ❌ Error processing {doc.title}: {str(e)}')
                db.session.rollback()
        
        print(f'\n🎉 All documents processed!')

if __name__ == "__main__":
    fix_all_documents()
