#!/usr/bin/env python3

from app import app, db
from models import LegalDocument
from legal_parser import LegalDocumentParser
import json

def fix_single_document():
    with app.app_context():
        doc = LegalDocument.query.get(4)  # Luật Đường Bộ 2024
        if doc:
            print(f'Fixing: {doc.title}')
            print(f'Current parsed_structure articles: {json.loads(doc.parsed_structure).get("total_articles", 0) if doc.parsed_structure else "No structure"}')
            
            # Parse with parser
            parser = LegalDocumentParser()
            parsed = parser.parse_document(doc.title, doc.content)
            
            print(f'Parser found: {parsed.get("total_articles", 0)} articles')
            
            # Update database
            doc.parsed_structure = json.dumps(parsed, ensure_ascii=False)
            db.session.commit()
            
            print('✅ Database updated!')
            
            # Verify
            doc_check = LegalDocument.query.get(4)
            new_structure = json.loads(doc_check.parsed_structure)
            print(f'Verification: {new_structure.get("total_articles", 0)} articles now in database')

if __name__ == "__main__":
    fix_single_document()
