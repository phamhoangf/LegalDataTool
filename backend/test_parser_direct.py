#!/usr/bin/env python3

from app import app, db
from models import LegalDocument
from legal_parser import LegalDocumentParser
import json
import re

def test_direct_parsing():
    with app.app_context():
        doc = LegalDocument.query.get(4)  # Luật Đường Bộ 2024
        if doc:
            print(f'Testing direct parsing of: {doc.title}')
            print(f'Content length: {len(doc.content)}')
            
            # Check for articles in raw content
            if 'Điều ' in doc.content:
                article_pattern = r'\nĐiều \d+\.'
                found_articles = re.findall(article_pattern, doc.content)
                print(f'Articles found in raw content with pattern: {len(found_articles)}')
                if found_articles:
                    for i, match in enumerate(found_articles[:5]):
                        print(f'  - Found: "{match.strip()}"')
                    
                # Try actual parser
                parser = LegalDocumentParser()
                print('\n🔄 Testing with LegalDocumentParser...')
                parsed = parser.parse_document(doc.title, doc.content)
                
                print(f'Parser result:')
                print(f'  - Total articles: {parsed.get("total_articles", 0)}')
                print(f'  - Articles array length: {len(parsed.get("articles", []))}')
                print(f'  - Chapters: {len(parsed.get("chapters", []))}')
                
                if parsed.get('chapters'):
                    for i, chapter in enumerate(parsed['chapters'][:2]):
                        print(f'  Chapter {i+1}: {chapter.get("title", "No title")}')
                        print(f'    - Direct articles: {len(chapter.get("articles", []))}')
                        print(f'    - Sections: {len(chapter.get("sections", []))}')
                        if chapter.get('sections'):
                            for j, section in enumerate(chapter['sections'][:2]):
                                print(f'      Section {j+1}: {section.get("title", "No title")}')
                                print(f'        - Articles: {len(section.get("articles", []))}')

if __name__ == "__main__":
    test_direct_parsing()
