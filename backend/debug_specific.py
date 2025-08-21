#!/usr/bin/env python3

from app import app, db
from models import LegalDocument  
import json

def debug_specific_doc():
    with app.app_context():
        doc = LegalDocument.query.get(4)  # Luật Đường Bộ 2024
        if doc and doc.parsed_structure:
            structure = json.loads(doc.parsed_structure)
            print(f'Document: {doc.title}')
            print(f'Total articles claimed: {structure.get("total_articles", 0)}')
            articles = structure.get('articles', [])
            print(f'Actual articles count: {len(articles)}')
            
            # Show detailed structure
            print(f'Structure keys: {list(structure.keys())}')
            if structure.get('chapters'):
                print(f'Chapters count: {len(structure["chapters"])}')
                
                # Look for articles in the raw content
                if doc.content and 'Điều ' in doc.content:
                    import re
                    article_pattern = r'\nĐiều \d+\.'
                    found_articles = re.findall(article_pattern, doc.content)
                    print(f'Articles found in raw content: {len(found_articles)}')
                    for i, match in enumerate(found_articles[:5]):
                        print(f'  - {match.strip()}')
                    if len(found_articles) > 5:
                        print(f'  ... and {len(found_articles) - 5} more')
                
                for i, chapter in enumerate(structure['chapters'][:2]):
                    print(f'Chapter {i+1}: {chapter.get("name", "N/A")[:50]}...')
                    if chapter.get('sections'):
                        print(f'  - Sections in chapter: {len(chapter["sections"])}')
                        for j, section in enumerate(chapter['sections'][:2]):
                            print(f'    Section {j+1}: {section.get("name", "N/A")[:50]}...')
                            if section.get('articles'):
                                print(f'      - Articles in section: {len(section["articles"])}')
                    elif chapter.get('articles'):
                        print(f'  - Direct articles in chapter: {len(chapter["articles"])}')

if __name__ == "__main__":
    debug_specific_doc()
