#!/usr/bin/env python3
"""
Script để re-parse tất cả documents với legal parser mới
"""
import sys
import json
import os

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import db, LegalDocument  
from legal_parser import LegalDocumentParser
from app import app

def reparse_all_documents():
    """Re-parse tất cả documents chưa có parsed_structure"""
    
    with app.app_context():
        # Get ALL documents to force re-parse with new structure including articles
        documents = LegalDocument.query.all()
        
        if not documents:
            print("❌ Không có documents nào")
            return
        
        print(f"� Force re-parsing ALL {len(documents)} documents để thêm articles:")
        for doc in documents:
            print(f"  - ID {doc.id}: {doc.title}")
        
        parser = LegalDocumentParser()
        success_count = 0
        
        for doc in documents:
            try:
                print(f"\n🔄 Processing document ID {doc.id}: {doc.title[:50]}...")
                
                # Parse document
                parsed_structure = parser.parse_document(doc.content, doc.title)
                
                if parsed_structure:
                    # Save parsed structure to database
                    doc.parsed_structure = json.dumps(parsed_structure, ensure_ascii=False)
                    db.session.commit()
                    
                    # Statistics
                    stats = parsed_structure.get('metadata', {}).get('parsing_stats', {})
                    print(f"  ✅ Parsed successfully:")
                    print(f"     - {stats.get('chapters', 0)} chương")
                    print(f"     - {stats.get('total_sections', 0)} mục") 
                    print(f"     - {stats.get('articles', 0)} điều")
                    
                    success_count += 1
                else:
                    print(f"  ❌ Failed to parse document ID {doc.id}")
                    
            except Exception as e:
                print(f"  ❌ Error parsing document ID {doc.id}: {str(e)}")
        
        print(f"\n🎯 Summary:")
        print(f"   - Successfully parsed: {success_count}/{len(documents)}")
        print(f"   - Failed: {len(documents) - success_count}/{len(documents)}")
        
        if success_count > 0:
            print(f"\n✅ Re-parsing completed! {success_count} documents now have parsed structure.")
        else:
            print(f"\n❌ No documents were successfully parsed.")

if __name__ == "__main__":
    print("🚀 Starting document re-parsing process...")
    reparse_all_documents()
