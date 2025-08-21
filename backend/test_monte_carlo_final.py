#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Monte Carlo sampling với tài liệu đã được re-parse
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import db, LegalDocument
from data_generator import DataGenerator
from app import app
import json

def test_monte_carlo_with_reparsed_docs():
    """Test Monte Carlo với tài liệu đã re-parse"""
    print("🧪 Testing Monte Carlo with re-parsed documents...")
    
    with app.app_context():
        # Lấy tài liệu đã được parse
        docs = LegalDocument.query.filter(LegalDocument.parsed_structure.isnot(None)).all()
        print(f"📚 Found {len(docs)} documents with parsed structure")
        
        if not docs:
            print("❌ No documents with parsed structure found!")
            return
            
        # Test với document đầu tiên
        if not docs:
            print("❌ No documents with parsed structure found!")
            return
            
        doc = docs[0]
        print(f"🎯 Testing with document: {doc.title}")
        
        # Initialize data generator
        generator = DataGenerator()
        
        # Generate sample data - test simple generation
        print("\n📝 Testing direct article selection...")
        
        # Parse document để lấy articles
        from legal_parser import LegalDocumentParser
        parser = LegalDocumentParser()
        parsed_data = parser.parse_document(doc.title, doc.content)
        articles = parsed_data.get('articles', [])
        
        if articles:
            print(f"📜 Document has {len(articles)} articles")
            
            # Test Monte Carlo sampling
            from data_generator import monte_carlo_sample_articles
            selected_articles = monte_carlo_sample_articles(articles, num_articles=3)
            
            if selected_articles:
                print(f"✅ Monte Carlo selected {len(selected_articles)} articles:")
                for article in selected_articles:
                    print(f"   - {article['path']}: {article['content'][:100]}...")
            else:
                print("❌ Monte Carlo failed to select articles!")
        else:
            print("❌ No articles found in parsed document!")

def test_monte_carlo_distribution():
    """Test phân phối Monte Carlo"""
    print("\n🎲 Testing Monte Carlo distribution...")
    
    with app.app_context():
        from data_generator import monte_carlo_sample_articles
        from legal_parser import LegalDocumentParser
        
        # Lấy tài liệu có parsed structure
        doc = LegalDocument.query.filter(LegalDocument.parsed_structure.isnot(None)).first()
        if not doc:
            print("❌ No document with parsed structure!")
            return
            
        print(f"📄 Testing with: {doc.title}")
        
        # Parse lại để có articles
        parser = LegalDocumentParser()
        parsed_data = parser.parse_document(doc.title, doc.content)
        articles = parsed_data.get('articles', [])
        
        if not articles:
            print("❌ No articles found!")
            return
            
        print(f"📜 Found {len(articles)} articles")
        
        # Test Monte Carlo sampling
        sample_counts = {}
        num_tests = 100
        
        for _ in range(num_tests):
            selected = monte_carlo_sample_articles(articles, num_articles=1)
            if selected:
                article_path = selected[0]['path']
                sample_counts[article_path] = sample_counts.get(article_path, 0) + 1
        
        print(f"\n📊 Distribution over {num_tests} samples:")
        for path, count in sorted(sample_counts.items()):
            percentage = (count / num_tests) * 100
            print(f"   {path}: {count} times ({percentage:.1f}%)")
            
        # Check if distribution is reasonable (not uniform)
        unique_selections = len(sample_counts)
        if unique_selections > 1:
            print(f"✅ Monte Carlo working - {unique_selections} different articles selected")
        else:
            print("⚠️  Only one article selected (may be expected for small samples)")

if __name__ == "__main__":
    test_monte_carlo_with_reparsed_docs()
    test_monte_carlo_distribution()
