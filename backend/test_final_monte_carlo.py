#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Monte Carlo hoàn chỉnh
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import db, LegalDocument
from app import app
import json

def test_final_monte_carlo():
    """Test Monte Carlo hoàn chỉnh với tài liệu có articles"""
    with app.app_context():
        # Tìm document có articles
        docs = LegalDocument.query.all()
        
        for doc in docs:
            if doc.parsed_structure:
                try:
                    structure = json.loads(doc.parsed_structure)
                    articles = structure.get('articles', [])
                    
                    if articles:
                        print(f"✅ Found document with {len(articles)} articles: {doc.title}")
                        
                        # Test Monte Carlo sampling
                        # Import directly from data_generator 
                        import sys
                        import os
                        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                        
                        # Check if this file has the function
                        try:
                            from data_generator import monte_carlo_sample_articles
                            print("✅ Successfully imported monte_carlo_sample_articles from data_generator")
                        except ImportError:
                            print("❌ Could not import from data_generator, trying legal_parser...")
                            try:
                                from legal_parser import LegalDocumentParser
                                parser = LegalDocumentParser()
                                selected = parser.monte_carlo_sample_articles(articles, sample_size=3)
                                print("✅ Using parser.monte_carlo_sample_articles")
                            except Exception as e2:
                                print(f"❌ Parser method failed: {e2}")
                                return
                        
                        # Sample 3 articles
                        selected = monte_carlo_sample_articles(articles, num_articles=3)
                        
                        if selected:
                            print(f"🎯 Monte Carlo selected {len(selected)} articles:")
                            for i, article in enumerate(selected):
                                print(f"   {i+1}. Path: {article.get('path', 'No path')}")
                                print(f"      Content: {article.get('content', 'No content')[:100]}...")
                        else:
                            print("❌ Monte Carlo failed!")
                        
                        return  # Test with first found document
                        
                except Exception as e:
                    print(f"❌ Error parsing JSON for doc {doc.id}: {e}")
        
        print("❌ No documents with articles found!")

if __name__ == "__main__":
    test_final_monte_carlo()
