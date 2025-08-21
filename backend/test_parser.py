#!/usr/bin/env python3

from legal_parser import LegalDocumentParser

def test_parser():
    parser = LegalDocumentParser()
    test_text = '''
    Điều 1. Phạm vi điều chỉnh
    Luật này quy định về giao thông đường bộ.

    Điều 2. Giải thích từ ngữ
    Trong Luật này, các từ ngữ được hiểu như sau:
    1. Giao thông đường bộ là hoạt động di chuyển.
    '''

    result = parser.parse_document('Test Document', test_text)
    print(f'Articles found: {len(result.get("articles", []))}')
    print(f'Total articles: {result.get("total_articles", 0)}')
    if result.get('articles'):
        print(f'First article: {result["articles"][0]}')
    else:
        print('No articles found!')
        
    print(f'Structure keys: {list(result.keys())}')

if __name__ == "__main__":
    test_parser()
