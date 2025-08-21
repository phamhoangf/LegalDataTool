#!/usr/bin/env python3
"""
Test split văn bản luật
"""

import re
from coverage_analyzer import CoverageAnalyzer

def test_article_split():
    """Test split articles"""
    
    sample_text = """
    Chương I. QUY ĐỊNH CHUNG
    
    Điều 1. Phạm vi điều chỉnh
    Luật này quy định về giao thông đường bộ; quyền, nghĩa vụ của tổ chức, cá nhân tham gia giao thông đường bộ.
    
    Điều 2. Giải thích từ ngữ
    Trong Luật này, các từ ngữ dưới đây được hiểu như sau:
    1. Giao thông đường bộ là hoạt động di chuyển của người và phương tiện giao thông.
    2. Phương tiện giao thông đường bộ bao gồm xe cơ giới đường bộ, xe thô sơ.
    
    Điều 3. Các nguyên tắc cơ bản
    Phát triển giao thông đường bộ phải tuân thủ các nguyên tắc:
    a) Bảo đảm thống nhất, đồng bộ, an toàn, thuận tiện.
    
    Điều 86. Quy định chuyển tiếp
    Luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2009.
    """
    
    analyzer = CoverageAnalyzer()
    units = analyzer.split_into_units(sample_text, 'sentence')
    
    print(f"📊 Tìm được {len(units)} units:")
    for unit in units:
        print(f"  - {unit['id']}: {unit.get('article_title', 'N/A')}")
        print(f"    Content preview: {unit['content'][:100]}...")
        print()
    
    # Test với pattern thực tế
    article_pattern = r'Điều\s+(\d+)\.\s*([^\n]*)'
    matches = list(re.finditer(article_pattern, sample_text, re.IGNORECASE))
    print(f"🔍 Pattern tìm được {len(matches)} matches:")
    for match in matches:
        print(f"  - Điều {match.group(1)}: {match.group(2)}")

if __name__ == "__main__":
    test_article_split()
