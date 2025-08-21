#!/usr/bin/env python3
"""
Debug split văn bản
"""

import re

def debug_article_pattern():
    """Debug pattern matching"""
    
    # Test với một đoạn văn bản mẫu
    sample_text = """
    LUẬT GIAO THÔNG ĐƯỜNG BỘ
    
    Chương I
    QUY ĐỊNH CHUNG
    
    Điều 1. Phạm vi điều chỉnh
    Luật này quy định về giao thông đường bộ; quyền, nghĩa vụ của tổ chức, cá nhân tham gia giao thông đường bộ; quy tắc giao thông đường bộ; tín hiệu giao thông đường bộ; kết cấu hạ tầng giao thông đường bộ; phương tiện giao thông đường bộ và người lái xe; vận tải đường bộ; thanh tra, xử lý vi phạm pháp luật về giao thông đường bộ.
    
    Điều 2. Giải thích từ ngữ
    Trong Luật này, các từ ngữ dưới đây được hiểu như sau:
    1. Giao thông đường bộ là hoạt động di chuyển của người và phương tiện giao thông qua đường bộ.
    2. Tham gia giao thông đường bộ là hoạt động của người và phương tiện giao thông trên đường bộ.
    
    Điều 3. Nguyên tắc bảo đảm trật tự, an toàn giao thông đường bộ
    1. Nhà nước có chính sách đầu tư phát triển kết cấu hạ tầng giao thông đường bộ phù hợp với chiến lược, quy hoạch, kế hoạch phát triển kinh tế - xã hội trong từng thời kỳ, bảo đảm đồng bộ, hiện đại, an toàn, thông suốt, thuận tiện và tiết kiệm.
    2. Việc quy hoạch, đầu tư xây dựng kết cấu hạ tầng giao thông đường bộ phải phù hợp với quy hoạch tổng thể phát triển kinh tế - xã hội.
    
    Điều 86. Quy định chuyển tiếp
    Luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2009; các quy định trước đây trái với Luật này đều bị bãi bỏ.
    """
    
    print("📝 Văn bản test:")
    print(sample_text[:200] + "...")
    print()
    
    # Test các pattern khác nhau
    patterns = [
        r'Điều\s+(\d+)\.\s*([^\r\n]*)',  # Pattern hiện tại (đơn giản)
        r'Điều\s+(\d+)\.\s*([^\r\n]*(?:\r?\n[^\r\n]*(?!Điều\s+\d+\.))*)',  # Pattern phức tạp
        r'Điều\s+(\d+)\.([^Điều]*?)(?=Điều\s+\d+\.|$)',  # Pattern lookahead
        r'(?=Điều\s+\d+\.)',  # Pattern split đơn giản
    ]
    
    for i, pattern in enumerate(patterns, 1):
        print(f"🔍 Pattern {i}: {pattern}")
        try:
            matches = list(re.finditer(pattern, sample_text, re.IGNORECASE | re.MULTILINE | re.DOTALL))
            print(f"   Tìm được {len(matches)} matches:")
            for match in matches:
                if len(match.groups()) >= 2:
                    article_num = match.group(1)
                    article_title = match.group(2).strip()[:50]
                    print(f"     - Điều {article_num}: {article_title}...")
                else:
                    print(f"     - Match: {match.group(0)[:30]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        print()
    
    # Test với split approach
    print("🔪 Split approach:")
    parts = re.split(r'(?=Điều\s+\d+\.)', sample_text, flags=re.IGNORECASE)
    print(f"Split thành {len(parts)} parts:")
    for i, part in enumerate(parts):
        if part.strip():
            preview = part.strip()[:100].replace('\n', ' ')
            print(f"  Part {i}: {preview}...")
    print()

if __name__ == "__main__":
    debug_article_pattern()
