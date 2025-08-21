#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script thử nghiệm parse văn bản pháp luật thành cấu trúc JSON
Hierarchy: Tài liệu -> Chươn    def _create_chapter(self, match, line_num: int, chapter_title: str = "") -> Dict[str, Any]:
        """Tạo chapter object"""
        chapter_num = match.group(1)
        
        return {
            'type': 'chapter',
            'number': chapter_num,
            'title': chapter_title,
            'line_number': line_num,
            'sections': [],
            'articles': []
        }iều
"""

import re
import json
import random
from typing import Dict, List, Any, Optional

class LegalDocumentParser:
    """Parse văn bản pháp luật thành cấu trúc hierarchical"""
    
    def __init__(self):
        # Patterns để nhận diện các cấp độ
        self.patterns = {
            'chuong': r'CHƯƠNG\s+([IVXLC]+|[0-9]+)\.?\s*([^\n\r]*)',
            'muc': r'Mục\s+(\d+)\.?\s*([^\n\r]*)', 
            'dieu': r'Điều\s+(\d+)\.?\s*([^\n\r]*)',
            'khoan': r'^(\d+)\.\s*([^\n\r]*)',  # Các khoản trong điều
            'diem': r'^([a-z])\)\s*([^\n\r]*)'   # Các điểm trong khoản
        }
    
    def parse_document(self, title: str, content: str) -> Dict[str, Any]:
        """
        Parse toàn bộ document thành cấu trúc JSON
        
        Args:
            title: Tên tài liệu
            content: Nội dung văn bản
            
        Returns:
            Dict: Cấu trúc JSON của tài liệu
        """
        print(f"📄 Parsing document: {title}")
        
        # Làm sạch content
        content = self._clean_content(content)
        
        # Khởi tạo structure
        document_structure = {
            'title': title,
            'type': 'document',
            'content_length': len(content),
            'chapters': [],
            'total_articles': 0,
            'metadata': {
                'has_chapters': False,
                'has_sections': False,
                'parsing_stats': {}
            }
        }
        
        # Split content thành lines để process
        lines = content.split('\n')
        
        current_chapter = None
        current_section = None
        current_article = None
        
        article_count = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # 1. Kiểm tra CHƯƠNG
            chapter_match = re.match(self.patterns['chuong'], line, re.IGNORECASE)
            if chapter_match:
                # Tìm title của chương ở dòng tiếp theo (nếu có)
                chapter_title = ""
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    # Nếu dòng tiếp theo không phải là pattern đặc biệt, coi là title
                    if (next_line and 
                        not re.match(self.patterns['chuong'], next_line, re.IGNORECASE) and
                        not re.match(self.patterns['muc'], next_line, re.IGNORECASE) and
                        not re.match(self.patterns['dieu'], next_line, re.IGNORECASE)):
                        chapter_title = next_line
                
                current_chapter = self._create_chapter(chapter_match, i, chapter_title)
                document_structure['chapters'].append(current_chapter)
                current_section = None
                current_article = None
                document_structure['metadata']['has_chapters'] = True
                continue
            
            # 2. Kiểm tra MỤC
            section_match = re.match(self.patterns['muc'], line, re.IGNORECASE)
            if section_match:
                current_section = self._create_section(section_match, i)
                if current_chapter:
                    current_chapter['sections'].append(current_section)
                else:
                    # Mục độc lập không thuộc chương nào
                    if 'independent_sections' not in document_structure:
                        document_structure['independent_sections'] = []
                    document_structure['independent_sections'].append(current_section)
                current_article = None
                document_structure['metadata']['has_sections'] = True
                continue
            
            # 3. Kiểm tra ĐIỀU
            article_match = re.match(self.patterns['dieu'], line, re.IGNORECASE)
            if article_match:
                current_article = self._create_article(article_match, i, lines)
                article_count += 1
                
                # Gán article vào đúng container
                if current_section:
                    current_section['articles'].append(current_article)
                elif current_chapter:
                    current_chapter['articles'].append(current_article)
                else:
                    # Article độc lập
                    if 'independent_articles' not in document_structure:
                        document_structure['independent_articles'] = []
                    document_structure['independent_articles'].append(current_article)
                continue
        
        # Finalize document
        document_structure['total_articles'] = article_count
        document_structure['metadata']['parsing_stats'] = {
            'chapters': len(document_structure.get('chapters', [])),
            'total_sections': self._count_sections(document_structure),
            'articles': article_count
        }
        
        print(f"✅ Parsed successfully:")
        print(f"   📚 Chapters: {document_structure['metadata']['parsing_stats']['chapters']}")
        print(f"   📋 Sections: {document_structure['metadata']['parsing_stats']['total_sections']}")  
        print(f"   📜 Articles: {article_count}")
        
        return document_structure
    
    def _clean_content(self, content: str) -> str:
        """Làm sạch content"""
        # Chỉ remove extra spaces trên cùng một dòng, giữ nguyên newlines
        lines = content.split('\n')
        cleaned_lines = [re.sub(r'[ \t]+', ' ', line.strip()) for line in lines]
        return '\n'.join(cleaned_lines)
    
    def _create_section(self, match, line_num: int) -> Dict[str, Any]:
        """Tạo section (mục) object"""
        section_num = match.group(1)
        section_title = match.group(2).strip()
        
        return {
            'type': 'section',
            'number': int(section_num),
            'title': section_title,
            'line_number': line_num,
            'articles': []
        }
    
    def _create_article(self, match, line_num: int, all_lines: List[str]) -> Dict[str, Any]:
        """Tạo article (điều) object với full content"""
        article_num = match.group(1)
        article_title = match.group(2).strip()
        
        # Lấy full content của điều này
        article_content = self._extract_article_content(line_num, all_lines)
        
        return {
            'type': 'article',
            'number': int(article_num),
            'title': article_title,
            'line_number': line_num,
            'content': article_content,
            'content_length': len(article_content),
            'paragraphs': self._split_paragraphs(article_content)
        }
    
    def _extract_article_content(self, start_line: int, all_lines: List[str]) -> str:
        """Trích xuất full content của một điều"""
        content_lines = []
        
        # Bắt đầu từ line hiện tại
        for i in range(start_line, len(all_lines)):
            line = all_lines[i].strip()
            
            # Dừng khi gặp điều tiếp theo
            if i > start_line and re.match(self.patterns['dieu'], line, re.IGNORECASE):
                break
            # Dừng khi gặp chương mới
            if i > start_line and re.match(self.patterns['chuong'], line, re.IGNORECASE):
                break
            # Dừng khi gặp mục mới  
            if i > start_line and re.match(self.patterns['muc'], line, re.IGNORECASE):
                break
                
            if line:
                content_lines.append(line)
        
        return ' '.join(content_lines)
    
    def _split_paragraphs(self, content: str) -> List[str]:
        """Chia content thành các đoạn"""
        # Simple paragraph splitting
        paragraphs = []
        sentences = content.split('.')
        
        current_para = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                current_para.append(sentence)
                # Mỗi 2-3 câu tạo thành 1 paragraph
                if len(current_para) >= 2:
                    paragraphs.append('. '.join(current_para) + '.')
                    current_para = []
        
        # Add remaining
        if current_para:
            paragraphs.append('. '.join(current_para) + '.')
            
        return paragraphs
    
    def _count_sections(self, document: Dict[str, Any]) -> int:
        """Đếm tổng số sections"""
        total = 0
        
        # Sections in chapters
        for chapter in document.get('chapters', []):
            total += len(chapter.get('sections', []))
        
        # Independent sections
        total += len(document.get('independent_sections', []))
        
        return total
    
    def get_all_articles(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Lấy tất cả articles từ document structure (for Monte Carlo sampling)"""
        articles = []
        
        # Articles từ chapters
        for chapter in document.get('chapters', []):
            # Articles trực tiếp trong chapter
            for article in chapter.get('articles', []):
                article['path'] = f"{document['title']} > {chapter['title']}"
                article['document_title'] = document['title']
                article['chapter_title'] = chapter['title']
                articles.append(article)
            
            # Articles trong sections của chapter
            for section in chapter.get('sections', []):
                for article in section.get('articles', []):
                    article['path'] = f"{document['title']} > {chapter['title']} > {section['title']}"
                    article['document_title'] = document['title']
                    article['chapter_title'] = chapter['title'] 
                    article['section_title'] = section['title']
                    articles.append(article)
        
        # Independent sections
        for section in document.get('independent_sections', []):
            for article in section.get('articles', []):
                article['path'] = f"{document['title']} > {section['title']}"
                article['document_title'] = document['title']
                article['section_title'] = section['title']
                articles.append(article)
        
        # Independent articles
        for article in document.get('independent_articles', []):
            article['path'] = document['title']
            article['document_title'] = document['title']
            articles.append(article)
        
        return articles
    
    def monte_carlo_sample_articles(self, articles: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
        """
        Monte Carlo sampling for articles - completely random selection
        Ensures fair coverage over multiple generations
        
        Args:
            articles: List of all available articles
            sample_size: Number of articles to select
            
        Returns:
            List of randomly selected articles
        """
        if not articles or sample_size <= 0:
            return []
            
        if sample_size >= len(articles):
            return articles.copy()
        
        # Pure random sampling without replacement
        return random.sample(articles, sample_size)


def test_parser():
    """Test parser với sample text"""
    
    sample_text = """
    LUẬT GIAO THÔNG ĐƯỜNG BỘ
    
    CHƯƠNG I
    QUY ĐỊNH CHUNG
    
    Điều 1. Phạm vi điều chỉnh
    Luật này quy định về giao thông đường bộ; quyền, nghĩa vụ của tổ chức, cá nhân tham gia giao thông đường bộ; quy tắc giao thông đường bộ; tín hiệu giao thông đường bộ; kết cấu hạ tầng giao thông đường bộ; phương tiện giao thông đường bộ và người lái xe; vận tải đường bộ; thanh tra, xử lý vi phạm pháp luật về giao thông đường bộ.
    
    Điều 2. Giải thích từ ngữ
    Trong Luật này, các từ ngữ dưới đây được hiểu như sau:
    1. Giao thông đường bộ là hoạt động di chuyển của người và phương tiện giao thông qua đường bộ.
    2. Tham gia giao thông đường bộ là hoạt động của người và phương tiện giao thông trên đường bộ.
    
    CHƯƠNG II
    QUYỀN VÀ NGHĨA VỤ CỦA TỔ CHỨC, CÁ NHÂN
    
    Mục 1. Quyền và nghĩa vụ chung
    
    Điều 3. Quyền của tổ chức, cá nhân
    Tổ chức, cá nhân có các quyền sau đây:
    1. Được sử dụng đường bộ an toàn, thông suốt.
    2. Được cung cấp thông tin về giao thông đường bộ.
    
    Điều 4. Nghĩa vụ của tổ chức, cá nhân  
    Tổ chức, cá nhân có các nghĩa vụ sau đây:
    1. Chấp hành quy tắc giao thông đường bộ.
    2. Tham gia bảo vệ kết cấu hạ tầng giao thông đường bộ.
    
    Mục 2. Quyền và nghĩa vụ riêng
    
    Điều 5. Quyền riêng của người điều khiển phương tiện
    Người điều khiển phương tiện giao thông đường bộ có quyền được ưu tiên đi trước trong các trường hợp quy định tại Luật này.
    
    CHƯƠNG III
    QUY TẮC GIAO THÔNG ĐƯỜNG BỘ
    
    Điều 86. Quy định chuyển tiếp
    Luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2009; các quy định trước đây trái với Luật này đều bị bãi bỏ.
    """
    
    parser = LegalDocumentParser()
    
    # Parse document  
    result = parser.parse_document("Luật Giao thông đường bộ", sample_text)
    
    # Pretty print JSON
    print(f"\n📄 PARSED DOCUMENT STRUCTURE:")
    print("=" * 80)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # Test getting all articles
    articles = parser.get_all_articles(result)
    print(f"\n📜 ALL ARTICLES FOR MONTE CARLO SAMPLING:")
    print("=" * 80)
    for i, article in enumerate(articles, 1):
        print(f"{i}. Article {article['number']}: {article['title']}")
        print(f"   Path: {article['path']}")
        print(f"   Content length: {article['content_length']} chars")
        print(f"   Preview: {article['content'][:100]}...")
        
    # Test Monte Carlo sampling
    print(f"\n🎲 MONTE CARLO SAMPLING TEST:")
    print("=" * 80)
    print(f"Total articles available: {len(articles)}")
    
    # Test multiple rounds of Monte Carlo sampling
    sample_sizes = [2, 3, 5]
    
    for sample_size in sample_sizes:
        if sample_size <= len(articles):
            print(f"\n🔥 Sample size: {sample_size}")
            selected = parser.monte_carlo_sample_articles(articles, sample_size)
            for i, article in enumerate(selected, 1):
                print(f"  {i}. Article {article['number']}: {article['title']}")
    
    print(f"\n✅ Parser test completed successfully!")
    print(f"📊 Final Stats:")
    print(f"   - Chapters: {result['metadata']['parsing_stats']['chapters']}")
    print(f"   - Sections: {result['metadata']['parsing_stats']['total_sections']}")
    print(f"   - Articles: {result['metadata']['parsing_stats']['articles']}")
    print(f"   - Content Length: {result['content_length']} chars")

if __name__ == "__main__":
    test_parser()
