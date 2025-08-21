#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legal Document Parser for DataTool
Parse Vietnamese legal document        # Tính toán thống kê cuối cùng
        document_structure['total_articles'] = article_count
        document_structure['metadata']['parsing_stats'] = {
            'chapters': len(document_structure.get('chapters', [])),
            'total_sections': self._count_sections(document_structure),
            'articles': article_count
        }
        
        # Thêm list tất cả articles vào structure để sử dụng cho Monte Carlo
        document_structure['articles'] = self.get_all_articles(document_structure)
        
        print(f"✅ Parsed successfully:")
        print(f"   📚 Chapters: {document_structure['metadata']['parsing_stats']['chapters']}")
        print(f"   📋 Sections: {document_structure['metadata']['parsing_stats']['total_sections']}")
        print(f"   📜 Articles: {document_structure['metadata']['parsing_stats']['articles']}")
        
        return document_structurehical JSON structure
"""

import re
import json
import random
from typing import Dict, List, Any

class LegalDocumentParser:
    """Parse văn bản pháp luật thành cấu trúc hierarchical"""
    
    def __init__(self):
        # Patterns để nhận diện các cấp độ
        self.patterns = {
            'chuong': r'CHƯƠNG\s+([IVXLC]+|[0-9]+)\.?\s*([^\n\r]*)',
            'muc': r'Mục\s+(\d+)\.?\s*([^\n\r]*)', 
            'dieu': r'Điều\s+(\d+)\.?\s*([^\n\r]*)'
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
        
        lines = content.strip().split('\n')
        
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
        
        # State tracking
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
        
        # Thêm list tất cả articles vào structure để sử dụng cho Monte Carlo
        document_structure['articles'] = self.get_all_articles(document_structure)
        
        print(f"✅ Parsed successfully:")
        print(f"   📚 Chapters: {document_structure['metadata']['parsing_stats']['chapters']}")
        print(f"   📋 Sections: {document_structure['metadata']['parsing_stats']['total_sections']}")
        print(f"   📜 Articles: {document_structure['metadata']['parsing_stats']['articles']}")
        
        return document_structure
    
    def _clean_content(self, content: str) -> str:
        """Làm sạch content"""
        # Chỉ remove extra spaces trên cùng một dòng, giữ nguyên newlines
        lines = content.split('\n')
        cleaned_lines = [re.sub(r'[ \t]+', ' ', line.strip()) for line in lines]
        return '\n'.join(cleaned_lines)
    
    def _create_chapter(self, match, line_num: int, chapter_title: str = "") -> Dict[str, Any]:
        """Tạo chapter object"""
        chapter_num = match.group(1)
        
        return {
            'type': 'chapter',
            'number': chapter_num,
            'title': chapter_title,
            'line_number': line_num,
            'sections': [],
            'articles': []
        }
    
    def _create_section(self, match, line_num: int) -> Dict[str, Any]:
        """Tạo section object"""
        section_num = int(match.group(1))
        section_title = match.group(2).strip()
        
        return {
            'type': 'section',
            'number': section_num,
            'title': section_title,
            'line_number': line_num,
            'articles': []
        }
    
    def _create_article(self, match, line_num: int, all_lines: List[str]) -> Dict[str, Any]:
        """Tạo article object với full content"""
        article_num = int(match.group(1))
        article_title = match.group(2).strip()
        
        # Lấy full content của article
        content = self._extract_article_content(line_num, all_lines)
        
        # Parse thành paragraphs
        paragraphs = self._parse_paragraphs(content)
        
        return {
            'type': 'article',
            'number': article_num,
            'title': article_title,
            'line_number': line_num,
            'content': content,
            'content_length': len(content),
            'paragraphs': paragraphs
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
    
    def _parse_paragraphs(self, content: str) -> List[str]:
        """Parse content thành paragraphs"""
        # Split theo số thứ tự (1., 2., 3., ...)
        paragraphs = re.split(r'\s+(?=\d+\.)', content)
        return [p.strip() for p in paragraphs if p.strip()]
    
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
        document_title = document.get('title', 'Document')
        
        # Articles in chapters
        for chapter in document.get('chapters', []):
            chapter_name = f"Chương {chapter['number']}: {chapter['title']}" if chapter['title'] else f"Chương {chapter['number']}"
            
            # Articles trực tiếp thuộc chapter
            for article in chapter.get('articles', []):
                article_info = {
                    'number': article['number'],
                    'title': article['title'],
                    'content': article['content'],
                    'content_length': article['content_length'],
                    'path': f"{document_title}, {chapter_name}"
                }
                articles.append(article_info)
            
            # Articles trong sections của chapter
            for section in chapter.get('sections', []):
                section_name = f"Mục {section['number']}: {section['title']}"
                for article in section.get('articles', []):
                    article_info = {
                        'number': article['number'],
                        'title': article['title'],
                        'content': article['content'],
                        'content_length': article['content_length'],
                        'path': f"{document_title}, {chapter_name}, {section_name}"
                    }
                    articles.append(article_info)
        
        # Independent sections
        for section in document.get('independent_sections', []):
            section_name = f"Mục {section['number']}: {section['title']}"
            for article in section.get('articles', []):
                article_info = {
                    'number': article['number'],
                    'title': article['title'],
                    'content': article['content'],
                    'content_length': article['content_length'],
                    'path': f"{document_title}, {section_name}"
                }
                articles.append(article_info)
        
        # Independent articles
        for article in document.get('independent_articles', []):
            article_info = {
                'number': article['number'],
                'title': article['title'],
                'content': article['content'],
                'content_length': article['content_length'],
                'path': f"{document_title}"
            }
            articles.append(article_info)
        
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
