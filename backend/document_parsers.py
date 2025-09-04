#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legal Document Parser Module
Chứa các class và function để parse văn bản pháp luật thành cấu trúc JSON
"""

import re
import json
import random
from typing import Dict, List, Any

class LegalDocumentParser:
    """Parse văn bản pháp luật thành cấu trúc hierarchical JSON"""
    
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
                
                # Xác định article này thuộc về đâu
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
        
        # Tính toán thống kê cuối cùng
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
        """Làm sạch nội dung văn bản"""
        # Loại bỏ các ký tự thừa
        content = re.sub(r'\r\n', '\n', content)
        content = re.sub(r'\r', '\n', content)
        return content

    def _create_chapter(self, match, line_num: int, chapter_title: str = "") -> Dict[str, Any]:
        """Tạo structure cho chương"""
        number = match.group(1)
        title = match.group(2).strip() or chapter_title
        
        return {
            'type': 'chapter',
            'number': number,
            'title': title,
            'line_number': line_num,
            'sections': [],
            'articles': []  # Articles trực tiếp thuộc chapter (không thuộc section)
        }

    def _create_section(self, match, line_num: int) -> Dict[str, Any]:
        """Tạo structure cho mục"""
        number = match.group(1)
        title = match.group(2).strip()
        
        return {
            'type': 'section',
            'number': number,
            'title': title,
            'line_number': line_num,
            'articles': []
        }

    def _create_article(self, match, line_num: int, all_lines: List[str]) -> Dict[str, Any]:
        """Tạo structure cho điều"""
        number = match.group(1)
        title = match.group(2).strip()
        
        # Extract content cho article này
        content = self._extract_article_content(line_num, all_lines)
        
        # Parse paragraphs
        paragraphs = self._parse_paragraphs(content)
        
        return {
            'type': 'article',
            'number': number,
            'title': title,
            'line_number': line_num,
            'content': content,
            'content_length': len(content),
            'paragraphs': paragraphs,
            'paragraph_count': len(paragraphs)
        }

    def _extract_article_content(self, start_line: int, all_lines: List[str]) -> str:
        """Extract nội dung của một article"""
        content_lines = []
        
        # Bắt đầu từ dòng hiện tại (header của article)
        for i in range(start_line, len(all_lines)):
            line = all_lines[i].strip()
            
            # Stop khi gặp article tiếp theo, chapter, hoặc section
            if i > start_line:  # Skip dòng đầu (header)
                if (re.match(self.patterns['dieu'], line, re.IGNORECASE) or
                    re.match(self.patterns['chuong'], line, re.IGNORECASE) or
                    re.match(self.patterns['muc'], line, re.IGNORECASE)):
                    break
            
            content_lines.append(line)
        
        return '\n'.join(content_lines)

    def _parse_paragraphs(self, content: str) -> List[str]:
        """Parse content thành các paragraphs"""
        lines = content.split('\n')
        paragraphs = []
        
        current_paragraph = []
        for line in lines:
            line = line.strip()
            if line:
                current_paragraph.append(line)
            else:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
        
        # Thêm paragraph cuối nếu có
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return paragraphs

    def _count_sections(self, document: Dict[str, Any]) -> int:
        """Đếm tổng số sections trong document"""
        total = 0
        
        # Đếm sections trong các chapters
        for chapter in document.get('chapters', []):
            total += len(chapter.get('sections', []))
        
        # Đếm independent sections
        total += len(document.get('independent_sections', []))
        
        return total

    def get_all_articles(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract tất cả articles từ document để sử dụng cho Monte Carlo sampling
        
        Args:
            document: Document structure đã parse
            
        Returns:
            List of all articles with metadata
        """
        articles = []
        
        # Articles trong chapters và sections
        for chapter in document.get('chapters', []):
            chapter_path = f"Chương {chapter['number']}"
            
            # Articles trực tiếp trong chapter
            for article in chapter.get('articles', []):
                articles.append({
                    'number': article['number'],
                    'title': article['title'],
                    'content': article['content'],
                    'content_length': article['content_length'],
                    'path': f"{chapter_path}/Điều {article['number']}",
                    'location': {
                        'chapter': chapter['number'],
                        'section': None
                    },
                    'metadata': {
                        'line_number': article['line_number'],
                        'paragraph_count': article.get('paragraph_count', 0)
                    }
                })
            
            # Articles trong sections của chapter
            for section in chapter.get('sections', []):
                section_path = f"{chapter_path}/Mục {section['number']}"
                
                for article in section.get('articles', []):
                    articles.append({
                        'number': article['number'],
                        'title': article['title'],
                        'content': article['content'],
                        'content_length': article['content_length'],
                        'path': f"{section_path}/Điều {article['number']}",
                        'location': {
                            'chapter': chapter['number'],
                            'section': section['number']
                        },
                        'metadata': {
                            'line_number': article['line_number'],
                            'paragraph_count': article.get('paragraph_count', 0)
                        }
                    })
        
        # Independent sections
        for section in document.get('independent_sections', []):
            section_path = f"Mục {section['number']}"
            
            for article in section.get('articles', []):
                articles.append({
                    'number': article['number'],
                    'title': article['title'],
                    'content': article['content'],
                    'content_length': article['content_length'],
                    'path': f"{section_path}/Điều {article['number']}",
                    'location': {
                        'chapter': None,
                        'section': section['number']
                    },
                    'metadata': {
                        'line_number': article['line_number'],
                        'paragraph_count': article.get('paragraph_count', 0)
                    }
                })
        
        # Independent articles
        for article in document.get('independent_articles', []):
            articles.append({
                'number': article['number'],
                'title': article['title'],
                'content': article['content'],
                'content_length': article['content_length'],
                'path': f"Điều {article['number']}",
                'location': {
                    'chapter': None,
                    'section': None
                },
                'metadata': {
                    'line_number': article['line_number'],
                    'paragraph_count': article.get('paragraph_count', 0)
                }
            })
        
        return articles

    def monte_carlo_sample_articles(self, articles: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
        """
        Monte Carlo sampling cho articles
        
        Args:
            articles: List of articles to sample from
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