# --- Hàm xử lý đầu vào file/crawl cho backend ---
def process_input(input_data, input_type, title, parser=None):
    """
    Xử lý dữ liệu đầu vào (file hoặc crawl):
    - input_data: đường dẫn file hoặc chuỗi text crawl
    - input_type: 'pdf', 'docx', 'txt', 'text' (crawl)
    - title: tiêu đề văn bản
    - parser: instance của LegalDocumentParser (nếu có)
    """
    if input_type in ['pdf', 'docx', 'txt']:
        return process_uploaded_file(input_data, input_type, title, parser)
    elif input_type == 'text':
        if parser is None:
            from document_parsers import LegalDocumentParser
            parser = LegalDocumentParser()
        return parser.parse_document(title, input_data)
    else:
        raise ValueError("Unsupported input type")
# --- Các hàm xử lý file đa nguồn, không ảnh hưởng code gốc ---
import pdfplumber
import docx
import os

def extract_text_from_pdf(pdf_path):
    """Trích xuất text từ file PDF"""
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_text_from_docx(docx_path):
    """Trích xuất text từ file DOCX"""
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(txt_path):
    """Trích xuất text từ file TXT"""
    with open(txt_path, encoding='utf-8') as f:
        return f.read()

def process_uploaded_file(file_path, file_type, title, parser=None):
    """
    Nhận file bất kỳ, chuyển về text, parse bằng LegalDocumentParser
    file_type: 'pdf', 'docx', 'txt'
    title: tiêu đề văn bản
    parser: instance của LegalDocumentParser (nếu có), nếu không sẽ tạo mới
    """
    if file_type == 'pdf':
        text = extract_text_from_pdf(file_path)
    elif file_type == 'docx':
        text = extract_text_from_docx(file_path)
    elif file_type == 'txt':
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file type")
    if parser is None:
        from document_parsers import LegalDocumentParser
        parser = LegalDocumentParser()
    return parser.parse_document(title, text)
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

"""
Legal Document Parser Module - Version with Header Normalization & Quote State Tracking
"""

import re
import pandas as pd
from typing import Dict, List, Any, Tuple

class LegalDocumentParser:
    """
    [NÂNG CẤP LẦN 3] Thêm cơ chế chuẩn hóa header đa dòng (Điều \n 1) và
    xử lý cấu trúc Điều -> Điểm (không có Khoản).
    """
    
    def __init__(self):
        self.patterns = {
            'chuong_header': r'^Chương\s+([IVXLCDM]+)',
            'muc_header': r'^\s*Mục\s+(\d+)',
            'dieu_header': r'(?m)^\s*Điều(?:\s|\n)*(\d+)\.?\s*',
            'khoan_header': r'^\s*(\d+)[\.\-]?\s*(.*)',
            'diem_header': r'^\s*([a-z])\)\s*(.*)'
        }
        self.MIN_UNIT_LENGTH = 800
        self.MAX_UNIT_LENGTH = 1200


    def _normalize_khoan_headers(self, lines: List[str]) -> List[str]:
        """
        [HÀM CŨ] Tiền xử lý để gộp các header của Khoản bị ngắt dòng.
        """
        content_str = "\n".join(lines)
        normalized_content = re.sub(r'\n\s*(\d+)\s*\n\s*\.\s*', r'\n\1. ', content_str)
        return normalized_content.split('\n')

    def _normalize_multiline_headers(self, content: str) -> str:
        """
        [HÀM MỚI] Tiền xử lý để gộp các header chính bị ngắt dòng.
        Ví dụ: "Điều\n1" -> "Điều 1", "Chương\nI" -> "Chương I"
        Sử dụng cờ re.MULTILINE để `^` khớp với đầu mỗi dòng.
        """
        # Gộp dòng cho Chương, Mục, Điều
        content = re.sub(r'^(Chương|Mục|Điều)\s*\n\s*([IVXLCDM\d]+)', r'\1 \2', content, flags=re.MULTILINE | re.IGNORECASE)
        return content

    def parse_document(self, title: str, content: str) -> Dict[str, Any]:
        """[CÓ SỬA LỖI] CẤP 1: Tách khối Điều, bỏ qua các header 'Điều' trong trích dẫn."""
        print(f"📄 Parsing document: {title}")
        content = self._clean_content(content)
        
        # # << BƯỚC SỬA LỖI MỚI >>
        content = self._normalize_multiline_headers(content)

        lines = content.split('\n')
        document_structure = {'title': title, 'articles': []}
        
        article_blocks, current_block_lines = [], []
        parsing_context = {'chuong': None, 'muc': None}
        last_valid_article_num = 0
        in_quote = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            # Loại bỏ dấu chấm sau số thứ tự Điều nếu có, để xử lý nhất quán
            # line = re.sub(r'^(Điều\s+\d+)\.', r'\1', line)
            line = re.sub(r'^\s*(Điều\s+\d+)\.', r'\1', line)

            is_header_candidate = not in_quote
            chuong_match = is_header_candidate and re.match(self.patterns['chuong_header'], line.strip(), re.IGNORECASE)
            if chuong_match:
                if current_block_lines: article_blocks.append((parsing_context.copy(), current_block_lines))
                chuong_id = chuong_match.group(1); i, chuong_title = self._extract_multiline_title(lines, i)
                chuong_title = re.sub(self.patterns['chuong_header'], '', chuong_title, 1, re.IGNORECASE).strip()
                parsing_context['chuong'] = f"Chương {chuong_id}: {chuong_title}"; parsing_context['muc'] = None; 
                # last_valid_article_num = 0; current_block_lines = []
                i += 1; continue
            muc_match = is_header_candidate and re.match(self.patterns['muc_header'], line.strip(), re.IGNORECASE)
            if muc_match:
                if current_block_lines: article_blocks.append((parsing_context.copy(), current_block_lines))
                muc_id = muc_match.group(1); i, muc_title = self._extract_multiline_title(lines, i)
                muc_title = re.sub(self.patterns['muc_header'], '', muc_title, 1, re.IGNORECASE).strip()
                parsing_context['muc'] = f"Mục {muc_id}: {muc_title}"; 
                # last_valid_article_num = 0; current_block_lines = []
                i += 1; continue
            # dieu_match = is_header_candidate and re.match(self.patterns['dieu_header'], line)
            dieu_match = is_header_candidate and re.match(self.patterns['dieu_header'], line.strip())

            is_new_valid_article = False
            if dieu_match and int(dieu_match.group(1)) == last_valid_article_num + 1: is_new_valid_article = True
            if is_new_valid_article:
                if current_block_lines: article_blocks.append((parsing_context.copy(), current_block_lines))
                current_block_lines = [line]; last_valid_article_num = int(dieu_match.group(1))
            elif line.strip() or current_block_lines:
                current_block_lines.append(line)
            
            quote_char_count = line.count('“') + line.count('”')
            # quote_char_count = sum(line.count(q) for q in self.ALL_QUOTES)
            if quote_char_count % 2 != 0: in_quote = not in_quote
            i += 1
        if current_block_lines: article_blocks.append((parsing_context.copy(), current_block_lines))
        for context, block_lines in article_blocks:
            if block_lines and re.match(self.patterns['dieu_header'], block_lines[0]):
                article_object = self._process_article_block(title, context, block_lines)
                if article_object: document_structure['articles'].append(article_object)
        print(f"✅ Parsed successfully: Found and processed {len(document_structure['articles'])} articles."); 
        return document_structure
        # return len(document_structure['articles']), document_structure

    def _parse_article_content(self, article_number: str, base_path: str, article_content_lines: List[str]) -> List[Dict]:
        """[CÓ SỬA LỖI] CẤP 2: Chuẩn hóa header, tách khối Khoản hoặc xử lý trực tiếp Điểm."""
        article_content_lines = self._normalize_khoan_headers(article_content_lines)
        
        units = []
        khoan_blocks_of_lines, current_block = [], []
        last_valid_khoan_num = 0
        in_quote = False
        for line in article_content_lines:
            is_new_valid_khoan = False
            if not in_quote:
                match = re.match(self.patterns['khoan_header'], line.strip())
                if match:
                    current_num = int(match.group(1))
                    if current_num == last_valid_khoan_num + 1: is_new_valid_khoan = True
            if is_new_valid_khoan:
                if current_block: khoan_blocks_of_lines.append(current_block)
                current_block = [line]; last_valid_khoan_num = current_num
            elif line.strip() or current_block:
                if not current_block and not line.strip(): continue
                current_block.append(line)
            quote_char_count = line.count('“') + line.count('”')
            if quote_char_count % 2 != 0: in_quote = not in_quote
        if current_block: khoan_blocks_of_lines.append(current_block)
        
        # <<< SỬA LỖI LOGIC FALLBACK ĐA CẤP >>>
        if last_valid_khoan_num == 0:
            # Nếu không có Khoản, kiểm tra xem có cấu trúc Điểm không
            has_diem_structure = any(re.match(self.patterns['diem_header'], line.strip()) for line in article_content_lines)
            
            if has_diem_structure:
                # Nếu có Điểm, xử lý toàn bộ nội dung Điều như một "Khoản" không tên
                # Ta truyền một list các dòng giả mạo vào _group_diem_within_khoan
                # để hàm đó có thể chạy mà không cần header Khoản thật.
                # Dòng header giả "0. " sẽ bị bỏ qua khi tạo intro_text.
                fake_khoan_lines = ["0. "] + article_content_lines
                diem_units = self._group_diem_within_khoan(article_number, base_path, fake_khoan_lines)
                # Sửa lại path và source_khoan cho đúng
                for unit in diem_units:
                    unit['path'] = unit['path'].replace(" > Khoản 0", "")
                    unit['source_khoan'] = 'N/A'
                return diem_units
            else:
                # Nếu không có Khoản và không có Điểm, coi cả Điều là 1 unit
                full_content = "\n".join(article_content_lines).strip()
                if full_content:
                    units.append({'path': base_path, 'content': full_content, 'content_length_no_spaces': len(re.sub(r'\s', '', full_content)), 'source_article': article_number, 'source_khoan': 'N/A', 'source_diem': 'N/A'})
                return units

        for khoan_lines in khoan_blocks_of_lines:
            khoan_units = self._group_diem_within_khoan(article_number, base_path, khoan_lines)
            units.extend(khoan_units)
        return units

    def _group_diem_within_khoan(self, article_number: str, base_path: str, khoan_lines: List[str]) -> List[Dict]:
        """[CÓ SỬA LỖI] CẤP 3: Xử lý Khoản, bỏ qua header 'Điểm' trong trích dẫn."""
        units = []
        khoan_header_match = re.match(self.patterns['khoan_header'], khoan_lines[0].strip())
        # Sửa đổi: Nếu không khớp header, vẫn có thể xử lý (trường hợp Điều -> Điểm)
        khoan_id = khoan_header_match.group(1) if khoan_header_match else "0"

        diem_blocks_of_lines, current_block = [], []
        # Nếu không có header Khoản, toàn bộ dòng đầu tiên là nội dung
        khoan_intro_lines = [khoan_lines[0]] if khoan_header_match else []
        content_start_index = 1 if khoan_header_match else 0
        
        diem_started = False
        in_quote = False
        for line in khoan_lines[content_start_index:]:
            is_new_diem = False
            if not in_quote:
                match = re.match(self.patterns['diem_header'], line.strip())
                if match and not re.match(r'^\d+[a-z]\.', line.strip()):
                    is_new_diem = True
            if is_new_diem:
                if not diem_started:
                    diem_started = True; current_block = [line]
                else:
                    if current_block: diem_blocks_of_lines.append(current_block)
                    current_block = [line]
            elif diem_started:
                current_block.append(line)
            else:
                khoan_intro_lines.append(line)
            quote_char_count = line.count('“') + line.count('”')
            if quote_char_count % 2 != 0: in_quote = not in_quote
        if current_block: diem_blocks_of_lines.append(current_block)
        
        if not diem_started:
            khoan_content = "\n".join(khoan_lines).strip()
            # Chỉ tạo unit nếu đây là một Khoản thực sự (có header)
            if khoan_content and khoan_header_match:
                units.append({'path': f"{base_path} > Khoản {khoan_id}", 'content': khoan_content, 'content_length_no_spaces': len(re.sub(r'\s', '', khoan_content)), 'source_article': article_number, 'source_khoan': khoan_id, 'source_diem': 'N/A'})
            return units
            
        khoan_intro_text = "\n".join(khoan_intro_lines).strip()
        # Bỏ header giả "0." nếu có
        if khoan_intro_text == "0.": khoan_intro_text = ""
        
        current_group_blocks, current_group_length = [], 0
        for block in diem_blocks_of_lines:
            block_content = "\n".join(block).strip(); block_length = len(re.sub(r'\s', '', block_content))
            if not current_group_blocks:
                current_group_blocks.append(block); current_group_length = block_length
            elif current_group_length + block_length > self.MAX_UNIT_LENGTH and current_group_length >= self.MIN_UNIT_LENGTH:
                unit = self._create_unit_from_diem_group(article_number, base_path, khoan_id, khoan_intro_text, current_group_blocks)
                if unit: units.append(unit)
                current_group_blocks, current_group_length = [block], block_length
            else:
                current_group_blocks.append(block); current_group_length += block_length
            if current_group_length >= self.MIN_UNIT_LENGTH:
                unit = self._create_unit_from_diem_group(article_number, base_path, khoan_id, khoan_intro_text, current_group_blocks)
                if unit: units.append(unit)
                current_group_blocks, current_group_length = [], 0
        if current_group_blocks:
            unit = self._create_unit_from_diem_group(article_number, base_path, khoan_id, khoan_intro_text, current_group_blocks)
            if unit: units.append(unit)
        return units

    # ==============================================================================
    # CÁC HÀM TRỢ GIÚP (KHÔNG THAY ĐỔI)
    # ==============================================================================
    def _extract_multiline_title(self, lines: List[str], start_index: int) -> Tuple[int, str]:
        title_lines = [lines[start_index].strip()]; i = start_index + 1
        while i < len(lines):
            line = lines[i].strip()
            if re.match(self.patterns['dieu_header'], line) or re.match(self.patterns['muc_header'], line) or re.match(self.patterns['khoan_header'], line) or re.match(self.patterns['chuong_header'], line, re.IGNORECASE) or (not line and title_lines): break
            if line: title_lines.append(line)
            i += 1
        return i - 1, ' '.join(title_lines)
    def _process_article_block(self, doc_title: str, context: Dict, article_lines: List[str]) -> Dict:
        match = re.match(self.patterns['dieu_header'], article_lines[0])
        if not match: return None
        
        article_number = match.group(1)
        # Tách tiêu đề có thể có trên cùng dòng với 'Điều X.'
        title_on_first_line = re.sub(self.patterns['dieu_header'], '', article_lines[0]).strip()
        title_lines = [title_on_first_line] if title_on_first_line else []
        
        content_start_index = 1
        for i in range(1, len(article_lines)):
            line_strip = article_lines[i].strip()
            
            # Điều kiện dừng 1: Gặp một cấu trúc rõ ràng như Khoản hoặc Điểm.
            is_content_marker = (
                re.match(self.patterns['khoan_header'], line_strip) or
                re.match(self.patterns['diem_header'], line_strip)
            )
            
            # Điều kiện dừng 2: Gặp một đoạn văn mới.
            # Chỉ coi là đoạn văn mới nếu đã có ít nhất một dòng tiêu đề.
            has_title = any(t.strip() for t in title_lines)
            is_new_paragraph = has_title and re.match(r'^[A-ZÀ-Ỹ]', line_strip)

            if is_content_marker or is_new_paragraph:
                break

            # Nếu chưa dừng, tiếp tục thêm dòng này vào tiêu đề
            if line_strip:
                title_lines.append(line_strip)
            content_start_index = i + 1
            
        article_title = ' '.join(filter(None, title_lines))
        article_content_lines = article_lines[content_start_index:]
        
        path_parts = [doc_title]
        if context.get('chuong'): path_parts.append(context['chuong'])
        if context.get('muc'): path_parts.append(context['muc'])
        path_parts.append(f"Điều {article_number}: {article_title}"); base_path = " > ".join(path_parts)
        
        units = self._parse_article_content(article_number, base_path, article_content_lines)
        return {'number': article_number, 'title': article_title, 'units': units}

    def _create_unit_from_diem_group(self, article_number: str, base_path: str, khoan_id: str, khoan_intro_text: str, group_of_diem_blocks: List[List[str]]) -> Dict:
        if not group_of_diem_blocks: return None
        first_diem_match = re.match(self.patterns['diem_header'], group_of_diem_blocks[0][0].strip()); last_diem_match = re.match(self.patterns['diem_header'], group_of_diem_blocks[-1][0].strip())
        if not first_diem_match or not last_diem_match: return None
        start_diem_id, end_diem_id = first_diem_match.group(1), last_diem_match.group(1)
        diem_range_str = start_diem_id if start_diem_id == end_diem_id else f"{start_diem_id}-{end_diem_id}"
        combined_diem_content = "\n\n".join(["\n".join(block).strip() for block in group_of_diem_blocks])
        final_content = (f"{khoan_intro_text}\n{combined_diem_content}").strip()
        path = f"{base_path} > Khoản {khoan_id} > Điểm {diem_range_str}"
        return {'path': path, 'content': final_content, 'content_length_no_spaces': len(re.sub(r'\s', '', final_content)), 'source_article': article_number, 'source_khoan': khoan_id, 'source_diem': diem_range_str}
    def _clean_content(self, content: str) -> str: 
        return re.sub(r'\r\n|\r', '\n', content)
    
    def get_all_units(self, parsed_data: Dict) -> List[Dict]:
        """Lấy tất cả units từ parsed structure"""
        all_units = []
        
        if 'articles' not in parsed_data:
            return all_units
            
        for article in parsed_data['articles']:
            if 'units' in article:
                for unit in article['units']:
                    # Convert unit format for data generator
                    converted_unit = {
                        'path': unit.get('path', ''),
                        'content': unit.get('content', ''),
                        'content_length': unit.get('content_length_no_spaces', 0),
                        'source_article': unit.get('source_article', ''),
                        'source_khoan': unit.get('source_khoan', 'N/A'),
                        'source_diem': unit.get('source_diem', 'N/A')
                    }
                    all_units.append(converted_unit)
        
        return all_units


