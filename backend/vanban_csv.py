import pandas as pd
import os
import re
from typing import List, Dict, Optional
from document_parsers import LegalDocumentParser

class VanBanCSVReader:
    """Module đọc dữ liệu văn bản pháp luật từ file CSV"""
    
    def __init__(self, csv_file_path: str = "data/van_ban_phap_luat_async.csv"):
        self.csv_file_path = csv_file_path
        self.total_rows = 0
        self.parser = LegalDocumentParser()
        self._init_csv_info()
    
    def _init_csv_info(self):
        """Lấy thông tin cơ bản về CSV"""
        if os.path.exists(self.csv_file_path):
            # Count total rows nhanh
            with open(self.csv_file_path, 'r', encoding='utf-8-sig') as f:
                self.total_rows = sum(1 for line in f) - 1  # -1 for header
            print(f"✅ CSV loaded: {self.total_rows} văn bản")
        else:
            print(f"❌ CSV file not found: {self.csv_file_path}")
            self.total_rows = 0
    
    def search_documents(self, search: str = "", limit: int = 20, offset: int = 0) -> List[Dict]:
        """Tìm kiếm văn bản theo tên với pagination"""
        if not os.path.exists(self.csv_file_path):
            return []

        documents = []
        count = 0
        skipped = 0
        row_number = 0  # Track actual row number
        
        # Đọc từng chunk để tiết kiệm memory
        chunk_size = 1000
        for chunk in pd.read_csv(self.csv_file_path, chunksize=chunk_size, encoding='utf-8-sig'):
            for idx, row in chunk.iterrows():
                title = row.get('TenVanBan', '').strip()
                content = row.get('NoiDung', '').strip()
                
                if not title or not content:
                    row_number += 1
                    continue
                
                # Count articles first to filter out invalid documents
                article_count = self._count_articles(content)
                if article_count == 0:  # Skip documents with no articles
                    row_number += 1
                    continue
                
                # Search filter
                if search and search.lower() not in title.lower():
                    row_number += 1
                    continue
                
                # Skip until we reach offset
                if skipped < offset:
                    skipped += 1
                    row_number += 1
                    continue
                
                documents.append({
                    "id": row_number,
                    "title": title,
                    "preview": content[:200] + "..." if len(content) > 200 else content,
                    "content_length": len(content),
                    "article_count": article_count  # Already calculated above
                })
                
                count += 1
                row_number += 1
                
                if count >= limit:
                    break
            
            if count >= limit:
                break
        
        return documents
    
    def get_document_content(self, document_id: int) -> Optional[Dict]:
        """Lấy nội dung đầy đủ văn bản theo ID"""
        try:
            # Đọc chỉ 1 dòng cụ thể
            df_single = pd.read_csv(
                self.csv_file_path,
                skiprows=range(1, document_id + 1) if document_id > 0 else None,
                nrows=1,
                encoding='utf-8-sig'
            )
            
            if df_single.empty:
                return None
                
            row = df_single.iloc[0]
            title = row.get('TenVanBan', '').strip()
            content = row.get('NoiDung', '').strip()
            
            if not title or not content:
                return None
                
            return {
                "title": title,
                "content": content,
                "so_hieu": row.get('SoHieu', '').strip(),
                "url": row.get('URL', '').strip()
            }
        except Exception as e:
            print(f"Error reading document {document_id}: {e}")
            return None
    
    def _count_articles(self, content: str) -> int:
        """Đếm số điều trong văn bản sử dụng regex pattern của LegalDocumentParser"""
        try:
            # Sử dụng pattern từ parser
            pattern = self.parser.patterns['dieu']  # r'Điều\s+(\d+)\.?\s*([^\n\r]*)'
            matches = re.findall(pattern, content)
            return len(matches)
        except Exception as e:
            print(f"Error counting articles: {e}")
            return 0
