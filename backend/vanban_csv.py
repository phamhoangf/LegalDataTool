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
        self.ALL_QUOTES = ['“', '”', '"']
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
        """Đếm số điều trong văn bản sử dụng document parser"""
        try:

            content = self.parser._clean_content(content)
            content = self.parser._normalize_multiline_headers(content)

            lines = content.split('\n')
            # 2. Khởi tạo các biến trạng thái tối giản
            article_count = 0
            last_article_num = 0
            in_quote = False # Cờ để theo dõi trạng thái đang trong trích dẫn hay không

            # 3. Duyệt qua từng dòng với logic gọn nhẹ
            for line in lines:
                # Cập nhật trạng thái 'in_quote' cho dòng tiếp theo
                # Logic này giúp bỏ qua các header "Điều X" nằm bên trong một câu trích dẫn
                quote_char_count = sum(line.count(q) for q in self.ALL_QUOTES)
                if quote_char_count % 2 != 0:
                    in_quote = not in_quote

                # Nếu dòng hiện tại nằm trong một trích dẫn, bỏ qua không xử lý
                if in_quote:
                    continue

                # Kiểm tra xem dòng có phải là một header "Điều" hay không
                dieu_match = re.match(self.parser.patterns['dieu_header'], line)
                # dieu_match = self.parser.patterns['dieu_header'].match(line)
                
                if dieu_match:
                    current_article_num = int(dieu_match.group(1))
                    
                    # >> LOGIC CỐT LÕI: Chỉ đếm nếu số Điều là tuần tự (N+1) <<
                    # Điều này giúp loại bỏ các tham chiếu sai ("... theo quy định tại Điều 15 ...")
                    if current_article_num == last_article_num + 1:
                        article_count += 1
                        last_article_num = current_article_num
            
            return article_count
        except Exception as e:
            print(f"Error counting articles: {e}")
            return 0
