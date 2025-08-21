#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Migration Script
Thêm trường parsed_structure vào bảng legal_documents
"""

import sqlite3
import os
import sys

def migrate_database():
    """Thêm trường parsed_structure vào table legal_documents"""
    
    # Database path
    db_path = os.path.join('instance', 'legal_data.db')
    
    if not os.path.exists(db_path):
        print(f"❌ Database không tồn tại: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Kiểm tra xem column đã tồn tại chưa
        cursor.execute("PRAGMA table_info(legal_documents)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'parsed_structure' in columns:
            print("✅ Column 'parsed_structure' đã tồn tại")
            return True
        
        # Thêm column mới
        print("🔄 Thêm column 'parsed_structure' vào table legal_documents...")
        cursor.execute("ALTER TABLE legal_documents ADD COLUMN parsed_structure TEXT")
        
        conn.commit()
        conn.close()
        
        print("✅ Migration hoàn thành!")
        return True
        
    except Exception as e:
        print(f"❌ Migration thất bại: {str(e)}")
        return False

if __name__ == "__main__":
    migrate_database()
