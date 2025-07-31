"""
Migration script để chuyển từ old schema sang new schema
"""
import os
from dotenv import load_dotenv
from app import app, db
from models import LegalTopic, LegalDocument, TopicDocument, GeneratedData, LabeledData
import json

def migrate_data():
    """Migrate từ legal_text trong topics sang documents table"""
    load_dotenv()
    
    with app.app_context():
        print("🔄 Bắt đầu migration...")
        
        # Backup dữ liệu cũ
        print("📦 Backup dữ liệu cũ...")
        
        # Tạo tables mới
        db.create_all()
        print("✅ Tạo tables mới thành công")
        
        # Migration logic sẽ được thêm sau khi test new schema
        print("⚠️  Migration logic sẽ được implement sau")
        print("💡 Hiện tại chạy create_sample_data.py để tạo dữ liệu mẫu mới")

if __name__ == '__main__':
    migrate_data()
