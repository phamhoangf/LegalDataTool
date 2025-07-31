"""
Test script để kiểm tra multi-document generation
"""
import os
from dotenv import load_dotenv
from app import app, db
from models import LegalTopic, LegalDocument, TopicDocument
from data_generator import DataGenerator

def test_multi_document_generation():
    """Test generation với multiple documents"""
    load_dotenv()
    
    with app.app_context():
        # Lấy topic và documents
        topic = LegalTopic.query.filter_by(name="Giấy phép lái xe").first()
        if not topic:
            print("❌ Không tìm thấy topic. Chạy create_sample_data.py trước.")
            return
        
        documents = db.session.query(LegalDocument).join(TopicDocument).filter(
            TopicDocument.topic_id == topic.id
        ).all()
        
        print(f"🎯 Topic: {topic.name}")
        print(f"📄 Documents: {len(documents)}")
        for i, doc in enumerate(documents, 1):
            print(f"   {i}. {doc.title} ({len(doc.content)} chars)")
        
        # Test data generator
        data_generator = DataGenerator()
        
        print("\n🤖 Testing Word Matching generation...")
        try:
            # Test với 1 document
            single_doc_samples = data_generator.generate_word_matching_data(
                documents[0].content, topic.name, 3
            )
            print(f"✅ Single document: {len(single_doc_samples)} samples")
            
            # Test với combined content
            combined_content = "\n\n".join([
                f"--- {doc.title} ---\n{doc.content}" 
                for doc in documents
            ])
            
            if len(combined_content) > 4000:
                combined_content = combined_content[:4000] + "..."
            
            multi_doc_samples = data_generator.generate_word_matching_data(
                combined_content, topic.name, 5
            )
            print(f"✅ Multi document: {len(multi_doc_samples)} samples")
            
            # Display samples
            print("\n📝 Sample từ multi-document:")
            for i, sample in enumerate(multi_doc_samples[:2], 1):
                print(f"\n   Sample {i}:")
                print(f"   Q: {sample['question'][:100]}...")
                print(f"   A: {sample['answer'][:100]}...")
                print(f"   Difficulty: {sample['difficulty']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == '__main__':
    test_multi_document_generation()
