"""
Script tạo dữ liệu mẫu để test hệ thống sử dụng Google AI
"""
import os
from dotenv import load_dotenv
from app import app, db
from models import LegalTopic, LegalDocument, TopicDocument, GeneratedData, LabeledData
import json

def create_sample_data():
    """Tạo dữ liệu mẫu"""
    load_dotenv()
    
    with app.app_context():
        # Xóa tất cả tables cũ và tạo lại
        db.drop_all()
        db.create_all()
        
        # Tạo chủ đề mẫu
        sample_topic = LegalTopic(
            name="Giấy phép lái xe",
            description="Quy định về giấy phép lái xe các loại phương tiện"
        )
        
        db.session.add(sample_topic)
        db.session.flush()  # Để có ID
        
        # Tạo document 1 - Luật Giao thông đường bộ
        document1 = LegalDocument(
            title="Luật Giao thông đường bộ 2008 - Điều 60",
            content="""
            LUẬT GIAO THÔNG ĐƯỜNG BỘ 2008
            
            Điều 60. Giấy phép lái xe
            1. Giấy phép lái xe là văn bản do cơ quan có thẩm quyền cấp cho người đủ điều kiện 
            để điều khiển phương tiện giao thông cơ giới đường bộ.
            
            2. Giấy phép lái xe được phân thành các hạng sau đây:
            a) Hạng A1: Xe mô tô có dung tích xi-lanh từ 50 cm3 đến dưới 175 cm3;
            b) Hạng A2: Xe mô tô có dung tích xi-lanh từ 175 cm3 trở lên và các loại xe quy định tại hạng A1;
            c) Hạng B1: Xe ô tô không hành nghề lái xe, có trọng tải thiết kế dưới 3.500 kg;
            
            3. Độ tuổi tối thiểu để được cấp giấy phép lái xe:
            a) Hạng A1: đủ 18 tuổi;
            b) Hạng A2: đủ 20 tuổi;
            c) Hạng B1: đủ 18 tuổi;
            """,
            document_type="law",
            document_number="23/2008/QH12",
            uploaded_by="system"
        )
        
        # Tạo document 2 - Nghị định về đào tạo lái xe
        document2 = LegalDocument(
            title="Nghị định 12/2017/NĐ-CP - Đào tạo lái xe",
            content="""
            NGHỊ ĐỊNH VỀ ĐÀO TẠO, SÁT HẠCH, CẤP GIẤY PHÉP LÁI XE
            
            Điều 15. Điều kiện đào tạo lái xe
            1. Người học lái xe phải đáp ứng các điều kiện:
            a) Có độ tuổi phù hợp với từng hạng giấy phép lái xe;
            b) Có đủ sức khỏe để điều khiển phương tiện theo quy định;
            c) Có trình độ văn hóa tối thiểu là biết đọc, biết viết tiếng Việt;
            
            2. Thời gian đào tạo lý thuyết và thực hành:
            a) Hạng A1: Lý thuyết 18 giờ, thực hành 8 giờ;
            b) Hạng A2: Lý thuyết 20 giờ, thực hành 12 giờ;
            c) Hạng B1: Lý thuyết 58 giờ, thực hành 36 giờ;
            
            3. Học phí đào tạo lái xe do trung tâm đào tạo quy định.
            """,
            document_type="decree",
            document_number="12/2017/NĐ-CP",
            uploaded_by="system"
        )
        
        # Tạo document 3 - Thông tư về sát hạch
        document3 = LegalDocument(
            title="Thông tư 58/2020/TT-BCA - Sát hạch lái xe",
            content="""
            THÔNG TƯ QUY ĐỊNH VỀ SÁT HẠCH LÁI XE
            
            Điều 12. Nội dung sát hạch
            1. Sát hạch lý thuyết bằng hình thức trắc nghiệm trên máy tính:
            a) Hạng A1: 25 câu, thời gian 19 phút, đạt từ 21/25 câu;
            b) Hạng A2: 25 câu, thời gian 19 phút, đạt từ 21/25 câu;
            c) Hạng B1: 35 câu, thời gian 22 phút, đạt từ 32/35 câu;
            
            2. Sát hạch thực hành trên đường:
            a) Hạng A1, A2: Đi trên đường thử nghiệm, thời gian tối thiểu 8 phút;
            b) Hạng B1: Đi trên đường thử nghiệm, thời gian tối thiểu 15 phút;
            
            3. Kết quả sát hạch có hiệu lực trong 12 tháng.
            """,
            document_type="circular",
            document_number="58/2020/TT-BCA",
            uploaded_by="system"
        )
        
        db.session.add_all([document1, document2, document3])
        db.session.flush()  # Để có ID
        
        # Liên kết tất cả documents với topic
        topic_docs = [
            TopicDocument(topic_id=sample_topic.id, document_id=document1.id, relevance_score=1.0, added_by="system"),
            TopicDocument(topic_id=sample_topic.id, document_id=document2.id, relevance_score=0.9, added_by="system"),
            TopicDocument(topic_id=sample_topic.id, document_id=document3.id, relevance_score=0.8, added_by="system")
        ]
        
        db.session.add_all(topic_docs)
        db.session.commit()
        
        # Tạo dữ liệu Word Matching mẫu (đơn giản nhất)
        word_matching_samples = [
            {
                "question": "Theo Luật Giao thông đường bộ 2008 Điều 60, độ tuổi tối thiểu để được cấp giấy phép lái xe hạng A1 là bao nhiêu?",
                "answer": "Theo Luật Giao thông đường bộ 2008 Điều 60, độ tuổi tối thiểu để được cấp giấy phép lái xe hạng A1 là đủ 18 tuổi.",
                "difficulty": "word_matching"
            },
            {
                "question": "Theo Luật Giao thông đường bộ 2008 Điều 60, giấy phép lái xe hạng A1 dùng để lái loại xe nào?",
                "answer": "Giấy phép lái xe hạng A1 dùng để lái xe mô tô có dung tích xi-lanh từ 50 cm3 đến dưới 175 cm3.",
                "difficulty": "word_matching"
            }
        ]
        
        for sample in word_matching_samples:
            generated_data = GeneratedData(
                topic_id=sample_topic.id,
                data_type='word_matching',
                content=json.dumps(sample, ensure_ascii=False)
            )
            db.session.add(generated_data)
        
        # Tạo dữ liệu Concept Understanding mẫu
        concept_understanding_samples = [
            {
                "question": "Theo Luật Giao thông đường bộ 2008, tại sao giấy phép lái xe hạng A2 yêu cầu độ tuổi cao hơn hạng A1?",
                "answer": "Vì xe mô tô hạng A2 có dung tích xi-lanh từ 175 cm3 trở lên, mạnh hơn và nguy hiểm hơn xe hạng A1 (50-175 cm3), nên cần người lái có kinh nghiệm và sự trưởng thành hơn, do đó yêu cầu đủ 20 tuổi thay vì 18 tuổi.",
                "difficulty": "concept_understanding"
            }
        ]
        
        for sample in concept_understanding_samples:
            generated_data = GeneratedData(
                topic_id=sample_topic.id,
                data_type='concept_understanding',
                content=json.dumps(sample, ensure_ascii=False)
            )
            db.session.add(generated_data)
        
        # Tạo dữ liệu Multi-Paragraph Reading mẫu
        multi_paragraph_samples = [
            {
                "question": "Theo Luật Giao thông đường bộ 2008 Điều 60, một người 19 tuổi có thể lái được những loại xe nào và cần giấy phép lái xe hạng gì?",
                "answer": "Một người 19 tuổi có thể lái xe mô tô từ 50 cm3 đến dưới 175 cm3 (cần GPLX hạng A1) và xe ô tô không hành nghề có trọng tải dưới 3.500 kg (cần GPLX hạng B1). Tuy nhiên, chưa thể lái xe mô tô từ 175 cm3 trở lên vì cần đủ 20 tuổi để có GPLX hạng A2.",
                "difficulty": "multi_paragraph_reading"
            }
        ]
        
        for sample in multi_paragraph_samples:
            generated_data = GeneratedData(
                topic_id=sample_topic.id,
                data_type='multi_paragraph_reading',
                content=json.dumps(sample, ensure_ascii=False)
            )
            db.session.add(generated_data)
        
        # Tạo dữ liệu Multi-Hop Reasoning mẫu (phức tạp nhất)
        multi_hop_samples = [
            {
                "question": "Theo Luật Giao thông đường bộ 2008, nếu một công ty muốn tuyển lái xe cho đoàn xe gồm cả mô tô 150 cm3 và ô tô tải nhẹ 2 tấn, họ cần tuyển người có độ tuổi và bằng lái như thế nào để tối ưu chi phí nhân sự?",
                "answer": "Để tối ưu chi phí, công ty nên tuyển người từ 18 tuổi trở lên có GPLX hạng B1, vì theo Điều 60: GPLX B1 cho phép lái ô tô dưới 3.500 kg (bao gồm xe tải 2 tấn), và người có B1 thường có thể lái cả mô tô 150 cm3 nếu có thêm A1. Tuy nhiên, nếu muốn một người lái được cả hai loại xe, cần tuyển người có cả A1 (cho mô tô 150 cm3) và B1 (cho ô tô tải), hoặc tuyển riêng từng loại lái xe chuyên biệt.",
                "difficulty": "multi_hop_reasoning"
            }
        ]
        
        for sample in multi_hop_samples:
            generated_data = GeneratedData(
                topic_id=sample_topic.id,
                data_type='multi_hop_reasoning',
                content=json.dumps(sample, ensure_ascii=False)
            )
            db.session.add(generated_data)
        
        db.session.commit()
        
        # Tạo một số label mẫu
        all_generated = GeneratedData.query.all()
        for i, data in enumerate(all_generated[:3]):  # Label 3 mẫu đầu
            label = ['accept', 'modify', 'accept'][i % 3]
            labeled_data = LabeledData(
                generated_data_id=data.id,
                label=label,
                notes=f"Sample label {i+1}"
            )
            db.session.add(labeled_data)
        
        db.session.commit()
        
        print("Dữ liệu mẫu đã được tạo thành công!")
        print(f"- Tạo 1 chủ đề: {sample_topic.name}")
        print(f"- Tạo 3 tài liệu: Luật, Nghị định, Thông tư")
        print(f"- Tạo {len(word_matching_samples)} mẫu Word Matching")
        print(f"- Tạo {len(concept_understanding_samples)} mẫu Concept Understanding") 
        print(f"- Tạo {len(multi_paragraph_samples)} mẫu Multi-Paragraph Reading")
        print(f"- Tạo {len(multi_hop_samples)} mẫu Multi-Hop Reasoning")
        print("- Tạo 3 label mẫu")
        print("\n📊 Tổng quan dữ liệu:")
        print(f"   🎯 Topics: 1")
        print(f"   📄 Documents: 3")
        print(f"   🔗 Topic-Document links: 3")
        print(f"   💾 Generated samples: {len(word_matching_samples) + len(concept_understanding_samples) + len(multi_paragraph_samples) + len(multi_hop_samples)}")
        print(f"   🏷️  Labeled samples: 3")
        print("\n Cấu hình Google AI:")
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if api_key:
            print(f"API Key: {api_key[:10]}...")
        else:
            print("Chưa có GOOGLE_API_KEY trong .env")
            print("Thêm GOOGLE_API_KEY vào file .env để sinh dữ liệu tự động")

if __name__ == '__main__':
    create_sample_data()
