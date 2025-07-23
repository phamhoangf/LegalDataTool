"""
Script tạo dữ liệu mẫu để test hệ thống sử dụng Google AI
"""
import os
from dotenv import load_dotenv
from app import app, db
from models import LegalTopic, GeneratedData, LabeledData
import json

def create_sample_data():
    """Tạo dữ liệu mẫu"""
    load_dotenv()
    
    with app.app_context():
        # Tạo database tables
        db.create_all()
        
        # Tạo chủ đề mẫu
        sample_topic = LegalTopic(
            name="Giấy phép lái xe",
            description="Quy định về giấy phép lái xe các loại phương tiện",
            legal_text="""
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
            """
        )
        
        db.session.add(sample_topic)
        db.session.commit()
        
        # Tạo dữ liệu SFT mẫu
        sft_samples = [
            {
                "instruction": "Độ tuổi tối thiểu để thi giấy phép lái xe hạng A1 là bao nhiêu?",
                "output": "Theo Luật Giao thông đường bộ 2008, độ tuổi tối thiểu để được cấp giấy phép lái xe hạng A1 là đủ 18 tuổi."
            },
            {
                "instruction": "Giấy phép lái xe hạng A1 dùng để lái loại xe nào?",
                "output": "Giấy phép lái xe hạng A1 dùng để lái xe mô tô có dung tích xi-lanh từ 50 cm3 đến dưới 175 cm3."
            },
            {
                "instruction": "Khác biệt giữa GPLX hạng A1 và A2 là gì?",
                "output": "GPLX hạng A1 dành cho xe mô tô 50-175 cm3, yêu cầu từ 18 tuổi. GPLX hạng A2 dành cho xe mô tô từ 175 cm3 trở lên, yêu cầu từ 20 tuổi và có thể lái cả loại xe hạng A1."
            }
        ]
        
        for sample in sft_samples:
            generated_data = GeneratedData(
                topic_id=sample_topic.id,
                data_type='sft',
                content=json.dumps(sample, ensure_ascii=False)
            )
            db.session.add(generated_data)
        
        # Tạo dữ liệu CoT mẫu
        cot_samples = [
            {
                "instruction": "Một người 17 tuổi có thể thi giấy phép lái xe hạng A1 không?",
                "reasoning_steps": [
                    "Bước 1: Xác định yêu cầu độ tuổi cho GPLX hạng A1",
                    "Bước 2: Theo Luật Giao thông đường bộ 2008, độ tuổi tối thiểu cho hạng A1 là đủ 18 tuổi",
                    "Bước 3: So sánh 17 tuổi với yêu cầu 18 tuổi",
                    "Bước 4: Kết luận người 17 tuổi chưa đủ điều kiện về tuổi"
                ],
                "final_answer": "Không, người 17 tuổi chưa thể thi giấy phép lái xe hạng A1 vì chưa đủ 18 tuổi theo quy định."
            }
        ]
        
        for sample in cot_samples:
            generated_data = GeneratedData(
                topic_id=sample_topic.id,
                data_type='cot',
                content=json.dumps(sample, ensure_ascii=False)
            )
            db.session.add(generated_data)
        
        # Tạo dữ liệu RLHF mẫu
        rlhf_samples = [
            {
                "prompt": "Tư vấn thủ tục đổi giấy phép lái xe hết hạn",
                "response_a": "Để đổi GPLX hết hạn, bạn cần: 1) GPLX cũ, 2) CMND/CCCD, 3) Giấy khám sức khỏe, 4) 2 ảnh 3x4, 5) Lệ phí. Nộp hồ sơ tại phòng CSGT hoặc trung tâm đăng kiểm. Thời gian xử lý 3-5 ngày làm việc.",
                "response_b": "Mang GPLX cũ và CMND đến phòng CSGT để đổi mới. Có thể cần khám sức khỏe.",
                "preferred": "A"
            }
        ]
        
        for sample in rlhf_samples:
            generated_data = GeneratedData(
                topic_id=sample_topic.id,
                data_type='rlhf',
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
        print(f"- Tạo {len(sft_samples)} mẫu SFT")
        print(f"- Tạo {len(cot_samples)} mẫu CoT") 
        print(f"- Tạo {len(rlhf_samples)} mẫu RLHF")
        print("- Tạo 3 label mẫu")
        print("\n Cấu hình Google AI:")
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if api_key:
            print(f"API Key: {api_key[:10]}...")
        else:
            print("Chưa có GOOGLE_API_KEY trong .env")
            print("Thêm GOOGLE_API_KEY vào file .env để sinh dữ liệu tự động")

if __name__ == '__main__':
    create_sample_data()
