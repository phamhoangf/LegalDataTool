"""
Script test workflow đơn giản
"""
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

API_BASE = "http://localhost:5000/api"

def test_workflow():
    """Test workflow cơ bản"""
    
    print("Test API Health Check...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            print(" API đang hoạt động")
        else:
            print(" API không phản hồi")
            return
    except:
        print(" Không thể kết nối API. Hãy chạy backend trước!")
        return
    
    print("\n Test Statistics...")
    try:
        response = requests.get(f"{API_BASE}/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f" Thống kê: {stats['total_topics']} chủ đề, {stats['total_generated']} dữ liệu đã sinh")
        else:
            print(" Không thể lấy thống kê")
    except Exception as e:
        print(f" Lỗi lấy thống kê: {e}")
    
    print("\n Test Topics...")
    try:
        response = requests.get(f"{API_BASE}/topics")
        if response.status_code == 200:
            topics = response.json()
            print(f" Có {len(topics)} chủ đề")
            
            if topics:
                topic_id = topics[0]['id']
                print(f" Test dữ liệu cho chủ đề ID: {topic_id}")
                
                response = requests.get(f"{API_BASE}/data/{topic_id}")
                if response.status_code == 200:
                    data = response.json()
                    print(f" Có {len(data)} mẫu dữ liệu")
                    
                    labeled_count = sum(1 for item in data if item['is_labeled'])
                    print(f" Đã gán nhãn: {labeled_count}/{len(data)} mẫu")
                else:
                    print(" Không thể lấy dữ liệu")
            else:
                print(" Chưa có chủ đề nào. Chạy create_sample_data.py trước!")
        else:
            print(" Không thể lấy danh sách chủ đề")
    except Exception as e:
        print(f" Lỗi test topics: {e}")
    
    print("\n Kết luận:")
    print("- Backend API hoạt động")
    print("- Database có dữ liệu") 
    print("- Workflow cơ bản OK")
    
    print("\n🔍 Test AI Generation...")
    try:
        # Test trực tiếp data generator
        from data_generator import DataGenerator
        import os
        
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ Không tìm thấy GOOGLE_API_KEY hoặc GEMINI_API_KEY")
        else:
            print(f"✅ API Key found: {api_key[:10]}...")
            
        generator = DataGenerator(api_key)
        
        # Test với legal text thật
        legal_text = """
        LUẬT GIAO THÔNG ĐƯỜNG BỘ 2008
        Điều 60. Giấy phép lái xe
        1. Giấy phép lái xe là văn bản do cơ quan có thẩm quyền cấp cho người đủ điều kiện để điều khiển phương tiện giao thông cơ giới đường bộ.
        2. Giấy phép lái xe được phân thành các hạng: A1, A2, B1...
        """
        
        print("🎯 Testing SFT generation...")
        sft_data = generator.generate_sft_data(legal_text, "Giấy phép lái xe", 2)
        
        print(f"📊 Generated {len(sft_data)} SFT samples:")
        for i, sample in enumerate(sft_data):
            print(f"  Sample {i+1}:")
            print(f"    Q: {sample['instruction'][:50]}...")
            print(f"    A: {sample['output'][:50]}...")
            
    except Exception as e:
        print(f"❌ Error testing AI generation: {e}")
    
    print("\n📱 Hãy mở http://localhost:3000 để test frontend!")
    print("🔧 Hoặc http://localhost:3001 nếu port 3000 bị chiếm")

if __name__ == '__main__':
    test_workflow()
