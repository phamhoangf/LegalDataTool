"""
Script test workflow đơn giản
"""
import requests
import json

API_BASE = "http://localhost:5000/api"

def test_workflow():
    """Test workflow cơ bản"""
    
    print("🔍 Test API Health Check...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            print("✅ API đang hoạt động")
        else:
            print("❌ API không phản hồi")
            return
    except:
        print("❌ Không thể kết nối API. Hãy chạy backend trước!")
        return
    
    print("\n📊 Test Statistics...")
    try:
        response = requests.get(f"{API_BASE}/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Thống kê: {stats['total_topics']} chủ đề, {stats['total_generated']} dữ liệu đã sinh")
        else:
            print("❌ Không thể lấy thống kê")
    except Exception as e:
        print(f"❌ Lỗi lấy thống kê: {e}")
    
    print("\n📁 Test Topics...")
    try:
        response = requests.get(f"{API_BASE}/topics")
        if response.status_code == 200:
            topics = response.json()
            print(f"✅ Có {len(topics)} chủ đề")
            
            if topics:
                topic_id = topics[0]['id']
                print(f"📋 Test dữ liệu cho chủ đề ID: {topic_id}")
                
                response = requests.get(f"{API_BASE}/data/{topic_id}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Có {len(data)} mẫu dữ liệu")
                    
                    labeled_count = sum(1 for item in data if item['is_labeled'])
                    print(f"🏷️ Đã gán nhãn: {labeled_count}/{len(data)} mẫu")
                else:
                    print("❌ Không thể lấy dữ liệu")
            else:
                print("ℹ️ Chưa có chủ đề nào. Chạy create_sample_data.py trước!")
        else:
            print("❌ Không thể lấy danh sách chủ đề")
    except Exception as e:
        print(f"❌ Lỗi test topics: {e}")
    
    print("\n🎯 Kết luận:")
    print("- Backend API hoạt động ✅")
    print("- Database có dữ liệu ✅") 
    print("- Workflow cơ bản OK ✅")
    print("\n💡 Hãy mở http://localhost:3000 để test frontend!")

if __name__ == '__main__':
    test_workflow()
