"""
Script test kết nối Google AI
"""
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

def test_google_ai():
    """Test kết nối Google AI"""
    load_dotenv()
    
    # Kiểm tra API key
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print(" Không tìm thấy GOOGLE_API_KEY hoặc GEMINI_API_KEY trong file .env")
        return False
    
    print(f" Tìm thấy API key: {api_key[:10]}...")
    
    try:
        # Set environment variable
        os.environ['GEMINI_API_KEY'] = api_key
        
        # Tạo client
        client = genai.Client()
        
        # Test với prompt đơn giản
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents="Xin chào, bạn là ai?",
            config=types.GenerateContentConfig(
                # thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.7,
                max_output_tokens=100
            )
        )
        
        print(" Kết nối Google AI thành công!")
        print(f" Response: {response.text}")
        
        # Test với prompt pháp lý
        legal_prompt = """
        Tạo 1 mẫu dữ liệu SFT về luật giao thông:
        Format JSON:
        {
            "instruction": "Câu hỏi về luật giao thông",
            "output": "Câu trả lời"
        }
        """
        
        legal_response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=legal_prompt,
            config=types.GenerateContentConfig(
                # thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.7,
                max_output_tokens=500
            )
        )
        
        print("\n Test sinh dữ liệu pháp lý thành công!")
        print(f" Legal Response: {legal_response.text}")
        
        return True
        
    except Exception as e:
        print(f"Lỗi kết nối Google AI: {e}")
        print("Hướng dẫn khắc phục:")
        print("1. Kiểm tra API key có đúng không")
        print("2. Kiểm tra quota tại https://aistudio.google.com")
        print("3. Kiểm tra kết nối internet")
        return False

if __name__ == '__main__':
    print("Đang test kết nối Google AI...")
    print("=" * 50)
    success = test_google_ai()
    print("=" * 50)
    if success:
        print("Hệ thống sẵn sàng sử dụng!")
    else:
        print("Cần khắc phục lỗi trước khi sử dụng!")
