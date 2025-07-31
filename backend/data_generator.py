from google import genai
from google.genai import types
import json
import random
import os
from typing import List, Dict, Any

class DataGenerator:
    """Class sinh dữ liệu huấn luyện cho LegalSLM theo độ khó reasoning"""
    
    def __init__(self, api_key: str = None):
        # Set API key từ parameter hoặc environment variable
        if api_key:
            os.environ['GEMINI_API_KEY'] = api_key
        elif not os.environ.get('GEMINI_API_KEY'):
            # Fallback to GOOGLE_API_KEY
            google_key = os.environ.get('GOOGLE_API_KEY')
            if google_key:
                os.environ['GEMINI_API_KEY'] = google_key
        
        self.client = genai.Client()
        self.model = "gemini-2.0-flash-exp"
    
    def generate_word_matching_data(self, legal_text: str, topic: str, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Sinh dữ liệu Word Matching - đơn giản nhất, chỉ cần tìm từ khóa trong văn bản"""
        
        prompt = f"""
        Dựa trên văn bản luật sau về chủ đề "{topic}":
        
        {legal_text}
        
        Hãy tạo {num_samples} câu hỏi dạng Word Matching - đây là loại câu hỏi đơn giản nhất, chỉ cần tìm kiếm từ khóa/cụm từ trong văn bản.
        
        Yêu cầu:
        - Câu hỏi có thể trả lời bằng cách tìm kiếm trực tiếp trong văn bản
        - Không cần hiểu sâu về khái niệm pháp lý
        - Thông tin cần thiết nằm rõ ràng trong văn bản
        - MỖI CÂU HỎI PHẢI ĐỘC LẬP, KHÔNG ĐƯỢC DÙNG "luật này", "văn bản này" mà phải nói rõ tên luật/văn bản cụ thể
        - Câu hỏi phải rõ ràng, người đọc không cần biết context trước
        
        Format mong muốn (CHỈ 3 TRƯỜNG):
        [
            {{
                "question": "Câu hỏi rõ ràng, độc lập, có tên luật cụ thể",
                "answer": "Trả lời chính xác từ văn bản",
                "difficulty": "word_matching"
            }}
        ]
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,  # Thấp hơn để đảm bảo tính chính xác
                    max_output_tokens=4000
                )
            )
            
            content = response.text
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return self._generate_fallback_word_matching_data(topic, num_samples)
                
        except Exception as e:
            print(f"Lỗi khi sinh Word Matching data: {e}")
            return self._generate_fallback_word_matching_data(topic, num_samples)
    
    def generate_concept_understanding_data(self, legal_text: str, topic: str, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Sinh dữ liệu Concept Understanding - cần hiểu khái niệm pháp lý"""
        
        prompt = f"""
        Dựa trên văn bản luật sau về chủ đề "{topic}":
        
        {legal_text}
        
        Hãy tạo {num_samples} câu hỏi dạng Concept Understanding - yêu cầu hiểu các khái niệm pháp lý để trả lời.
        
        Yêu cầu:
        - Câu hỏi yêu cầu hiểu ý nghĩa của các thuật ngữ pháp lý
        - Cần nắm được khái niệm để áp dụng vào tình huống cụ thể
        - Không chỉ tìm từ khóa mà phải hiểu nghĩa sâu hơn
        - MỖI CÂU HỎI PHẢI ĐỘC LẬP, KHÔNG ĐƯỢC DÙNG "luật này", "văn bản này" mà phải nói rõ tên luật/văn bản cụ thể
        - Câu hỏi phải rõ ràng, người đọc không cần biết context trước
        
        Format mong muốn (CHỈ 3 TRƯỜNG):
        [
            {{
                "question": "Câu hỏi rõ ràng yêu cầu hiểu khái niệm pháp lý, có tên luật cụ thể",
                "answer": "Trả lời dựa trên hiểu biết về khái niệm",
                "difficulty": "concept_understanding"
            }}
        ]
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=4000
                )
            )
            
            content = response.text
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return self._generate_fallback_concept_understanding_data(topic, num_samples)
                
        except Exception as e:
            print(f"Lỗi khi sinh Concept Understanding data: {e}")
            return self._generate_fallback_concept_understanding_data(topic, num_samples)
    
    def generate_multi_paragraph_reading_data(self, legal_text: str, topic: str, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Sinh dữ liệu Multi-Paragraph Reading - cần đọc nhiều đoạn để tập hợp thông tin"""
        
        prompt = f"""
        Dựa trên văn bản luật sau về chủ đề "{topic}":
        
        {legal_text}
        
        Hãy tạo {num_samples} câu hỏi dạng Multi-Paragraph Reading - yêu cầu đọc và tổng hợp thông tin từ nhiều đoạn văn khác nhau.
        
        Yêu cầu:
        - Câu hỏi không thể trả lời chỉ bằng một đoạn văn duy nhất
        - Cần tập hợp thông tin từ 2-3 đoạn văn khác nhau
        - Phải kết hợp các thông tin để đưa ra câu trả lời hoàn chỉnh
        - MỖI CÂU HỎI PHẢI ĐỘC LẬP, KHÔNG ĐƯỢC DÙNG "luật này", "văn bản này" mà phải nói rõ tên luật/văn bản cụ thể
        - Câu hỏi phải rõ ràng, người đọc không cần biết context trước
        
        Format mong muốn (CHỈ 3 TRƯỜNG):
        [
            {{
                "question": "Câu hỏi rõ ràng cần đọc nhiều đoạn văn, có tên luật cụ thể",
                "answer": "Trả lời tổng hợp từ nhiều nguồn",
                "difficulty": "multi_paragraph_reading"
            }}
        ]
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.6,
                    max_output_tokens=4000
                )
            )
            
            content = response.text
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return self._generate_fallback_multi_paragraph_data(topic, num_samples)
                
        except Exception as e:
            print(f"Lỗi khi sinh Multi-Paragraph Reading data: {e}")
            return self._generate_fallback_multi_paragraph_data(topic, num_samples)

    def generate_multi_hop_reasoning_data(self, legal_text: str, topic: str, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Sinh dữ liệu Multi-Hop Reasoning - phức tạp nhất, cần nhiều bước suy luận logic"""
        
        prompt = f"""
        Dựa trên văn bản luật sau về chủ đề "{topic}":
        
        {legal_text}
        
        Hãy tạo {num_samples} câu hỏi dạng Multi-Hop Reasoning - phức tạp nhất, yêu cầu nhiều bước suy luận logic.
        
        Yêu cầu:
        - Câu hỏi cần nhiều bước suy luận logic để trả lời
        - Phải kết hợp hiểu khái niệm + đọc nhiều đoạn + suy luận logic
        - Quá trình reasoning phải rõ ràng và có thể giải thích được
        - MỖI CÂU HỎI PHẢI ĐỘC LẬP, KHÔNG ĐƯỢC DÙNG "luật này", "văn bản này" mà phải nói rõ tên luật/văn bản cụ thể
        - Câu hỏi phải rõ ràng, người đọc không cần biết context trước
        
        Format mong muốn (CHỈ 3 TRƯỜNG):
        [
            {{
                "question": "Câu hỏi phức tạp cần suy luận nhiều bước, có tên luật cụ thể",
                "answer": "Kết luận cuối cùng với giải thích quá trình suy luận",
                "difficulty": "multi_hop_reasoning"
            }}
        ]
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=5000
                )
            )
            
            content = response.text
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return self._generate_fallback_multi_hop_data(topic, num_samples)
                
        except Exception as e:
            print(f"Lỗi khi sinh Multi-Hop Reasoning data: {e}")
            return self._generate_fallback_multi_hop_data(topic, num_samples)
    
    def _generate_fallback_word_matching_data(self, topic: str, num_samples: int) -> List[Dict[str, Any]]:
        """Tạo dữ liệu Word Matching mẫu khi API lỗi"""
        templates = [
            {
                "question": f"Theo Luật Giao thông đường bộ, {topic} là gì?",
                "answer": f"Theo Luật Giao thông đường bộ, {topic} được quy định là...",
                "difficulty": "word_matching"
            },
            {
                "question": f"Ai có thẩm quyền quyết định về {topic} theo quy định của pháp luật?",
                "answer": f"Thẩm quyền về {topic} thuộc về cơ quan...",
                "difficulty": "word_matching"
            }
        ]
        
        result = []
        for i in range(num_samples):
            template = templates[i % len(templates)]
            result.append({
                "question": template["question"],
                "answer": template["answer"] + f" (Mẫu {i+1})",
                "difficulty": template["difficulty"]
            })
        
        return result
    
    def _generate_fallback_concept_understanding_data(self, topic: str, num_samples: int) -> List[Dict[str, Any]]:
        """Tạo dữ liệu Concept Understanding mẫu khi API lỗi"""
        result = []
        for i in range(num_samples):
            result.append({
                "question": f"Theo Luật Giao thông đường bộ, trong trường hợp nào thì hành vi liên quan đến {topic} được coi là vi phạm?",
                "answer": f"Theo quy định của Luật Giao thông đường bộ, hành vi vi phạm về {topic} bao gồm các trường hợp...",
                "difficulty": "concept_understanding"
            })
        
        return result
    
    def _generate_fallback_multi_paragraph_data(self, topic: str, num_samples: int) -> List[Dict[str, Any]]:
        """Tạo dữ liệu Multi-Paragraph Reading mẫu khi API lỗi"""
        result = []
        for i in range(num_samples):
            result.append({
                "question": f"Theo Luật Giao thông đường bộ, quy trình hoàn chỉnh để xử lý vấn đề {topic} như thế nào?",
                "answer": f"Theo quy định của Luật Giao thông đường bộ, quy trình xử lý {topic} bao gồm nhiều giai đoạn...",
                "difficulty": "multi_paragraph_reading"
            })
        
        return result
    
    def _generate_fallback_multi_hop_data(self, topic: str, num_samples: int) -> List[Dict[str, Any]]:
        """Tạo dữ liệu Multi-Hop Reasoning mẫu khi API lỗi"""
        result = []
        for i in range(num_samples):
            result.append({
                "question": f"Theo Luật Giao thông đường bộ, phân tích tình huống phức tạp về {topic} và đưa ra giải pháp pháp lý phù hợp",
                "answer": f"Kết luận về tình huống {topic}: Dựa trên việc xác định các khái niệm pháp lý liên quan, tìm hiểu quy định từ nhiều điều luật khác nhau, phân tích mối quan hệ giữa các quy định, và áp dụng logic pháp lý để kết luận...",
                "difficulty": "multi_hop_reasoning"
            })
        
        return result
