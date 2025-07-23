from google import genai
from google.genai import types
import json
import random
import os
from typing import List, Dict, Any

class DataGenerator:
    """Class sinh dữ liệu huấn luyện cho LegalSLM sử dụng Google Gemini AI"""
    
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
    
    def generate_sft_data(self, legal_text: str, topic: str, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Sinh dữ liệu SFT (Supervised Fine-Tuning)"""
        
        prompt = f"""
        Dựa trên văn bản luật sau về chủ đề "{topic}":
        
        {legal_text[:2000]}...
        
        Hãy tạo {num_samples} cặp instruction-output cho việc huấn luyện mô hình AI pháp lý.
        
        Yêu cầu:
        - Instruction phải là câu hỏi thực tế về pháp luật
        - Output phải chính xác, dựa trên văn bản luật
        - Trả về format JSON array
        
        Format mong muốn:
        [
            {{
                "instruction": "Câu hỏi về pháp luật",
                "output": "Trả lời chính xác dựa trên luật"
            }}
        ]
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    temperature=0.7,
                    max_output_tokens=4000
                )
            )
            
            content = response.text
            # Tìm JSON trong response
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return self._generate_fallback_sft_data(topic, num_samples)
                
        except Exception as e:
            print(f"Lỗi khi sinh SFT data: {e}")
            return self._generate_fallback_sft_data(topic, num_samples)
    
    def generate_cot_data(self, legal_text: str, topic: str, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Sinh dữ liệu CoT (Chain-of-Thought)"""
        
        prompt = f"""
        Dựa trên văn bản luật sau về chủ đề "{topic}":
        
        {legal_text[:2000]}...
        
        Hãy tạo {num_samples} mẫu dữ liệu Chain-of-Thought cho việc huấn luyện mô hình AI pháp lý.
        
        Yêu cầu:
        - Instruction là câu hỏi phức tạp cần suy luận
        - Reasoning_steps là các bước suy luận rõ ràng
        - Final_answer là kết luận cuối cùng
        
        Format mong muốn:
        [
            {{
                "instruction": "Câu hỏi phức tạp về pháp luật",
                "reasoning_steps": [
                    "Bước 1: Xác định vấn đề...",
                    "Bước 2: Áp dụng điều luật...",
                    "Bước 3: Kết luận..."
                ],
                "final_answer": "Kết luận cuối cùng"
            }}
        ]
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    temperature=0.7,
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
                return self._generate_fallback_cot_data(topic, num_samples)
                
        except Exception as e:
            print(f"Lỗi khi sinh CoT data: {e}")
            return self._generate_fallback_cot_data(topic, num_samples)
    
    def generate_rlhf_data(self, legal_text: str, topic: str, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Sinh dữ liệu RLHF (Reinforcement Learning from Human Feedback)"""
        
        prompt = f"""
        Dựa trên văn bản luật sau về chủ đề "{topic}":
        
        {legal_text[:2000]}...
        
        Hãy tạo {num_samples} mẫu dữ liệu RLHF với 2 câu trả lời khác nhau cho mỗi prompt.
        
        Yêu cầu:
        - Prompt là câu hỏi về tư vấn pháp luật
        - Response_A tốt hơn Response_B (để human feedback chọn)
        - Response_A: chính xác, đầy đủ, dễ hiểu
        - Response_B: có thiếu sót hoặc không chính xác
        
        Format mong muốn:
        [
            {{
                "prompt": "Câu hỏi tư vấn pháp luật",
                "response_a": "Câu trả lời tốt, chính xác",
                "response_b": "Câu trả lời kém hơn",
                "preferred": "A"
            }}
        ]
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    temperature=0.8,
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
                return self._generate_fallback_rlhf_data(topic, num_samples)
                
        except Exception as e:
            print(f"Lỗi khi sinh RLHF data: {e}")
            return self._generate_fallback_rlhf_data(topic, num_samples)
    
    def _generate_fallback_sft_data(self, topic: str, num_samples: int) -> List[Dict[str, Any]]:
        """Tạo dữ liệu SFT mẫu khi API lỗi"""
        templates = [
            {
                "instruction": f"Quy định về {topic} là gì?",
                "output": f"Theo quy định pháp luật hiện hành về {topic}..."
            },
            {
                "instruction": f"Thủ tục liên quan đến {topic} như thế nào?",
                "output": f"Thủ tục {topic} bao gồm các bước sau..."
            },
            {
                "instruction": f"Ai có thẩm quyền quyết định về {topic}?",
                "output": f"Thẩm quyền về {topic} thuộc về..."
            }
        ]
        
        result = []
        for i in range(num_samples):
            template = templates[i % len(templates)]
            result.append({
                "instruction": template["instruction"],
                "output": template["output"] + f" (Mẫu {i+1})"
            })
        
        return result
    
    def _generate_fallback_cot_data(self, topic: str, num_samples: int) -> List[Dict[str, Any]]:
        """Tạo dữ liệu CoT mẫu khi API lỗi"""
        result = []
        for i in range(num_samples):
            result.append({
                "instruction": f"Phân tích trường hợp {topic} số {i+1}",
                "reasoning_steps": [
                    f"Bước 1: Xác định vấn đề về {topic}",
                    f"Bước 2: Áp dụng quy định pháp luật",
                    f"Bước 3: Đưa ra kết luận"
                ],
                "final_answer": f"Kết luận về trường hợp {topic} này"
            })
        
        return result
    
    def _generate_fallback_rlhf_data(self, topic: str, num_samples: int) -> List[Dict[str, Any]]:
        """Tạo dữ liệu RLHF mẫu khi API lỗi"""
        result = []
        for i in range(num_samples):
            result.append({
                "prompt": f"Tư vấn về {topic} trong trường hợp {i+1}",
                "response_a": f"Tư vấn chính xác và đầy đủ về {topic}",
                "response_b": f"Tư vấn sơ sài về {topic}",
                "preferred": "A"
            })
        
        return result
