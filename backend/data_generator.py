from google import genai
from google.genai import types
import json
import random
import os
import re
import time  # Add time import for rate limiting
import requests  # For cloud HuggingFace API calls
from typing import List, Dict, Any
from pydantic import BaseModel
from similarity_checker import QuestionSimilarityChecker
from document_parsers import LegalDocumentParser

# HuggingFace imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ HuggingFace transformers not available. Install with: pip install transformers torch")

class SourceReference(BaseModel):
    """Tham chiếu đến nguồn của thông tin dựa trên units"""
    unit_path: str       # Đường dẫn unit (ví dụ: "Luật Giao thông > Điều 60 > Khoản 1")
    unit_id: str         # ID của unit (ví dụ: "unit_60_1_a")
    document_title: str  # Tên tài liệu (ví dụ: "Luật Giao thông đường bộ 2008")

class LegalQA(BaseModel):
    """Cấu trúc câu hỏi-đáp án pháp lý"""
    question: str
    answer: str

class LegalQAList(BaseModel):
    """Danh sách câu hỏi-đáp án (không cần sources vì đã rule-based)"""
    qa_pairs: List[LegalQA]

class DataGenerator:
    """Class sinh dữ liệu huấn luyện cho LegalSLM - Version gọn gàng"""
    
    def __init__(self, api_key: str = None, similarity_threshold: float = 0.75):
        # Set API key
        if api_key:
            os.environ['GEMINI_API_KEY'] = api_key
        elif not os.environ.get('GEMINI_API_KEY'):
            google_key = os.environ.get('GOOGLE_API_KEY')
            if google_key:
                os.environ['GEMINI_API_KEY'] = google_key
        
        self.client = genai.Client()
        self.model = "gemini-2.5-flash"
        
        # Khởi tạo similarity checker
        self.similarity_checker = QuestionSimilarityChecker(similarity_threshold=similarity_threshold)
        print(f"🔍 Initialized similarity checker with threshold {similarity_threshold}")
        
        # HuggingFace model setup
        self.hf_model = None
        self.hf_tokenizer = None
        
        # Rate limiting for Gemini API (15 req/min = 4 seconds per request)
        self.last_api_call = 0
        self.min_interval = 4.0  # seconds between API calls
    
    def init_huggingface_model(self, model_name: str = "phamhoangf/qwen3-4b-generate-data"):
        """Initialize HuggingFace model"""
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers not available")
        
        try:
            print(f"🤖 Loading HuggingFace model: {model_name}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            print(f"✅ HuggingFace model loaded on {device}")
            
        except Exception as e:
            print(f"❌ Failed to load HuggingFace model: {str(e)}")
            raise
    
    def generate_qa_with_gemini(self, prompt: str, temperature: float = 0.7) -> LegalQAList:
        """Sinh QA bằng Gemini API với rate limiting"""
        # Rate limiting: đảm bảo 15 req/min (4 giây/request)
        current_time = time.time()
        elapsed = current_time - self.last_api_call
        
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            print(f"⏳ Rate limiting: sleeping {sleep_time:.1f}s...")
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                top_p=random.uniform(0.85, 0.95),
                max_output_tokens=3000,
                response_mime_type="application/json",
                response_schema=LegalQAList,
                seed=random.randint(1, 1000000)
            )
        )
        return response.parsed
    
    def generate_qa_with_huggingface(self, prompt: str, temperature: float = 0.7) -> LegalQAList:
        """Sinh QA bằng HuggingFace model trên cloud (Kaggle/ngrok)"""
        import requests
        
        # URL ngrok từ Kaggle notebook
        NGROK_URL = "https://evidently-cheerful-griffon.ngrok-free.app/generate"
        
        # Format messages như trong test_ngrok.py
        messages = [{
            "role": "user",
            "content": prompt
        }]
        
        payload = {"messages": messages}
        
        try:
            print("🌐 Đang gửi yêu cầu đến HuggingFace model trên cloud...")
            response = requests.post(NGROK_URL, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get('response', '')
            
            print(f"📥 Nhận được phản hồi từ cloud model")
            
            # Parse JSON response
            try:
                response_json = json.loads(response_text)
                return LegalQAList(**response_json)
            except json.JSONDecodeError as e:
                print(f"⚠️ Không thể parse JSON từ cloud model: {e}")
                print(f"Raw response: {response_text[:200]}...")
                # Fallback - tạo single QA nếu không parse được
                return LegalQAList(qa_pairs=[LegalQA(question="Sample question", answer="Sample answer")])
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Lỗi khi kết nối đến cloud model: {e}")
            raise ValueError(f"Failed to connect to HuggingFace cloud model: {e}")
        except Exception as e:
            print(f"❌ Lỗi không xác định với cloud model: {e}")
            raise ValueError(f"Error with HuggingFace cloud model: {e}")
    
    def generate_qa(self, prompt: str, llm_type: str = "gemini", temperature: float = 0.7) -> LegalQAList:
        """Sinh QA với LLM được chọn"""
        if llm_type == "gemini":
            return self.generate_qa_with_gemini(prompt, temperature)
        elif llm_type == "huggingface":
            return self.generate_qa_with_huggingface(prompt, temperature)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    
    def get_rule_based_difficulty(self, data_type: str, num_sources: int) -> str:
        """Rule-based difficulty thay vì yêu cầu LLM tạo ra"""
        if data_type == 'word_matching':
            return 'easy'
        elif data_type == 'concept_understanding':
            return 'easy' if num_sources == 1 else 'medium'
        elif data_type == 'multi_paragraph_reading':
            return 'medium' if num_sources <= 3 else 'hard'
        elif data_type == 'multi_hop_reasoning':
            return 'hard'
        else:
            return 'medium'

    def update_similarity_corpus(self, existing_questions_data: List[Dict[str, Any]]):
        """Cập nhật corpus cho similarity checker với dữ liệu hiện có"""
        self.similarity_checker.update_corpus(existing_questions_data)
    
    def filter_duplicate_questions(self, new_samples: List[Dict[str, Any]], verbose: bool = True) -> List[Dict[str, Any]]:
        """Lọc bỏ các câu hỏi trùng lặp từ danh sách samples mới"""
        if not new_samples:
            return []
        
        questions = [sample.get('question', '') for sample in new_samples]
        similarity_results = self.similarity_checker.check_batch_similarity(questions)
        
        filtered_samples = []
        duplicates_found = 0
        
        for i, (sample, result) in enumerate(zip(new_samples, similarity_results)):
            if not result['is_duplicate']:
                filtered_samples.append(sample)
            else:
                duplicates_found += 1
                if verbose:
                    print(f"🚫 Filtered duplicate question {i+1}:")
                    print(f"   Question: {result['question'][:80]}...")
                    print(f"   Max similarity: {result['max_similarity']:.3f}")
        
        if verbose and duplicates_found > 0:
            print(f"🔍 Filtered {duplicates_found}/{len(new_samples)} duplicate questions")
        
        return filtered_samples

    def get_units_from_parsed_structure(self, document) -> List[Dict]:
        """Lấy units từ document sử dụng LegalDocumentParser"""
        try:
            # Sử dụng parser để lấy units trực tiếp
            parser = LegalDocumentParser()
            
            # Nếu có parsed structure thì sử dụng
            if hasattr(document, 'parsed_structure') and document.parsed_structure:
                try:
                    parsed_data = json.loads(document.parsed_structure)
                    print(f"🔍 Using existing parsed structure for {document.title}")
                    print(f"   Articles found: {len(parsed_data.get('articles', []))}")
                    units = parser.get_all_units(parsed_data)
                    print(f"   Units extracted: {len(units)}")
                except Exception as e:
                    print(f"⚠️ Failed to use parsed structure: {str(e)}, re-parsing...")
                    # Fallback: parse lại từ content
                    parsed_data = parser.parse_document(document.title, document.content)
                    print(f"   Re-parsed articles: {len(parsed_data.get('articles', []))}")
                    units = parser.get_all_units(parsed_data)
                    print(f"   Units from re-parse: {len(units)}")
            else:
                # Parse từ đầu nếu chưa có parsed structure
                print(f"🔄 Parsing document from scratch: {document.title}")
                parsed_data = parser.parse_document(document.title, document.content)
                print(f"   Articles parsed: {len(parsed_data.get('articles', []))}")
                units = parser.get_all_units(parsed_data)
                print(f"   Units generated: {len(units)}")
            
            # Convert to format cho data generator
            converted_units = []
            for unit in units:
                converted_units.append({
                    "id": f"unit_{unit['source_article']}_{unit['source_khoan']}_{unit['source_diem']}",
                    "title": unit['path'].split(' > ')[-1] if ' > ' in unit['path'] else f"Unit {unit['source_article']}",
                    "content": unit['content'],
                    "document_title": document.title,
                    "path": unit['path'],
                    "metadata": {
                        "source_article": unit['source_article'],
                        "source_khoan": unit['source_khoan'], 
                        "source_diem": unit['source_diem'],
                        "source_document": document.title,
                        "unit_type": "content_unit",
                        "length": unit.get('content_length', len(unit['content']))
                    }
                })
            
            print(f"✅ Extracted {len(converted_units)} units from {document.title}")
            return converted_units
            
        except Exception as e:
            print(f"❌ Failed to extract units from {document.title}: {str(e)}")
            return []

    def monte_carlo_sample_articles(self, all_articles: List[Dict], sample_size: int, iteration: int = 0) -> List[Dict]:
        """Monte Carlo sampling với độ ngẫu nhiên cao hơn"""
        if not all_articles or sample_size <= 0:
            return []
            
        if sample_size >= len(all_articles):
            articles_copy = all_articles.copy()
            # Thêm iteration seed để mỗi lần gọi khác nhau
            random.seed(hash(f"shuffle_{iteration}_{time.time()}") % 100000)
            random.shuffle(articles_copy)
            random.seed()  # Reset seed
            return articles_copy
        
        # Tạo random seed khác nhau cho mỗi iteration
        random.seed(hash(f"sampling_{iteration}_{time.time()}_{random.randint(1, 10000)}") % 100000)
        
        # Tính weights với nhiều yếu tố ngẫu nhiên hơn
        weights = []
        for i, article in enumerate(all_articles):
            # Base weight từ content length với random factor lớn hơn
            content_length = article.get('metadata', {}).get('length', len(article.get('content', '')))
            length_weight = min(content_length / 800, 2.0)
            
            # Position weight với thêm random
            position_weight = random.uniform(0.8, 1.5)  # Hoàn toàn random thay vì dựa vào article number
            
            # Iteration-based randomness để đảm bảo mỗi lần khác nhau
            iteration_factor = random.uniform(0.5, 2.0) * (1 + 0.1 * (iteration % 10))
            
            # Index-based diversity để tránh bias vị trí
            index_factor = random.uniform(0.7, 1.3) * (1 + 0.05 * (i % 7))
            
            # Time-based randomness
            time_factor = random.uniform(0.8, 1.2) * (1 + 0.001 * (int(time.time()) % 1000))
            
            # Kết hợp tất cả factors
            final_weight = length_weight * position_weight * iteration_factor * index_factor * time_factor
            weights.append(max(final_weight, 0.1))
        
        # Thêm noise vào weights để tăng entropy
        for i in range(len(weights)):
            noise = random.uniform(0.9, 1.1)
            weights[i] *= noise
        
        # Monte Carlo sampling với multiple rounds để tăng randomness
        selected = []
        available_indices = list(range(len(all_articles)))
        available_weights = weights.copy()
        
        for round_idx in range(sample_size):
            if not available_indices:
                break
            
            # Thêm round-based randomness
            round_factor = random.uniform(0.9, 1.1)
            adjusted_weights = [w * round_factor * random.uniform(0.95, 1.05) for w in available_weights]
            
            total_weight = sum(adjusted_weights)
            if total_weight == 0:
                chosen_idx = random.randint(0, len(available_indices) - 1)
            else:
                # Multiple random attempts để tăng entropy
                best_choice = None
                for attempt in range(3):  # Thử 3 lần, chọn lần cuối
                    rand_val = random.uniform(0, total_weight)
                    cumsum = 0
                    for i, weight in enumerate(adjusted_weights):
                        cumsum += weight
                        if rand_val <= cumsum:
                            best_choice = i
                            break
                    if best_choice is None:
                        best_choice = len(adjusted_weights) - 1
                
                chosen_idx = best_choice if best_choice is not None else random.randint(0, len(available_indices) - 1)
            
            # Add selected article
            selected.append(all_articles[available_indices[chosen_idx]])
            
            # Remove from available
            available_indices.pop(chosen_idx)
            available_weights.pop(chosen_idx)
        
        # Final shuffle với iteration seed
        random.seed(hash(f"final_{iteration}_{len(selected)}_{time.time()}") % 100000)
        random.shuffle(selected)
        random.seed()  # Reset seed
        
        print(f"🎲 Monte Carlo sampling (iteration {iteration}): chọn {len(selected)}/{len(all_articles)} articles với high entropy")
        return selected

    def generate_from_multiple_documents(self, documents, topic_name, data_type, num_samples, llm_type="gemini"):
        """Sinh dữ liệu từ nhiều documents - main method"""
        if not documents:
            return []

        print(f"🔍 Phân tích {len(documents)} documents...")

        # Lấy units từ parsed structure
        all_units = []
        for doc in documents:
            units = self.get_units_from_parsed_structure(doc)
            all_units.extend(units)
            print(f"  📋 {doc.title}: {len(units)} units")

        print(f"📊 Tổng cộng: {len(all_units)} units từ {len(documents)} tài liệu")

        # Monte Carlo sampling
        max_units = min(len(all_units), max(num_samples // 2, 10))
        selected_units = self.monte_carlo_sample_articles(all_units, max_units)
        print(f"  🎯 Đã chọn {len(selected_units)} units")

        # Sinh dữ liệu
        all_samples = self.generate_samples_from_units(selected_units, topic_name, data_type, num_samples, llm_type)

        # Lọc trùng lặp
        print(f"🔍 Kiểm tra tương đồng cho {len(all_samples)} samples...")
        filtered_samples = self.filter_duplicate_questions(all_samples, verbose=True)

        print(f"✅ Hoàn thành: {len(filtered_samples)} samples (đã lọc {len(all_samples) - len(filtered_samples)} trùng lặp)")
        return filtered_samples[:num_samples]

    def generate_samples_from_units(self, units, topic, data_type, num_samples, llm_type="gemini"):
        """Sinh dữ liệu đơn giản với sources chung cho tất cả câu hỏi"""
        if not units:
            return []
        
        # Xác định số sources theo yêu cầu
        num_sources_map = {
            'word_matching': min(1, len(units)),
            'concept_understanding': min(1, len(units)),
            'multi_paragraph_reading': min(2, len(units)),
            'multi_hop_reasoning': min(3, len(units))
        }
        num_sources = num_sources_map.get(data_type, min(3, len(units)))
        
        # Tạo câu hỏi - mỗi iteration tự Monte Carlo chọn units
        all_samples = []
        
        # Rate limiting info
        if llm_type == "gemini":
            estimated_time = num_samples * self.min_interval / 60  # minutes
            print(f"⏳ Estimated time for {num_samples} samples with Gemini: {estimated_time:.1f} minutes")
        
        for i in range(num_samples):
            print(f"🔄 Generating sample {i+1}/{num_samples}...")
            
            # Monte Carlo sampling cho iteration này
            selected_units = self.monte_carlo_sample_articles(units, num_sources, iteration=i)
            
            # Tạo sources và content cho iteration này
            iteration_sources = []
            combined_content = []
            
            for unit in selected_units:
                source_ref = SourceReference(
                    unit_path=unit.get('path', unit['title']),
                    unit_id=unit.get('id', 'unknown_id'),
                    document_title=unit['document_title']
                )
                iteration_sources.append(source_ref)
                unit_path = unit.get('path', unit['title'])
                combined_content.append(f"--- {unit['title']} ({unit_path}) ---\n{unit['content']}")

            combined_text = "\n\n".join(combined_content)
            
            # Rule-based difficulty cho iteration này
            difficulty = self.get_rule_based_difficulty(data_type, len(selected_units))
            
            # Tạo prompt cho iteration này
            prompt = self.create_diverse_prompt(combined_text, topic, data_type, difficulty, i)
            
            try:
                temperature = random.uniform(0.6, 0.9)
                
                # Sử dụng LLM được chọn
                structured_data = self.generate_qa(prompt, llm_type, temperature)
                
                # Convert với sources chung và rule-based difficulty
                for qa_pair in structured_data.qa_pairs:
                    sample = {
                        'question': qa_pair.question,
                        'answer': qa_pair.answer,
                        'difficulty': difficulty,  
                        'sources': [
                            {
                                'unit_path': src.unit_path,
                                'unit_id': src.unit_id,
                                'document_title': src.document_title
                            } for src in iteration_sources  # Sources cho iteration này
                        ],
                        'metadata': {
                            'generation_method': 'per_iteration_monte_carlo',
                            'num_sources': len(selected_units),
                            'temperature': temperature,
                            'llm_type': llm_type
                        }
                    }
                    all_samples.append(sample)
                    
                print(f"✅ Sample {i+1}/{num_samples} completed")
                    
            except Exception as e:
                print(f"❌ Generation failed for sample {i+1}/{num_samples}: {e}")
                continue
        
        return all_samples

    def create_diverse_prompt(self, content, topic, data_type, difficulty, iteration):
        """Hàm gốc tạo prompt đa dạng - sử dụng làm base cho các loại câu hỏi"""
        # Cấu trúc câu hỏi đa dạng
        question_starters = [
            "Khi nào", "Trong trường hợp nào", "Ai có trách nhiệm",
            "Việc...được thực hiện như thế nào", "Điều kiện...là gì",
            "Mức phạt...là bao nhiêu", "Quy trình...diễn ra ra sao",
            "Tại sao", "Vì sao", "Làm cách nào", "Bằng phương thức nào",
            "Có được phép", "Có bắt buộc", "Có cần thiết",
            "Thủ tục...như thế nào", "Hình thức...là gì", "Phạm vi...ra sao"
        ]
        
        focus_areas = [
            "quy định thực tế và ứng dụng cụ thể",
            "trường hợp ngoại lệ và điều kiện đặc biệt", 
            "nghĩa vụ và quyền hạn của các đối tượng",
            "mức phạt và hậu quả pháp lý",
            "quy trình thủ tục pháp lý chi tiết",
            "định nghĩa thuật ngữ chuyên môn",
            "thẩm quyền và trách nhiệm quản lý"
        ]
        
        # Random selection với seed từ iteration để tạo diversity
        random.seed(hash(f"{data_type}_{iteration}_{topic}") % 10000)
        starter = random.choice(question_starters)
        focus = random.choice(focus_areas)
        
        # Reset seed
        random.seed()
        
        # Gọi hàm con tương ứng với data_type
        if data_type == "word_matching":
            return self.create_word_matching_prompt(content, topic, starter, focus, difficulty)
        elif data_type == "concept_understanding":
            return self.create_concept_understanding_prompt(content, topic, starter, focus, difficulty)
        elif data_type == "multi_paragraph_reading":
            return self.create_multi_paragraph_prompt(content, topic, starter, focus, difficulty)
        elif data_type == "multihop":
            return self.create_multihop_prompt(content, topic, starter, focus, difficulty)
        else:
            return self.create_concept_understanding_prompt(content, topic, starter, focus, difficulty)

    def create_word_matching_prompt(self, content, topic, starter, focus, difficulty):
        """Prompt cho loại Word Matching - tìm từ khóa, thuật ngữ cụ thể trong văn bản"""
        return f"""
    Dưới đây là các điều luật về chủ đề "{topic}":

    {content}

    Hãy tạo 1 câu hỏi loại WORD MATCHING (độ khó {difficulty}) tập trung vào {focus}.

    ĐẶC ĐIỂM CÂU HỎI WORD MATCHING:
    - Yêu cầu tìm từ khóa, thuật ngữ cụ thể 
    - Hỏi về định nghĩa chính xác của các khái niệm pháp lý
    - Câu trả lời là thông tin cụ thể, rõ ràng
    - Tập trung vào thuật ngữ chuyên môn, số liệu cụ thể

    YÊU CẦU QUAN TRỌNG:
    1. Câu hỏi và đáp án phải HOÀN TOÀN ĐỘC LẬP - có thể hiểu được mà không cần context bên ngoài
    2. TUYỆT ĐỐI KHÔNG dùng "dựa trên điều luật trên", "theo quy định trên", "căn cứ vào điều trên"
    3. Không cần thiết phải trích dẫn, NẾU cần trích dẫn: phải ghi ĐẦY ĐỦ tên văn bản (ví dụ: "Theo Luật Giao thông đường bộ 2008, Điều 25") hoặc nội dung phần văn bản cần trích dẫn
    4. Bạn có thể tham khảo bắt đầu câu hỏi bằng "{starter}..." 
    5. Đáp án phải DIỄN GIẢI ĐẦY ĐỦ, không cụt lủn như "11%" hay "Thống đốc"

    VÍ DỤ TỐT:
    Question: "Độ tuổi tối thiểu để được cấp bằng lái xe ô tô là bao nhiêu?"
    Answer: "Độ tuổi tối thiểu để được cấp bằng lái xe ô tô là 18 tuổi đối với xe ô tô con và 21 tuổi đối với xe tải, xe khách theo quy định của Luật Giao thông đường bộ."

    VÍ DỤ XẤU (TRÁNH):
    Answer: "Dựa trên điều luật trên, độ tuổi là 18 tuổi."

    Trả về output dưới dạng JSON với qa_pairs.
    """

    def create_concept_understanding_prompt(self, content, topic, starter, focus, difficulty):
        """Prompt cho loại Concept Understanding - hiểu khái niệm, nguyên tắc"""
        return f"""
    Dưới đây là các điều luật về chủ đề "{topic}":

    {content}

    Hãy tạo 1 câu hỏi loại CONCEPT UNDERSTANDING (độ khó {difficulty}) tập trung vào {focus}.

    ĐẶC ĐIỂM CÂU HỎI CONCEPT UNDERSTANDING:
    - Kiểm tra hiểu biết về khái niệm, nguyên tắc pháp lý
    - Yêu cầu giải thích ý nghĩa, mục đích của quy định
    - Câu trả lời cần diễn giải, giải thích rõ ràng
    - Tập trung vào việc hiểu "tại sao" và "như thế nào"

    YÊU CẦU QUAN TRỌNG:
    1. Câu hỏi và đáp án phải HOÀN TOÀN ĐỘC LẬP - có thể hiểu được mà không cần context bên ngoài
    2. TUYỆT ĐỐI KHÔNG dùng "dựa trên điều luật trên", "theo quy định trên", "căn cứ vào điều trên"
    3. Không cần thiết phải trích dẫn, NẾU cần trích dẫn: phải ghi ĐẦY ĐỦ tên văn bản (ví dụ: "Theo Luật Giao thông đường bộ 2008, Điều 25") hoặc nội dung phần văn bản cần trích dẫn
    4. Bạn có thể tham khảo bắt đầu câu hỏi bằng "{starter}..."
    5. Đáp án cần giải thích khái niệm đầy đủ, không cụt lủn như "Là việc kiểm tra định kỳ" hay "Để đảm bảo an toàn" nhưng cũng không diễn giải quá lê thê

    VÍ DỤ TỐT:
    Question: "Tại sao việc kiểm định định kỳ phương tiện giao thông là bắt buộc?"
    Answer: "Việc kiểm định định kỳ phương tiện giao thông là bắt buộc nhằm đảm bảo an toàn giao thông, kiểm tra tình trạng kỹ thuật của xe, phát hiện sớm các hư hỏng có thể gây tai nạn, đồng thời kiểm soát khí thải bảo vệ môi trường."

    Trả về output dưới dạng JSON với qa_pairs.
    ĐỊNH DẠNG OUTPUT BẮT BUỘC:
    Trả về duy nhất một khối mã JSON hợp lệ. Không thêm bất kỳ lời giải thích hay văn bản nào khác bên ngoài khối JSON.

    ```json
    {{
    "qa_pairs": [
        {{
        "question": "Nội dung câu hỏi được tạo ra ở đây, bắt đầu bằng '{starter}' và tập trung vào chủ đề '{focus}'.",
        "answer": "Nội dung câu trả lời chi tiết, giải thích rõ ràng khái niệm, nguyên tắc và có thể kèm theo ví dụ cụ thể để minh họa."
        }}
    ]
    }}
        """

    def create_multi_paragraph_prompt(self, content, topic, starter, focus, difficulty):
        """Prompt cho loại Multi-paragraph Reading - đọc hiểu nhiều đoạn văn"""
        return f"""
    Dưới đây là các điều luật về chủ đề "{topic}":

    {content}

    Hãy tạo 1 câu hỏi loại MULTI-PARAGRAPH READING (độ khó {difficulty}) tập trung vào {focus}.
    
    ĐẶC ĐIỂM CÂU HỎI MULTI-PARAGRAPH READING:
    - Yêu cầu tổng hợp thông tin từ nhiều quy định khác nhau
    - So sánh, đối chiếu các điều khoản
    - Tìm mối liên hệ giữa các quy định
    - Câu trả lời cần kết hợp thông tin từ nhiều nguồn
    
    YÊU CẦU QUAN TRỌNG:
    1. CÂU HỎI PHẢI CÓ ĐỘ BAO PHỦ RỘNG: Cố gắng thiết kế câu hỏi sao cho người trả lời BẮT BUỘC phải đọc và tổng hợp thông tin từ TẤT CẢ các đoạn trích đã cho để có thể trả lời đầy đủ.
    2. CÂU HỎI NÊN BẮT ĐẦU TỪ MỘT TÌNH HUỐNG THỰC TẾ: Hãy tưởng tượng một kịch bản cụ thể mà một chủ thể có thể gặp phải, sau đó đặt câu hỏi pháp lý liên quan đến kịch bản đó.
    3. Câu hỏi và đáp án phải HOÀN TOÀN ĐỘC LẬP - có thể hiểu được mà không cần context bên ngoài
    4. TUYỆT ĐỐI KHÔNG dùng "dựa trên điều luật trên", "theo các quy định trên", "căn cứ vào điều trên"
    5. NẾU cần trích dẫn: phải ghi ĐẦY ĐỦ tên văn bản cụ thể cho từng điều
    6. Bạn có thể tham khảo bắt đầu câu hỏi bằng "{starter}..."
    7. Đáp án phải tổng hợp, so sánh rõ ràng các quy định khác nhau
    
    VÍ DỤ TỐT:
    Question: "Nếu một người vừa muốn đăng ký kinh doanh hộ cá thể, vừa muốn mở một doanh nghiệp tư nhân khác, quy định pháp luật về các trường hợp này có điểm gì giống và khác nhau về quyền và nghĩa vụ?"
    Answer: "Điểm giống nhau là cả hai đều do một cá nhân làm chủ và chịu trách nhiệm vô hạn. Tuy nhiên, điểm khác biệt lớn là theo quy định, một cá nhân chỉ được thành lập một doanh nghiệp tư nhân, nhưng có thể đồng thời là chủ hộ kinh doanh. Do đó, người này có thể tiếp tục đăng ký hộ kinh doanh nhưng không thể mở thêm doanh nghiệp tư nhân thứ hai."
    
    Trả về output dưới dạng JSON với qa_pairs.
        """

    def create_multihop_prompt(self, content, topic, starter, focus, difficulty):
        """Prompt cho loại Multihop - suy luận qua nhiều bước"""
        return f"""
    Dưới đây là các điều luật về chủ đề "{topic}":

    {content}

    Hãy tạo 1 câu hỏi loại MULTIHOP (độ khó {difficulty}) tập trung vào {focus}.
    
    ĐẶC ĐIỂM CÂU HỎI MULTIHOP:
    - Yêu cầu suy luận logic qua nhiều bước
    - Kết hợp nhiều quy định để đưa ra kết luận
    - Áp dụng quy tắc vào tình huống phức tạp, thực tế
    - Câu trả lời cần chuỗi suy luận có logic rõ ràng
    
    YÊU CẦU QUAN TRỌNG:
    1. CÂU HỎI PHẢI CÓ ĐỘ BAO PHỦ RỘNG: Cố gắng thiết kế câu hỏi sao cho người trả lời BẮT BUỘC phải đọc và tổng hợp thông tin từ TẤT CẢ các đoạn trích đã cho để có thể trả lời đầy đủ.
    2. CÂU HỎI NÊN BẮT ĐẦU TỪ MỘT TÌNH HUỐNG THỰC TẾ: Hãy tưởng tượng một kịch bản cụ thể mà một chủ thể có thể gặp phải, sau đó đặt câu hỏi pháp lý liên quan đến kịch bản đó.                                                        
    3. Câu hỏi và đáp án phải HOÀN TOÀN ĐỘC LẬP - có thể hiểu được mà không cần context bên ngoài
    4. TUYỆT ĐỐI KHÔNG dùng "dựa trên điều luật trên", "theo các quy định trên", "căn cứ vào điều trên"
    5. NẾU cần trích dẫn: phải ghi ĐẦY ĐỦ tên văn bản và điều cụ thể
    6. Bạn có thể tham khảo bắt đầu câu hỏi bằng "{starter}..."
    7. Đáp án cần có chuỗi suy luận từng bước: tình huống → quy định áp dụng → kết luận
    
    VÍ DỤ TỐT:
    Question: "Một tài xế lái xe tải chở hàng quá tải 50% và không có bằng lái phù hợp sẽ bị xử lý như thế nào?"
    Answer: "Tài xế này sẽ bị xử phạt kép: đầu tiên bị phạt tiền 12-15 triệu đồng và tước bằng lái 2-4 tháng do chở quá tải theo Nghị định 100/2019, đồng thời bị phạt 16-18 triệu và tước bằng lái 10-12 tháng do không có bằng lái phù hợp. Tổng cộng có thể bị phạt đến 33 triệu đồng và tước bằng lái tối đa 16 tháng."
    
    VÍ DỤ XẤU (TRÁNH):
    Answer: "Căn cứ vào các điều luật trên, tài xế sẽ bị xử phạt..."
    
    Trả về output dưới dạng JSON với qa_pairs.
        """