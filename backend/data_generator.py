from google import genai
from google.genai import types
import json
import random
import os
import re
import time  # Add time import for rate limiting
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
    """Tham chiếu đến nguồn của thông tin"""
    article_number: str  # Số điều (ví dụ: "60", "61")
    article_title: str   # Tiêu đề điều (ví dụ: "Điều 60. Độ tuổi của người lái xe")
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
        """Sinh QA bằng HuggingFace model"""
        if not self.hf_model:
            raise ValueError("HuggingFace model not initialized. Call init_huggingface_model() first")
        
        # Format prompt cho model
        formatted_prompt = f"<|system|>Bạn là trợ lý AI chuyên về pháp luật Việt Nam. Hãy tạo câu hỏi và câu trả lời từ văn bản pháp luật.<|end|>\n<|user|>{prompt}<|end|>\n<|assistant|>"
        
        inputs = self.hf_tokenizer.encode(formatted_prompt, return_tensors="pt")
        if self.hf_model.device.type == "cuda":
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.hf_model.generate(
                inputs,
                max_new_tokens=2048,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.hf_tokenizer.eos_token_id
            )
        
        response_text = self.hf_tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Parse JSON response
        try:
            response_json = json.loads(response_text)
            return LegalQAList(**response_json)
        except json.JSONDecodeError:
            # Fallback - tạo single QA nếu không parse được
            return LegalQAList(qa_pairs=[LegalQA(question="Sample question", answer="Sample answer")])
    
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

    def get_articles_from_parsed_structure(self, document) -> List[Dict]:
        """Lấy articles từ parsed structure hoặc fallback"""
        # Kiểm tra parsed structure
        if hasattr(document, 'parsed_structure') and document.parsed_structure:
            try:
                parsed_data = json.loads(document.parsed_structure)
                parser = LegalDocumentParser()
                articles = parser.get_all_articles(parsed_data)
                
                # Convert to format cho data generator
                units = []
                for article in articles:
                    units.append({
                        "id": f"article_{article['number']}",
                        "title": f"Điều {article['number']}. {article['title']}",
                        "content": article['content'],
                        "document_title": document.title,
                        "metadata": {
                            "article_number": article['number'],
                            "source_document": document.title,
                            "unit_type": "article",
                            "length": article['content_length']
                        }
                    })
                
                print(f"✅ Using parsed structure: {len(units)} articles from {document.title}")
                return units
                
            except Exception as e:
                print(f"⚠️ Failed to use parsed structure: {str(e)}, fallback to simple parsing")
        
        # Fallback - simple article extraction
        return self.split_law_by_article(document.content, document.title)

    def split_law_by_article(self, text: str, document_title: str = "") -> List[Dict]:
        """Tách văn bản luật thành các điều"""
        units = []
        split_pattern = r'(?m)(?=^\s*Điều \d+\.)'
        chunks = re.split(split_pattern, text.strip())
        
        for chunk in chunks:
            chunk = chunk.strip()
            lines = chunk.split('\n')
            dieu_line = None
            for line in lines:
                if re.match(r'^\s*Điều \d+\.', line):
                    dieu_line = line.strip()
                    break
            
            if dieu_line:
                match = re.search(r'Điều (\d+)', dieu_line)
                if match:
                    article_number = match.group(1)
                    units.append({
                        "id": f"article_{article_number}",
                        "title": dieu_line,
                        "content": chunk,
                        "document_title": document_title,
                        "metadata": {
                            "article_number": article_number,
                            "source_document": document_title,
                            "unit_type": "article",
                            "length": len(chunk)
                        }
                    })
                
        return units

    def monte_carlo_sample_articles(self, all_articles: List[Dict], sample_size: int) -> List[Dict]:
        """Monte Carlo sampling đơn giản với weights dựa trên content length và position"""
        if not all_articles or sample_size <= 0:
            return []
            
        if sample_size >= len(all_articles):
            articles_copy = all_articles.copy()
            random.shuffle(articles_copy)
            return articles_copy
        
        # Tính weights đơn giản
        weights = []
        for article in all_articles:
            # Base weight từ content length
            content_length = article.get('metadata', {}).get('length', len(article.get('content', '')))
            length_weight = min(content_length / 800, 2.0)
            
            # Position weight (strategic articles)
            article_num = article.get('metadata', {}).get('article_number')
            position_weight = 1.0
            if article_num:
                try:
                    num = int(article_num)
                    if num <= 5 or num % 20 == 0:
                        position_weight = 1.5
                    elif num <= 20:
                        position_weight = 1.2
                except:
                    pass
            
            # Random factor cho diversity
            random_factor = random.uniform(0.7, 1.3)
            final_weight = length_weight * position_weight * random_factor
            weights.append(max(final_weight, 0.1))
        
        # Monte Carlo sampling
        selected = []
        available_indices = list(range(len(all_articles)))
        available_weights = weights.copy()
        
        for _ in range(sample_size):
            if not available_indices:
                break
            
            total_weight = sum(available_weights)
            if total_weight == 0:
                chosen_idx = random.randint(0, len(available_indices) - 1)
            else:
                rand_val = random.uniform(0, total_weight)
                cumsum = 0
                chosen_idx = len(available_weights) - 1
                
                for i, weight in enumerate(available_weights):
                    cumsum += weight
                    if rand_val <= cumsum:
                        chosen_idx = i
                        break
            
            # Add selected article
            selected.append(all_articles[available_indices[chosen_idx]])
            
            # Remove from available
            available_indices.pop(chosen_idx)
            available_weights.pop(chosen_idx)
        
        random.shuffle(selected)
        print(f"🎲 Monte Carlo sampling: chọn {len(selected)}/{len(all_articles)} articles")
        return selected

    def generate_from_multiple_documents(self, documents, topic_name, data_type, num_samples, llm_type="gemini"):
        """Sinh dữ liệu từ nhiều documents - main method"""
        if not documents:
            return []

        print(f"🔍 Phân tích {len(documents)} documents...")

        # Lấy articles từ parsed structure
        all_articles = []
        for doc in documents:
            articles = self.get_articles_from_parsed_structure(doc)
            all_articles.extend(articles)
            print(f"  📋 {doc.title}: {len(articles)} điều")

        print(f"📊 Tổng cộng: {len(all_articles)} điều từ {len(documents)} tài liệu")

        # Monte Carlo sampling
        max_articles = min(len(all_articles), max(num_samples // 2, 10))
        selected_articles = self.monte_carlo_sample_articles(all_articles, max_articles)
        print(f"  🎯 Đã chọn {len(selected_articles)} articles")

        # Sinh dữ liệu
        all_samples = self.generate_samples_from_articles(selected_articles, topic_name, data_type, num_samples, llm_type)

        # Lọc trùng lặp
        print(f"🔍 Kiểm tra tương đồng cho {len(all_samples)} samples...")
        filtered_samples = self.filter_duplicate_questions(all_samples, verbose=True)

        print(f"✅ Hoàn thành: {len(filtered_samples)} samples (đã lọc {len(all_samples) - len(filtered_samples)} trùng lặp)")
        return filtered_samples[:num_samples]

    def generate_samples_from_articles(self, articles, topic, data_type, num_samples, llm_type="gemini"):
        """Sinh dữ liệu đơn giản với sources chung cho tất cả câu hỏi"""
        if not articles:
            return []
        
        # Xác định số sources theo yêu cầu
        num_sources_map = {
            'word_matching': min(1, len(articles)),
            'concept_understanding': min(1, len(articles)),
            'multi_paragraph_reading': min(2, len(articles)),
            'multi_hop_reasoning': min(3, len(articles))
        }
        num_sources = num_sources_map.get(data_type, min(3, len(articles)))
        
        # Tạo câu hỏi - mỗi iteration tự Monte Carlo chọn articles
        all_samples = []
        
        # Rate limiting info
        if llm_type == "gemini":
            estimated_time = num_samples * self.min_interval / 60  # minutes
            print(f"⏳ Estimated time for {num_samples} samples with Gemini: {estimated_time:.1f} minutes")
        
        for i in range(num_samples):
            print(f"🔄 Generating sample {i+1}/{num_samples}...")
            
            # Monte Carlo sampling cho iteration này
            selected_articles = self.monte_carlo_sample_articles(articles, num_sources)
            
            # Tạo sources và content cho iteration này
            iteration_sources = []
            combined_content = []
            
            for article in selected_articles:
                source_ref = SourceReference(
                    article_number=str(article['metadata']['article_number']) if article['metadata']['article_number'] else "unknown",
                    article_title=article['title'],
                    document_title=article['document_title']
                )
                iteration_sources.append(source_ref)
                article_path = article.get('path', article['title'])
                combined_content.append(f"--- {article['title']} ({article_path}) ---\n{article['content']}")

            combined_text = "\n\n".join(combined_content)
            
            # Rule-based difficulty cho iteration này
            difficulty = self.get_rule_based_difficulty(data_type, len(selected_articles))
            
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
                                'article_number': src.article_number,
                                'article_title': src.article_title,
                                'document_title': src.document_title
                            } for src in iteration_sources  # Sources cho iteration này
                        ],
                        'metadata': {
                            'generation_method': 'per_iteration_monte_carlo',
                            'num_sources': len(selected_articles),
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
    5. Đáp án cần giải thích khái niệm đầy đủ, có thể bao gồm ví dụ minh họa

    VÍ DỤ TỐT:
    Question: "Tại sao việc kiểm định định kỳ phương tiện giao thông là bắt buộc?"
    Answer: "Việc kiểm định định kỳ phương tiện giao thông là bắt buộc nhằm đảm bảo an toàn giao thông, kiểm tra tình trạng kỹ thuật của xe, phát hiện sớm các hư hỏng có thể gây tai nạn, đồng thời kiểm soát khí thải bảo vệ môi trường."

    Trả về output dưới dạng JSON với qa_pairs.
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
    1. Câu hỏi và đáp án phải HOÀN TOÀN ĐỘC LẬP - có thể hiểu được mà không cần context bên ngoài
    2. TUYỆT ĐỐI KHÔNG dùng "dựa trên điều luật trên", "theo các quy định trên", "căn cứ vào điều trên"
    3. Không cần thiết phải trích dẫn, NẾU cần trích dẫn: phải ghi ĐẦY ĐỦ tên văn bản (ví dụ: "Theo Luật Giao thông đường bộ 2008, Điều 25") hoặc nội dung phần văn bản cần trích dẫn
    4. Bạn có thể tham khảo bắt đầu câu hỏi bằng "{starter}..."
    5. Đáp án phải tổng hợp, so sánh rõ ràng các quy định khác nhau

    VÍ DỤ TỐT:
    Question: "Có những hình thức xử phạt nào đối với vi phạm giao thông?"
    Answer: "Có 4 hình thức xử phạt chính: phạt cảnh cáo đối với vi phạm nhẹ lần đầu, phạt tiền từ 100.000 đến 40 triệu đồng tùy mức độ vi phạm, tước quyền sử dụng bằng lái xe từ 1-24 tháng đối với vi phạm nghiêm trọng, và tịch thu phương tiện đối với các trường hợp vi phạm đặc biệt nghiêm trọng."

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
    - Kết hợp thông tin từ nhiều quy định để đưa ra kết luận
    - Áp dụng quy tắc vào tình huống phức tạp, thực tế
    - Câu trả lời cần chuỗi suy luận có logic rõ ràng

    YÊU CẦU QUAN TRỌNG:
    1. Câu hỏi và đáp án phải HOÀN TOÀN ĐỘC LẬP - có thể hiểu được mà không cần context bên ngoài
    2. TUYỆT ĐỐI KHÔNG dùng "dựa trên điều luật trên", "theo các quy định trên", "căn cứ vào điều trên"
    3. Không cần thiết phải trích dẫn, NẾU cần trích dẫn: phải ghi ĐẦY ĐỦ tên văn bản (ví dụ: "Theo Luật Giao thông đường bộ 2008, Điều 25") hoặc nội dung phần văn bản cần trích dẫn
    4. Bạn có thể tham khảo bắt đầu câu hỏi bằng "{starter}..."
    5. Đáp án cần có chuỗi suy luận từng bước: tình huống → quy định áp dụng → kết luận

    VÍ DỤ TỐT:
    Question: "Một tài xế lái xe tải chở hàng quá tải 50% và không có bằng lái phù hợp sẽ bị xử lý như thế nào?"
    Answer: "Tài xế này sẽ bị xử phạt kép: đầu tiên bị phạt tiền 12-15 triệu đồng và tước bằng lái 2-4 tháng do chở quá tải theo Nghị định 100/2019, đồng thời bị phạt 16-18 triệu và tước bằng lái 10-12 tháng do không có bằng lái phù hợp. Tổng cộng có thể bị phạt đến 33 triệu đồng và tước bằng lái tối đa 16 tháng."

    VÍ DỤ XẤU (TRÁNH):
    Answer: "Căn cứ vào các điều luật trên, tài xế sẽ bị xử phạt..."

    Trả về output dưới dạng JSON với qa_pairs.
        """