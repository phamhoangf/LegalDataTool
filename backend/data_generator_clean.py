from google import genai
from google.genai import types
import json
import random
import os
import re
from typing import List, Dict, Any
from pydantic import BaseModel
from similarity_checker import QuestionSimilarityChecker
from legal_parser import LegalDocumentParser

class SourceReference(BaseModel):
    """Tham chiếu đến nguồn của thông tin"""
    article_number: str  # Số điều (ví dụ: "60", "61")
    article_title: str   # Tiêu đề điều (ví dụ: "Điều 60. Độ tuổi của người lái xe")
    document_title: str  # Tên tài liệu (ví dụ: "Luật Giao thông đường bộ 2008")

class LegalQA(BaseModel):
    """Cấu trúc câu hỏi-đáp án pháp lý"""
    question: str
    answer: str

class LegalQAResponse(BaseModel):
    """Response chứa danh sách QA và sources chung"""
    qa_pairs: List[LegalQA]
    sources: List[SourceReference]  # Sources chung cho tất cả QA

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
        self.model = "gemini-2.0-flash-exp"
        
        # Khởi tạo similarity checker
        self.similarity_checker = QuestionSimilarityChecker(similarity_threshold=similarity_threshold)
        print(f"🔍 Initialized similarity checker with threshold {similarity_threshold}")
    
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

    def generate_from_multiple_documents(self, documents, topic_name, data_type, num_samples):
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
        max_articles = min(len(all_articles), max(num_samples // 2, 5))
        selected_articles = self.monte_carlo_sample_articles(all_articles, max_articles)
        print(f"  🎯 Đã chọn {len(selected_articles)} articles")

        # Sinh dữ liệu
        all_samples = self.generate_samples_from_articles(selected_articles, topic_name, data_type, num_samples)

        # Lọc trùng lặp
        print(f"🔍 Kiểm tra tương đồng cho {len(all_samples)} samples...")
        filtered_samples = self.filter_duplicate_questions(all_samples, verbose=True)

        print(f"✅ Hoàn thành: {len(filtered_samples)} samples (đã lọc {len(all_samples) - len(filtered_samples)} trùng lặp)")
        return filtered_samples[:num_samples]

    def generate_samples_from_articles(self, articles, topic, data_type, num_samples):
        """Sinh dữ liệu đơn giản với sources chung cho tất cả câu hỏi"""
        if not articles:
            return []
        
        # Xác định số sources
        num_sources_map = {
            'word_matching': min(3, len(articles)),
            'concept_understanding': min(4, len(articles)), 
            'multi_paragraph_reading': min(5, len(articles)),
            'multi_hop_reasoning': min(6, len(articles))
        }
        num_sources = num_sources_map.get(data_type, min(3, len(articles)))
        
        # Chọn articles cho sources chung
        if len(articles) <= num_sources:
            selected_articles = articles.copy()
            random.shuffle(selected_articles)
        else:
            selected_articles = random.sample(articles, num_sources)
        
        # Tạo sources chung
        common_sources = []
        combined_content = []
        
        for article in selected_articles:
            source_ref = SourceReference(
                article_number=str(article['metadata']['article_number']) if article['metadata']['article_number'] else "unknown",
                article_title=article['title'],
                document_title=article['document_title']
            )
            common_sources.append(source_ref)
            combined_content.append(f"--- {article['title']} ---\n{article['content']}")

        combined_text = "\n\n".join(combined_content)
        
        # Rule-based difficulty
        difficulty = self.get_rule_based_difficulty(data_type, len(selected_articles))
        
        # Tạo câu hỏi
        all_samples = []
        for i in range(num_samples):
            prompt = self.create_diverse_prompt(combined_text, topic, data_type, difficulty, i)
            
            try:
                temperature = random.uniform(0.6, 0.9)
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        top_p=random.uniform(0.85, 0.95),
                        max_output_tokens=3000,
                        response_mime_type="application/json",
                        response_schema=LegalQAResponse,
                        seed=random.randint(1, 1000000)
                    )
                )
                
                structured_data: LegalQAResponse = response.parsed
                
                # Convert với sources chung và rule-based difficulty
                for qa_pair in structured_data.qa_pairs:
                    sample = {
                        'question': qa_pair.question,
                        'answer': qa_pair.answer,
                        'difficulty': difficulty,  # Rule-based, không từ LLM
                        'sources': [
                            {
                                'article_number': src.article_number,
                                'article_title': src.article_title,
                                'document_title': src.document_title
                            } for src in common_sources  # Sources chung
                        ],
                        'metadata': {
                            'generation_method': 'clean_multi_source',
                            'num_sources': len(selected_articles),
                            'temperature': temperature
                        }
                    }
                    all_samples.append(sample)
                    
            except Exception as e:
                print(f"❌ Generation failed for sample {i+1}: {e}")
                continue
        
        return all_samples

    def create_diverse_prompt(self, content, topic, data_type, difficulty, iteration):
        """Tạo prompt đa dạng để tránh trùng lặp"""
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
        entropy_id = random.randint(1000, 9999)
        
        # Reset seed
        random.seed()
        
        return f"""
Dưới đây là các điều luật về chủ đề "{topic}":

{content}

Hãy tạo 1 câu hỏi độ khó {difficulty} tập trung vào {focus}.

YÊU CẦU QUAN TRỌNG:
1. TUYỆT ĐỐI KHÔNG dùng cấu trúc "Theo Điều X của Luật..."
2. Bắt đầu câu hỏi bằng "{starter}..." hoặc cấu trúc tương tự
3. Câu hỏi phải độc lập, không nhắc đến tên điều luật cụ thể
4. Tập trung vào {focus}
5. Entropy ID: {entropy_id} (để đảm bảo uniqueness)

VÍ DỤ CẤU TRÚC TỐT:
- "Khi nào doanh nghiệp cần có giấy phép kinh doanh vận tải?"
- "Việc vi phạm tốc độ tối đa sẽ bị xử phạt như thế nao?"
- "Ai có trách nhiệm kiểm tra định kỳ phương tiện?"

Trả về JSON với qa_pairs và sources (để trống - sẽ được set ở code).
        """
