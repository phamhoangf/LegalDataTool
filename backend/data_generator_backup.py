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
    """Response chứa danh sách QA và sources"""
    qa_pairs: List[LegalQA]
    sources: List[SourceReference]  # Danh sách các nguồn tham chiếu cho tất cả QA

class DataGenerator:
    """Class sinh dữ liệu huấn luyện cho LegalSLM theo độ khó reasoning"""
    
    def __init__(self, api_key: str = None, similarity_threshold: float = 0.75):
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
        
        # Khởi tạo similarity checker
        self.similarity_checker = QuestionSimilarityChecker(similarity_threshold=similarity_threshold)
        print(f"🔍 Initialized similarity checker with threshold {similarity_threshold}")
    
    def get_rule_based_difficulty(self, data_type: str, num_sources: int) -> str:
        """
        Xác định độ khó theo rule-based thay vì yêu cầu LLM tạo ra
        
        Args:
            data_type: Loại data type
            num_sources: Số lượng nguồn được sử dụng
            
        Returns:
            str: Mức độ khó (easy/medium/hard)
        """
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
        """
        Cập nhật corpus cho similarity checker với dữ liệu hiện có
        
        Args:
            existing_questions_data: List các dict chứa câu hỏi từ database
        """
        self.similarity_checker.update_corpus(existing_questions_data)
    
    def filter_duplicate_questions(self, new_samples: List[Dict[str, Any]], verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Lọc bỏ các câu hỏi trùng lặp từ danh sách samples mới
        
        Args:
            new_samples: List các samples mới được generate
            verbose: In thông tin chi tiết
            
        Returns:
            List[Dict]: Danh sách samples sau khi lọc
        """
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
                    if result['similar_questions']:
                        best_match = result['similar_questions'][0]
                        print(f"   Similar to: {best_match['question'][:60]}...")
        
        if verbose and duplicates_found > 0:
            print(f"🔍 Filtered {duplicates_found}/{len(new_samples)} duplicate questions")
        elif verbose:
            print(f"✅ No duplicates found in {len(new_samples)} questions")
        
        return filtered_samples
    
    def split_law_by_article(self, text: str, document_title: str = "") -> List[Dict]:
        """
        Tách văn bản luật thành các đơn vị, mỗi đơn vị là một 'Điều'.
        Hàm sẽ bỏ qua các phần không phải là Điều (như tiêu đề Chương).
        
        Args:
            text (str): Chuỗi văn bản luật cần tách.
            document_title (str): Tiêu đề tài liệu để thêm vào metadata
            
        Returns:
            list[dict]: Một danh sách các từ điển, mỗi từ điển đại diện cho một Điều.
        """
        units = []
        
        # Pattern để tìm các dòng có "Điều X." (có thể có spaces trước)
        split_pattern = r'(?m)(?=^\s*Điều \d+\.)'
        
        # Tách văn bản thành các khối (chunks)
        chunks = re.split(split_pattern, text.strip())
        
        # Duyệt qua các khối và chỉ xử lý những khối chứa "Điều"
        for chunk in chunks:
            chunk = chunk.strip()
            # Tìm dòng bắt đầu bằng "Điều" (có thể có spaces)
            lines = chunk.split('\n')
            dieu_line = None
            for line in lines:
                if re.match(r'^\s*Điều \d+\.', line):
                    dieu_line = line.strip()
                    break
            
            if dieu_line:
                # Trích xuất số hiệu của Điều để làm ID
                match = re.search(r'Điều (\d+)', dieu_line)
                if match:
                    article_number = match.group(1)
                    unit_id = f"article_{article_number}"
                else:
                    # Nếu không tìm thấy số, tạo ID dự phòng
                    unit_id = f"unknown_article_{len(units) + 1}"
                    
                units.append({
                    "id": unit_id,
                    "title": dieu_line,
                    "content": chunk,
                    "document_title": document_title,
                    "metadata": {
                        "article_number": article_number if match else None,
                        "source_document": document_title,
                        "unit_type": "article",
                        "length": len(chunk)
                    }
                })
                
        return units
    
    def get_articles_from_parsed_structure(self, document) -> List[Dict]:
        """
        Lấy danh sách articles từ parsed structure (nếu có) hoặc fallback về split_law_by_article
        
        Args:
            document: Document object có .title, .content và .parsed_structure
            
        Returns:
            List[Dict]: Danh sách articles cho Monte Carlo sampling
        """
        # Kiểm tra parsed structure
        if hasattr(document, 'parsed_structure') and document.parsed_structure:
            try:
                parsed_data = json.loads(document.parsed_structure)
                parser = LegalDocumentParser()
                articles = parser.get_all_articles(parsed_data)
                
                # Convert to format compatible với data generator
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
                            "path": article['path'],
                            "length": article['content_length']
                        }
                    })
                
                print(f"✅ Using parsed structure: {len(units)} articles from {document.title}")
                return units
                
            except Exception as e:
                print(f"⚠️ Failed to use parsed structure: {str(e)}, falling back to split_law_by_article")
        
        # Fallback về method cũ
        print(f"📄 Using fallback parsing for {document.title}")
        return self.split_law_by_article(document.content, document.title)
    
    def monte_carlo_sample_articles(self, all_articles: List[Dict], sample_size: int) -> List[Dict]:
        """
        Improved Monte Carlo sampling với entropy injection để tránh trùng lặp
        
        Args:
            all_articles: Tất cả articles available
            sample_size: Số lượng articles cần lấy
            
        Returns:
            List[Dict]: Articles được chọn theo Monte Carlo distribution với high entropy
        """
        if not all_articles or sample_size <= 0:
            return []
            
        if sample_size >= len(all_articles):
            # Shuffle để đảm bảo random order ngay cả khi lấy hết
            articles_copy = all_articles.copy()
            random.shuffle(articles_copy)
            return articles_copy
        
        # Bước 1: Tính weights với entropy injection
        weights = []
        for i, article in enumerate(all_articles):
            # Base weight từ content length
            content_length = article.get('metadata', {}).get('length', len(article.get('content', '')))
            length_weight = min(content_length / 800, 2.5)  # Adjusted for better distribution
            
            # Position diversity weight
            article_num = article.get('metadata', {}).get('article_number')
            position_weight = 1.0
            if article_num:
                try:
                    num = int(article_num)
                    # Strategic articles có weight cao hơn
                    if num <= 5 or num % 25 == 0 or num in [10, 15, 20, 30, 40, 60, 80, 100]:
                        position_weight = 1.8
                    elif num <= 20 or num % 10 == 0:
                        position_weight = 1.4
                except:
                    pass
            
            # Document diversity weight (tránh chọn quá nhiều từ cùng 1 document)
            doc_title = article.get('document_title', '')
            doc_hash = hash(doc_title) % 1000
            doc_weight = 0.8 + (doc_hash / 1000) * 0.4  # Range: 0.8-1.2
            
            # Entropy injection - thêm random factor để tăng diversity
            entropy_factor = random.uniform(0.7, 1.3)
            
            # Final weight với multiple factors
            final_weight = length_weight * position_weight * doc_weight * entropy_factor
            weights.append(max(final_weight, 0.15))  # Higher minimum weight
        
        # Bước 2: Multi-round Monte Carlo sampling để tăng diversity
        selected = []
        available_articles = all_articles.copy()
        available_weights = weights.copy()
        
        # Perform sampling in multiple rounds với different strategies
        rounds = min(3, sample_size)  # Max 3 rounds
        samples_per_round = sample_size // rounds
        remaining_samples = sample_size % rounds
        
        for round_num in range(rounds):
            round_samples = samples_per_round + (1 if round_num < remaining_samples else 0)
            
            if not available_articles or round_samples <= 0:
                break
            
            # Different entropy per round
            entropy_multiplier = 1.0 + (round_num * 0.3)  # Tăng entropy qua mỗi round
            
            for _ in range(round_samples):
                if not available_articles:
                    break
                
                # Apply entropy multiplier to weights
                current_weights = [w * entropy_multiplier * random.uniform(0.9, 1.1) 
                                 for w in available_weights]
                
                total_weight = sum(current_weights)
                if total_weight == 0:
                    chosen_idx = random.randint(0, len(available_articles) - 1)
                else:
                    # Improved Monte Carlo selection
                    rand_val = random.uniform(0, total_weight)
                    cumsum = 0
                    chosen_idx = len(current_weights) - 1  # Default to last
                    
                    for i, weight in enumerate(current_weights):
                        cumsum += weight
                        if rand_val <= cumsum:
                            chosen_idx = i
                            break
                
                # Add selected article
                selected.append(available_articles[chosen_idx])
                
                # Remove from available (sampling without replacement)
                available_articles.pop(chosen_idx)
                available_weights.pop(chosen_idx)
        
        # Final shuffle để đảm bảo order không predictable
        random.shuffle(selected)
        
        print(f"🎲 Enhanced Monte Carlo sampling: chọn {len(selected)}/{len(all_articles)} articles với high entropy ({rounds} rounds)")
        return selected
    
    def generate_from_multiple_documents(self, documents, topic_name, data_type, num_samples):
        """
        Sinh dữ liệu từ nhiều documents bằng Monte Carlo sampling
        """
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
        print(f"🎲 Sử dụng Monte Carlo sampling cho articles...")
        max_articles = min(len(all_articles), max(num_samples // 2, 5))
        selected_articles = self.monte_carlo_sample_articles(all_articles, max_articles)

        print(f"  🎯 Đã chọn {len(selected_articles)} articles bằng Monte Carlo")

        # Sinh dữ liệu
        all_samples = self.generate_samples_from_articles(selected_articles, topic_name, data_type, num_samples)

        # Lọc trùng lặp
        print(f"🔍 Kiểm tra tương đồng cho {len(all_samples)} samples...")
        filtered_samples = self.filter_duplicate_questions(all_samples, verbose=True)

        print(f"✅ Hoàn thành: {len(filtered_samples)} samples (đã lọc {len(all_samples) - len(filtered_samples)} trùng lặp)")
        return filtered_samples[:num_samples]
    
    def generate_samples_from_articles(self, articles, topic, data_type, num_samples):
        """
        Sinh dữ liệu đơn giản từ articles với sources chung
        """
        all_samples = []
        
        # Xác định số sources cần thiết
        num_sources_map = {
            'word_matching': min(3, len(articles)),
            'concept_understanding': min(4, len(articles)), 
            'multi_paragraph_reading': min(6, len(articles)),
            'multi_hop_reasoning': min(8, len(articles))
        }
        num_sources = num_sources_map.get(data_type, min(3, len(articles)))
        
        # Chọn articles đa dạng cho sources chung
        if len(articles) <= num_sources:
            selected_articles = articles.copy()
            random.shuffle(selected_articles)
        else:
            selected_articles = random.sample(articles, num_sources)
        
        # Tạo sources chung cho tất cả câu hỏi
        common_sources = []
        combined_content = []
        
        for article in selected_articles:
            source_ref = SourceReference(
                article_number=str(article['metadata']['article_number']) if article['metadata']['article_number'] else "unknown",
                article_title=article['title'],
                document_title=article['document_title']
            )
            common_sources.append(source_ref)
            combined_content.append(f"--- {article['title']} (từ {article['document_title']}) ---\n{article['content']}")

        combined_text = "\n\n".join(combined_content)
        
        # Rule-based difficulty
        difficulty = self.get_rule_based_difficulty(data_type, len(selected_articles))
        
        # Tạo prompt đa dạng
        for i in range(num_samples):
            prompt = self.create_diverse_prompt(combined_text, topic, data_type, len(selected_articles), difficulty)
            
            try:
                # Dynamic parameters
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
                
                # Convert với sources chung
                for qa_pair in structured_data.qa_pairs:
                    sample = {
                        'question': qa_pair.question,
                        'answer': qa_pair.answer,
                        'difficulty': difficulty,  # Rule-based difficulty
                        'sources': [
                            {
                                'article_number': src.article_number,
                                'article_title': src.article_title,
                                'document_title': src.document_title
                            } for src in common_sources
                        ],
                        'metadata': {
                            'generation_method': 'simplified_multi_source',
                            'num_sources': len(selected_articles),
                            'temperature': temperature
                        }
                    }
                    all_samples.append(sample)
                    
            except Exception as e:
                print(f"❌ Generation failed for sample {i+1}: {e}")
                continue
        
        return all_samples[:num_samples]

    def create_diverse_prompt(self, content, topic, data_type, num_sources, difficulty):
        """
        Tạo prompt đa dạng để tránh trùng lặp
        """
        # Cấu trúc câu hỏi đa dạng
        question_starters = [
            "Khi nào", "Trong trường hợp nào", "Ai có trách nhiệm",
            "Việc...được thực hiện như thế nào", "Điều kiện...là gì",
            "Mức phạt...là bao nhiêu", "Quy trình...diễn ra ra sao",
            "Tại sao", "Vì sao", "Làm cách nào", "Bằng phương thức nào",
            "Có được phép", "Có bắt buộc", "Có cần thiết"
        ]
        
        focus_areas = [
            "quy định thực tế và ứng dụng",
            "trường hợp ngoại lệ và điều kiện đặc biệt", 
            "nghĩa vụ và quyền của các đối tượng",
            "mức phạt và hậu quả vi phạm",
            "quy trình và thủ tục pháp lý",
            "định nghĩa và thuật ngữ chuyên môn",
            "thẩm quyền và trách nhiệm"
        ]
        
        starter = random.choice(question_starters)
        focus = random.choice(focus_areas)
        entropy_id = random.randint(1000, 9999)
        
        return f"""
        Dưới đây là {num_sources} điều luật về chủ đề "{topic}":
        
        {content}
        
        Hãy tạo 1 câu hỏi độ khó {difficulty} về {focus}.
        
        YÊU CẦU:
        1. TUYỆT ĐỐI KHÔNG dùng cấu trúc "Theo Điều X..."
        2. Bắt đầu câu hỏi bằng: "{starter}..." hoặc tương tự
        3. Câu hỏi phải độc lập, không nhắc tên điều luật
        4. Tập trung vào {focus}
        5. Entropy ID: {entropy_id} (để tạo uniqueness)
        
        VÍ DỤ CẤAU TRÚC TỐT:
        - "Khi nào doanh nghiệp cần có giấy phép kinh doanh vận tải?"
        - "Việc vi phạm tốc độ sẽ bị xử phạt như thế nào?"
        - "Ai có trách nhiệm kiểm tra tình trạng kỹ thuật của xe?"
        
        Trả về JSON với qa_pairs và sources (để trống, sẽ được set ở code).
        """
    
    def generate_multi_source_data_from_articles(self, articles, topic, data_type, num_samples):
        """Sinh dữ liệu từ nhiều articles cho multi-paragraph và multi-hop reasoning"""
        
        # Group articles by document for better diversity
        articles_by_doc = {}
        for article in articles:
            doc_title = article['document_title']
            if doc_title not in articles_by_doc:
                articles_by_doc[doc_title] = []
            articles_by_doc[doc_title].append(article)
        
        all_samples = []
        
        for i in range(num_samples):
            # Chọn số nguồn phù hợp với từng loại câu hỏi
            if data_type == 'word_matching':
                num_sources = min(5, len(articles))  # Giảm xuống để đảm bảo quality
            elif data_type == 'concept_understanding':
                num_sources = min(5, len(articles))  
            elif data_type == 'multi_paragraph_reading':
                num_sources = min(7, len(articles))  
            elif data_type == 'multi_hop_reasoning':
                num_sources = min(10, len(articles))  
            else:
                num_sources = min(5, len(articles))  # Default
                
            selected_articles = self._select_diverse_articles(articles_by_doc, num_sources)
            
            # Fallback nếu không đủ articles
            if len(selected_articles) < 1:
                if articles:
                    samples = self.generate_structured_data_from_article(articles[0], topic, data_type, 1)
                    all_samples.extend(samples)
                continue
            
            # Tạo source references từ các articles đã chọn
            source_refs = []
            combined_content = []
            
            for article in selected_articles:
                source_ref = SourceReference(
                    article_number=str(article['metadata']['article_number']) if article['metadata']['article_number'] else "unknown",
                    article_title=article['title'],
                    document_title=article['document_title'],
                    document_number=article.get('document_number', 'unknown')
                )
                source_refs.append(source_ref)
                combined_content.append(f"--- {article['title']} (từ {article['document_title']}) ---\n{article['content']}")
            
            # Shuffle thứ tự các điều trong prompt để tạo đa dạng
            content_with_refs = list(zip(combined_content, source_refs))
            random.shuffle(content_with_refs)
            combined_content, source_refs = zip(*content_with_refs)
            combined_content = list(combined_content)
            source_refs = list(source_refs)
            
            # Tạo prompt với multiple sources
            difficulty_description = {
                'word_matching': 'Word Matching - câu hỏi đơn giản, có thể trả lời bằng tìm kiếm từ khóa trong văn bản. Nếu có nhiều nguồn, hỏi về thông tin có trong một trong các nguồn',
                'concept_understanding': 'Concept Understanding - cần hiểu khái niệm pháp lý cơ bản. Có thể kết hợp thông tin từ nhiều nguồn để hiểu rõ hơn về khái niệm',
                'multi_paragraph_reading': 'Multi-Paragraph Reading - cần đọc và tổng hợp thông tin từ nhiều đoạn/điều khác nhau',
                'multi_hop_reasoning': 'Multi-Hop Reasoning - phức tạp nhất, cần nhiều bước suy luận và kết hợp thông tin từ nhiều nguồn'
            }
            
            combined_text = "\n\n".join(combined_content)
            
            # Tạo instruction phù hợp với số nguồn
            if len(selected_articles) == 1:
                instruction = f"Hãy tạo 1 câu hỏi dạng {difficulty_description[data_type]}."
            else:
                if data_type in ['word_matching', 'concept_understanding']:
                    instruction = f"Hãy tạo 1 câu hỏi dạng {difficulty_description[data_type]}. HÃY CHỌN MỘT ĐIỀU CỤ THỂ từ {len(selected_articles)} điều được cung cấp để làm câu hỏi, không được trộn lẫn nhiều điều."
                else:
                    instruction = f"Hãy tạo 1 câu hỏi dạng {difficulty_description[data_type]}. Câu hỏi phải YÊU CẦU THÔNG TIN TỪ ÍT NHẤT 2 ĐIỀU KHÁC NHAU trong {len(selected_articles)} điều đã cho."
            
            # Tạo prompt với enhanced diversity và anti-duplication measures
            # Đa dạng cấu trúc câu hỏi để tránh "Theo Điều X..." 
            question_structures = [
                "Câu hỏi bắt đầu bằng 'Khi nào', 'Trong trường hợp nào', 'Ai có trách nhiệm'",
                "Câu hỏi dạng 'Việc... được thực hiện như thế nào?', 'Điều kiện... là gì?'",
                "Câu hỏi về 'Mức phạt', 'Hậu quả', 'Quy trình', 'Thủ tục'",
                "Câu hỏi dạng 'Tại sao...', 'Vì sao...', 'Lý do nào...'",
                "Câu hỏi về 'Có được phép...', 'Có bắt buộc...', 'Có cần thiết...'",
                "Câu hỏi dạng 'Làm cách nào...', 'Bằng cách nào...', 'Qua phương thức nào...'",
                "Câu hỏi về 'Khác biệt', 'Giống nhau', 'Phân biệt'",
                "Câu hỏi về 'Ưu tiên', 'Quan trọng', 'Cấp thiết'",
                "Câu hỏi dạng 'Ngoại trừ', 'Trừ trường hợp', 'Ngoại lệ'",
                "Câu hỏi về 'Giải pháp', 'Biện pháp', 'Cách thức'"
            ]
            
            diversity_hints = [
                "Tập trung vào khía cạnh thực tiễn, hỏi về ứng dụng thực tế",
                "Hỏi về quy định cụ thể và chi tiết trong thực hiện",
                "Chú ý đến các trường hợp ngoại lệ hoặc điều kiện đặc biệt",
                "Tập trung vào nghĩa vụ và quyền của từng đối tượng khác nhau",
                "Hỏi về mức phạt hoặc hậu quả vi phạm cụ thể",
                "Tập trung vào quy trình và thủ tục pháp lý chi tiết",
                "Hỏi về định nghĩa và thuật ngữ pháp lý chính xác",
                "Chú ý đến sự khác biệt giữa các loại đối tượng được quy định",
                "Tập trung vào các yêu cầu kỹ thuật cụ thể và rõ ràng",
                "Hỏi về thẩm quyền và trách nhiệm của các cơ quan"
            ]
            
            # Random selection cho diversity
            question_structure = random.choice(question_structures)
            diversity_hint = random.choice(diversity_hints)
            
            # Entropy injection cho prompt
            entropy_elements = [
                f"Mẫu số #{random.randint(100, 999)} - ",
                f"Góc độ #{random.choice(['A', 'B', 'C', 'D', 'E'])}: ",
                f"Khía cạnh #{random.randint(1, 20)}: ",
                ""  # Sometimes no prefix
            ]
            entropy_prefix = random.choice(entropy_elements)
            
            # Dynamic instruction với entropy
            base_instructions = [
                f"Hãy tạo 1 câu hỏi dạng {difficulty_description[data_type]}",
                f"Sinh 1 câu hỏi thuộc loại {difficulty_description[data_type]}",
                f"Tạo ra 1 câu hỏi có độ khó {difficulty_description[data_type]}",
            ]
            base_instruction = random.choice(base_instructions)
            
            if len(selected_articles) == 1:
                instruction = f"{entropy_prefix}{base_instruction}. {diversity_hint}. CẤU TRÚC: {question_structure}."
            else:
                if data_type in ['word_matching', 'concept_understanding']:
                    source_instructions = [
                        f"HÃY CHỌN MỘT ĐIỀU CỤ THỂ từ {len(selected_articles)} điều được cung cấp để làm câu hỏi, không được trộn lẫn nhiều điều",
                        f"Chọn một điều luật cụ thể trong {len(selected_articles)} điều đã cho để xây dựng câu hỏi",
                        f"Dựa trên một trong {len(selected_articles)} điều luật để tạo câu hỏi, không kết hợp nhiều điều",
                    ]
                    source_instruction = random.choice(source_instructions)
                    instruction = f"{entropy_prefix}{base_instruction}. {source_instruction}. {diversity_hint}. CẤU TRÚC: {question_structure}."
                else:
                    multi_instructions = [
                        f"Câu hỏi phải YÊU CẦU THÔNG TIN TỪ ÍT NHẤT 2 ĐIỀU KHÁC NHAU trong {len(selected_articles)} điều đã cho",
                        f"Kết hợp thông tin từ ít nhất 2 điều luật khác nhau trong {len(selected_articles)} điều",
                        f"Tổng hợp nội dung từ nhiều điều luật (tối thiểu 2 điều) trong {len(selected_articles)} điều đã cung cấp",
                    ]
                    multi_instruction = random.choice(multi_instructions)
                    instruction = f"{entropy_prefix}{base_instruction}. {multi_instruction}. {diversity_hint}. CẤU TRÚC: {question_structure}."
            
            # Additional anti-duplication measures
            anti_dup_phrases = [
                "Tránh hỏi những câu hỏi quá tương tự với các mẫu thông thường",
                "Đảm bảo câu hỏi có góc nhìn độc đáo và mới lạ",
                "Tạo câu hỏi có tính sáng tạo cao, khác biệt với các câu hỏi thường gặp",
                "Thiết kế câu hỏi theo hướng tiếp cận mới, không lặp lại các mẫu cũ"
            ]
            anti_dup_phrase = random.choice(anti_dup_phrases)
            
            prompt = f"""
            Dưới đây là {len(selected_articles)} điều luật khác nhau về chủ đề "{topic}":
            
            {combined_text}
            
            {instruction}
            
            YÊU CẦU QUAN TRỌNG:
            1. TUYỆT ĐỐI KHÔNG bắt đầu câu hỏi bằng "Theo Điều X của..."
            2. Sử dụng cấu trúc câu hỏi đa dạng, sáng tạo
            3. Câu hỏi phải ĐỘC LẬP, rõ ràng nhưng KHÔNG nhắc trực tiếp tên điều luật trong câu hỏi
            4. CHỈNH SỬA CÁCH ĐẶT CÂU: thay vì "Theo Điều X..." hãy dùng "Khi nào...", "Ai có trách nhiệm...", "Việc... được thực hiện như thế nào?", "Điều kiện... là gì?"
            5. Câu trả lời phải chính xác từ nội dung các điều luật và GHI RÕ NGUỒN trong phần sources
            6. {anti_dup_phrase}
            7. {self._get_source_requirement(data_type, len(selected_articles))}
            8. Entropy factor: {random.randint(1000, 9999)} - sử dụng để tạo uniqueness
            
            Danh sách các điều có sẵn (chỉ để tham khảo, KHÔNG đề cập trực tiếp trong câu hỏi):
            {chr(10).join([f"- Điều {ref.article_number}: {ref.article_title} (từ {ref.document_title})" for ref in source_refs])}
            
            HƯỚNG DẪN ĐA DẠNG: {diversity_hint}
            CẤU TRÚC CÂU HỎI: {question_structure}
            
            VÍ DỤ CẤU TRÚC TỐT:
            - "Khi nào doanh nghiệp vận tải cần phải có giấy phép?"
            - "Ai có trách nhiệm kiểm tra tình trạng kỹ thuật xe?"
            - "Việc vi phạm tải trọng xe sẽ bị xử phạt như thế nào?"
            - "Điều kiện để được cấp phép kinh doanh vận tải là gì?"
            
            Trả về theo format JSON với sources chứa các nguồn đã sử dụng.
            """
            
            try:
                # Dynamic temperature và parameters để tăng diversity
                temperature_range = {
                    'word_matching': (0.4, 0.7),
                    'concept_understanding': (0.5, 0.8), 
                    'multi_paragraph_reading': (0.6, 0.9),
                    'multi_hop_reasoning': (0.7, 1.0)
                }
                
                min_temp, max_temp = temperature_range.get(data_type, (0.3, 0.6))
                dynamic_temperature = random.uniform(min_temp, max_temp)
                
                # Top-p và Top-k sampling cho diversity
                top_p = random.uniform(0.8, 0.95)
                
                # Seed randomization để tránh deterministic output
                random_seed = random.randint(1, 1000000)
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=dynamic_temperature,
                        top_p=top_p,
                        max_output_tokens=4000,
                        response_mime_type="application/json",
                        response_schema=LegalQAResponse,
                        seed=random_seed  # Random seed cho mỗi generation
                    )
                )
                
                print(f"  🎯 Generated with temp={dynamic_temperature:.2f}, top_p={top_p:.2f}, seed={random_seed}")
                
                # Parse structured response
                structured_data: LegalQAResponse = response.parsed
                
                # Convert to legacy format với metadata từ nhiều nguồn
                for qa_pair in structured_data.qa_pairs:
                    legacy_sample = {
                        'question': qa_pair.question,
                        'answer': qa_pair.answer,
                        'difficulty': qa_pair.difficulty,
                        'sources': [
                            {
                                'article_number': src.article_number,
                                'article_title': src.article_title,
                                'document_title': src.document_title,
                                'document_number': src.document_number
                            } for src in source_refs  # Sử dụng tất cả sources
                        ],
                        'metadata': {
                            'source_articles': [art['title'] for art in selected_articles],
                            'source_documents': list(set([art['document_title'] for art in selected_articles])),
                            'article_ids': [art['id'] for art in selected_articles],
                            'article_numbers': [art['metadata']['article_number'] for art in selected_articles],
                            'generation_method': 'multi_source_structured',
                            'num_sources': len(selected_articles)
                        }
                    }
                    all_samples.append(legacy_sample)
                    
            except Exception as e:
                print(f"❌ Multi-source generation failed for sample {i+1}: {e}")
                # Fallback to single article
                if selected_articles:
                    samples = self.generate_structured_data_from_article(selected_articles[0], topic, data_type, 1)
                    all_samples.extend(samples)
        
        return all_samples[:num_samples]
    
    def _select_diverse_articles(self, articles_by_doc, num_sources):
        """
        Chọn articles đa dạng từ các documents khác nhau với entropy injection
        
        Args:
            articles_by_doc: Dict mapping document title -> list of articles
            num_sources: Số lượng sources cần chọn
            
        Returns:
            List[Dict]: Danh sách articles được chọn với high diversity
        """
        all_articles = []
        for doc_title, articles in articles_by_doc.items():
            all_articles.extend(articles)
        
        if len(all_articles) <= num_sources:
            random.shuffle(all_articles)  # Shuffle để tránh bias
            return all_articles
        
        selected = []
        used_documents = set()
        used_article_numbers = set()
        
        # Round 1: Chọn từ different documents để đảm bảo diversity
        available_docs = list(articles_by_doc.keys())
        random.shuffle(available_docs)  # Shuffle documents order
        
        for doc_title in available_docs:
            if len(selected) >= num_sources:
                break
                
            articles = articles_by_doc[doc_title]
            if not articles:
                continue
                
            # Thêm entropy cho việc chọn article trong document
            articles_with_weights = []
            for article in articles:
                article_num = article.get('metadata', {}).get('article_number')
                
                # Skip nếu đã dùng article number tương tự
                if article_num and article_num in used_article_numbers:
                    continue
                    
                # Weight dựa trên content length và randomness
                content_len = len(article.get('content', ''))
                length_weight = min(content_len / 500, 2.0)
                
                # Position diversity
                position_weight = 1.0
                if article_num:
                    try:
                        num = int(article_num)
                        if num <= 5 or num % 15 == 0:  # Strategic numbers
                            position_weight = 1.5
                    except:
                        pass
                
                # High entropy factor
                entropy_factor = random.uniform(0.5, 1.8)  # Wider range cho diversity
                
                weight = length_weight * position_weight * entropy_factor
                articles_with_weights.append((article, weight))
            
            if articles_with_weights:
                # Chọn article có weight cao nhất với random factor
                articles_with_weights.sort(key=lambda x: x[1] * random.uniform(0.8, 1.2), reverse=True)
                selected_article = articles_with_weights[0][0]
                
                selected.append(selected_article)
                used_documents.add(doc_title)
                
                article_num = selected_article.get('metadata', {}).get('article_number')
                if article_num:
                    used_article_numbers.add(article_num)
        
        # Round 2: Fill remaining slots với articles từ any document
        if len(selected) < num_sources:
            remaining_articles = []
            for article in all_articles:
                if article not in selected:
                    article_num = article.get('metadata', {}).get('article_number')
                    # Tránh duplicate article numbers
                    if not article_num or article_num not in used_article_numbers:
                        remaining_articles.append(article)
            
            # Sort by content length với random factor cho diversity
            remaining_articles.sort(
                key=lambda x: len(x.get('content', '')) * random.uniform(0.7, 1.3), 
                reverse=True
            )
            
            need_more = num_sources - len(selected)
            for article in remaining_articles[:need_more]:
                selected.append(article)
                article_num = article.get('metadata', {}).get('article_number')
                if article_num:
                    used_article_numbers.add(article_num)
        
        # Final shuffle để loại bỏ order bias
        random.shuffle(selected)
        
        # Debug info
        selected_docs = [art['document_title'] for art in selected]
        selected_nums = [art.get('metadata', {}).get('article_number', 'unknown') for art in selected]
        print(f"  🎯 Selected diverse articles: {selected_nums} from docs: {list(set(selected_docs))}")
        
        return selected
    
    def _get_source_requirement(self, data_type: str, num_sources: int) -> str:
        """Tạo yêu cầu về nguồn phù hợp với từng loại câu hỏi"""
        if num_sources == 1:
            return "Câu trả lời dựa trên thông tin từ nguồn đã cho"
        
        if data_type in ['word_matching', 'concept_understanding']:
            return f"Câu trả lời có thể dựa trên thông tin từ một hoặc nhiều nguồn trong {num_sources} nguồn đã cho"
        else:
            return f"Câu trả lời phải tổng hợp thông tin từ ít nhất 2 nguồn trong {num_sources} nguồn đã cho"

    def generate_structured_data_from_article(self, article, topic, data_type, num_samples):
        """Sinh dữ liệu có cấu trúc từ một article cụ thể"""
        
        # Tạo source reference từ article
        source_ref = SourceReference(
            article_number=str(article['metadata']['article_number']) if article['metadata']['article_number'] else "unknown",
            article_title=article['title'],
            document_title=article['document_title'],
            document_number=article.get('document_number', 'unknown')
        )
        
        # Tạo prompt với structured output
        difficulty_description = {
            'word_matching': 'Word Matching - câu hỏi đơn giản nhất, chỉ cần tìm kiếm từ khóa/cụm từ trong văn bản',
            'concept_understanding': 'Concept Understanding - cần hiểu khái niệm pháp lý cơ bản',
            'multi_paragraph_reading': 'Multi-Paragraph Reading - cần đọc và tổng hợp thông tin từ nhiều đoạn',
            'multi_hop_reasoning': 'Multi-Hop Reasoning - phức tạp nhất, cần nhiều bước suy luận'
        }
        
        prompt = f"""
        Dựa trên điều luật sau về chủ đề "{topic}":
        
        {article['content']}
        
        Hãy tạo {num_samples} câu hỏi dạng {difficulty_description[data_type]}.
        
        YÊU CÂU QUAN TRỌNG:
        1. Mỗi câu hỏi phải ĐỘC LẬP, rõ ràng, có tên luật/văn bản cụ thể
        2. KHÔNG dùng "luật này", "văn bản này", "điều này" - phải nói rõ tên
        3. Câu trả lời phải CHÍNH XÁC từ nội dung điều luật
        4. Phải ghi rõ nguồn tham chiếu đến điều và tài liệu cụ thể
        
        Thông tin nguồn:
        - Điều số: {source_ref.article_number}
        - Tiêu đề điều: {source_ref.article_title}
        - Tài liệu: {source_ref.document_title}
        
        Trả về theo format JSON structure yêu cầu.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=4000,
                    response_mime_type="application/json",
                    response_schema=LegalQAResponse
                )
            )
            
            # Parse structured response
            structured_data: LegalQAResponse = response.parsed
            
            # Convert to legacy format với metadata đầy đủ
            legacy_samples = []
            for qa_pair in structured_data.qa_pairs:
                legacy_sample = {
                    'question': qa_pair.question,
                    'answer': qa_pair.answer,
                    'difficulty': qa_pair.difficulty,
                    'sources': [
                        {
                            'article_number': src.article_number,
                            'article_title': src.article_title,
                            'document_title': src.document_title,
                            'document_number': src.document_number
                        } for src in qa_pair.sources
                    ],
                    'metadata': {
                        'source_article': article['title'],
                        'source_document': article['document_title'],
                        'article_id': article['id'],
                        'article_number': article['metadata']['article_number'],
                        'generation_method': 'structured_article_based'
                    }
                }
                legacy_samples.append(legacy_sample)
            
            return legacy_samples
            
        except Exception as e:
            print(f"❌ Structured generation failed, fallback to legacy: {e}")
            # Fallback to legacy method
            context = self._create_article_context(article)
            return self._call_generation_method(context, topic, data_type, num_samples)
    
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
        - Không được sinh ra các câu hỏi giống nhau
        
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
