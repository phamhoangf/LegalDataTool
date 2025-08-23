from google import genai
from google.genai import types
import json
import random
import os
import re
from typing import List, Dict, Any
from pydantic import BaseModel
from similarity_checker import QuestionSimilarityChecker

class SourceReference(BaseModel):
    """Tham chiếu đến nguồn của thông tin"""
    article_number: str  # Số điều (ví dụ: "60", "61")
    article_title: str   # Tiêu đề điều (ví dụ: "Điều 60. Độ tuổi của người lái xe")
    document_title: str  # Tên tài liệu (ví dụ: "Luật Giao thông đường bộ 2008")
    document_number: str # Số hiệu văn bản (ví dụ: "23/2008/QH12")

class LegalQA(BaseModel):
    """Cấu trúc câu hỏi-đáp án pháp lý"""
    question: str
    answer: str
    difficulty: str
    sources: List[SourceReference]  # Danh sách các nguồn tham chiếu

class LegalQAResponse(BaseModel):
    """Response chứa danh sách QA"""
    qa_pairs: List[LegalQA]

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
    
    def generate_from_multiple_documents(self, documents, topic_name, data_type, num_samples):
        """
        Sinh dữ liệu từ nhiều documents bằng cách tách theo Điều và phân bố công bằng
        
        Args:
            documents: List of document objects (có .title và .content)
            topic_name: Tên chủ đề
            data_type: Loại dữ liệu cần sinh
            num_samples: Số lượng samples cần sinh
            
        Returns:
            List[Dict]: Danh sách samples với metadata đầy đủ
        """
        if not documents:
            return []
        
        print(f"🔍 Phân tích {len(documents)} documents...")
        
        # Bước 1: Tách tất cả documents thành articles
        all_articles = []
        document_stats = {}
        
        for doc in documents:
            # Tách document thành các Điều
            articles = self.split_law_by_article(doc.content, doc.title)
            all_articles.extend(articles)
            
            document_stats[doc.title] = {
                'total_articles': len(articles),
                'total_length': len(doc.content)
            }
            
            print(f"  📋 {doc.title}: {len(articles)} điều")
        
        print(f"📊 Tổng cộng: {len(all_articles)} điều từ {len(documents)} tài liệu")
        
        # Bước 2: Chọn articles để sinh dữ liệu (round-robin hoặc random)
        selected_articles = self._select_articles_for_generation(all_articles, num_samples)
        
        # Bước 3: Sinh dữ liệu từ các articles đã chọn với multi-source cho tất cả loại
        all_samples = []
        
        # Sử dụng multi-source generation cho tất cả data types
        print(f"  � Sử dụng multi-source generation cho {data_type}")
        all_samples = self.generate_multi_source_data_from_articles(selected_articles, topic_name, data_type, num_samples)
        
        # Bước 4: Lọc các câu hỏi trùng lặp
        print(f"🔍 Kiểm tra tương đồng cho {len(all_samples)} samples...")
        filtered_samples = self.filter_duplicate_questions(all_samples, verbose=True)
        
        print(f"✅ Hoàn thành: {len(filtered_samples)} samples (đã lọc {len(all_samples) - len(filtered_samples)} trùng lặp)")
        return filtered_samples[:num_samples]  # Đảm bảo không vượt quá số lượng yêu cầu
    
    def _select_articles_for_generation(self, all_articles, num_samples):
        """Chọn articles để sinh dữ liệu"""
        if not all_articles:
            return []
        
        # Lọc articles có content đủ dài (tối thiểu 100 chars)
        valid_articles = [art for art in all_articles if len(art['content']) >= 100]
        
        if not valid_articles:
            return all_articles[:num_samples]  # Fallback
        
        # Sắp xếp theo độ dài (ưu tiên articles dài hơn)
        valid_articles.sort(key=lambda x: len(x['content']), reverse=True)
        
        # Chọn số lượng articles phù hợp
        max_articles = min(len(valid_articles), max(num_samples // 2, 5))
        
        return valid_articles[:max_articles]
    
    def _create_article_context(self, article):
        """Tạo context từ một article"""
        return f"""--- {article['title']} (từ {article['document_title']}) ---
{article['content']}"""
    
    def _call_generation_method(self, context, topic, data_type, num_samples):
        """Gọi method generation phù hợp"""
        try:
            if data_type == 'word_matching':
                return self.generate_word_matching_data(context, topic, num_samples)
            elif data_type == 'concept_understanding':
                return self.generate_concept_understanding_data(context, topic, num_samples)
            elif data_type == 'multi_paragraph_reading':
                return self.generate_multi_paragraph_reading_data(context, topic, num_samples)
            elif data_type == 'multi_hop_reasoning':
                return self.generate_multi_hop_reasoning_data(context, topic, num_samples)
            else:
                return []
        except Exception as e:
            print(f"❌ Lỗi khi sinh dữ liệu: {e}")
            return []
    
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
                num_sources = 1  # Word matching có thể từ 1 nguồn
            elif data_type == 'concept_understanding':
                num_sources = 2  # Concept understanding từ 2 nguồn
            elif data_type == 'multi_paragraph_reading':
                num_sources = 2  # Multi-paragraph từ 2-3 nguồn
            elif data_type == 'multi_hop_reasoning':
                num_sources = 3  # Multi-hop từ 3 nguồn
            else:
                num_sources = 2  # Default
                
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
                    instruction = f"Hãy tạo 1 câu hỏi dạng {difficulty_description[data_type]}. Câu hỏi có thể dựa trên thông tin từ một hoặc nhiều nguồn để tăng độ phong phú."
                else:
                    instruction = f"Hãy tạo 1 câu hỏi dạng {difficulty_description[data_type]}. Câu hỏi phải YÊU CẦU THÔNG TIN TỪ NHIỀU ĐIỀU/TÀI LIỆU ĐÃ CHO."
            
            prompt = f"""
            Dựa trên các điều luật sau về chủ đề "{topic}":
            
            {combined_text}
            
            {instruction}
            
            YÊU CÂU QUAN TRỌNG:
            1. Câu hỏi phải ĐỘC LẬP, rõ ràng, có tên luật/văn bản cụ thể
            2. KHÔNG dùng "luật này", "văn bản này", "điều này" - phải nói rõ tên
            3. Câu trả lời phải chính xác từ nội dung các điều luật
            4. {self._get_source_requirement(data_type, len(selected_articles))}
            5. Phải ghi rõ nguồn tham chiếu được sử dụng
            
            Thông tin các nguồn:
            {chr(10).join([f"- Điều {ref.article_number}: {ref.article_title} (từ {ref.document_title})" for ref in source_refs])}
            
            Trả về theo format JSON với sources chứa các nguồn đã sử dụng.
            """
            
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=4000,
                        response_mime_type="application/json",
                        response_schema=LegalQAResponse
                    )
                )
                
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
        """Chọn articles từ các documents khác nhau để đảm bảo diversity"""
        selected = []
        doc_names = list(articles_by_doc.keys())
        
        # Ưu tiên chọn từ các documents khác nhau
        for i in range(min(num_sources, len(doc_names))):
            doc_name = doc_names[i]
            if articles_by_doc[doc_name]:
                # Chọn article tốt nhất từ document này (dài nhất)
                best_article = max(articles_by_doc[doc_name], key=lambda x: len(x['content']))
                selected.append(best_article)
        
        # Nếu cần thêm và còn articles
        while len(selected) < num_sources:
            remaining_articles = []
            for doc_articles in articles_by_doc.values():
                for article in doc_articles:
                    if article not in selected:
                        remaining_articles.append(article)
            
            if not remaining_articles:
                break
                
            # Chọn article dài nhất còn lại
            best_remaining = max(remaining_articles, key=lambda x: len(x['content']))
            selected.append(best_remaining)
        
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
