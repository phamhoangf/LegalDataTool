#!/usr/bin/env python3
"""
Hybrid similarity checker for legal questions
Uses semantic search + BM25 for optimal accuracy
"""

import re
import json
from typing import List, Dict, Any, Tuple, Optional
from hybrid_search import HybridSearchEngine, create_hybrid_search_engine

class QuestionSimilarityChecker:
    """
    Kiểm tra tương đồng câu hỏi sử dụng Hybrid Search (BM25 + Semantic)
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.75,
                 hybrid_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            similarity_threshold: Ngưỡng tương đồng (0-1), > threshold = tương đồng
            hybrid_config: Configuration for hybrid search engine
        """
        self.similarity_threshold = similarity_threshold
        
        print(f"🧠 Initializing hybrid similarity checker with threshold {similarity_threshold}")
        self.hybrid_engine = create_hybrid_search_engine(hybrid_config)
        
        self.existing_questions = []
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Tiền xử lý văn bản tiếng Việt
        """
        if not text:
            return []
            
        # Chuyển thường
        text = text.lower()
        
        # Loại bỏ ký tự đặc biệt, giữ lại chữ cái, số và khoảng trắng
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tách từ đơn giản (split by space)
        words = text.split()
        
        # Loại bỏ từ dừng đơn giản
        stop_words = {
            'là', 'của', 'và', 'có', 'trong', 'với', 'theo', 'để', 'được', 'các', 
            'một', 'này', 'đó', 'những', 'từ', 'trên', 'dưới', 'về', 'cho', 'hay',
            'bằng', 'như', 'khi', 'nào', 'gì', 'ai', 'đâu', 'sao', 'bao', 'nhiều'
        }
        
        words = [word for word in words if word not in stop_words and len(word) > 1]
        
        return words
    
    def update_corpus(self, questions_data: List[Dict[str, Any]]):
        """
        Cập nhật corpus với các câu hỏi hiện có trong database
        
        Args:
            questions_data: List các dict chứa câu hỏi và metadata
        """
        self.existing_questions = []
        question_texts = []
        question_ids = []
        
        for item in questions_data:
            # Lấy câu hỏi từ content
            if isinstance(item.get('content'), str):
                content = json.loads(item['content'])
            else:
                content = item.get('content', {})
            
            question = content.get('question', '')
            if question:
                question_obj = {
                    'id': item.get('id'),
                    'question': question,
                    'data_type': item.get('data_type'),
                    'content': content
                }
                self.existing_questions.append(question_obj)
                question_texts.append(question)
                question_ids.append(item.get('id'))
        
        if self.existing_questions:
            # Use hybrid search engine
            print(f"🧠 Building hybrid search index with {len(self.existing_questions)} questions")
            self.hybrid_engine.index_documents(question_texts, question_ids)
            print(f"✅ Updated similarity corpus with {len(self.existing_questions)} questions")
        else:
            print("📚 No existing questions found - similarity checking disabled")
    
    def find_similar_questions(self, new_question: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Tìm các câu hỏi tương đồng với câu hỏi mới
        
        Args:
            new_question: Câu hỏi mới cần kiểm tra
            top_k: Số lượng câu hỏi tương đồng nhất trả về
            
        Returns:
            List[Tuple[Dict, float]]: (question_data, similarity_score)
        """
        if not self.existing_questions:
            return []
        
        if not new_question or not new_question.strip():
            return []
        
        results = []
        
        # Use hybrid search
        search_results = self.hybrid_engine.search(
            new_question, 
            top_k=top_k, 
            return_scores=True,
            adaptive_weighting=True
        )
        
        for result in search_results:
            # Find question object by ID
            question_obj = None
            for q in self.existing_questions:
                if q['id'] == result['doc_id']:
                    question_obj = q
                    break
            
            if question_obj:
                results.append((question_obj, result['combined_score']))
        
        return results
    
    def is_duplicate(self, new_question: str) -> Tuple[bool, List[Tuple[Dict, float]]]:
        """
        Kiểm tra xem câu hỏi mới có trùng lặp với câu hỏi hiện có không
        
        Args:
            new_question: Câu hỏi mới cần kiểm tra
            
        Returns:
            Tuple[bool, List]: (is_duplicate, similar_questions)
        """
        similar_questions = self.find_similar_questions(new_question, top_k=3)
        
        # Kiểm tra có câu hỏi nào vượt ngưỡng không
        is_duplicate = any(score >= self.similarity_threshold for _, score in similar_questions)
        
        return is_duplicate, similar_questions
    
    def check_batch_similarity(self, new_questions: List[str]) -> List[Dict[str, Any]]:
        """
        Kiểm tra tương đồng cho một batch câu hỏi
        
        Args:
            new_questions: List các câu hỏi mới
            
        Returns:
            List[Dict]: Kết quả kiểm tra cho từng câu hỏi
        """
        results = []
        
        for i, question in enumerate(new_questions):
            is_dup, similar = self.is_duplicate(question)
            
            result = {
                'index': i,
                'question': question,
                'is_duplicate': is_dup,
                'similar_questions': [
                    {
                        'question': sim_q['question'],
                        'similarity': score,
                        'data_type': sim_q['data_type'],
                        'id': sim_q['id']
                    }
                    for sim_q, score in similar
                ],
                'max_similarity': max([score for _, score in similar], default=0.0)
            }
            results.append(result)
        
        return results
