#!/usr/bin/env python3
"""
BM25-based similarity checker for legal questions
"""

import re
import json
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class QuestionSimilarityChecker:
    """
    Kiểm tra tương đồng câu hỏi sử dụng BM25 và TF-IDF
    """
    
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Args:
            similarity_threshold: Ngưỡng tương đồng (0-1), > threshold = tương đồng
        """
        self.similarity_threshold = similarity_threshold
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.existing_questions = []
        self.preprocessed_questions = []
        
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
        self.preprocessed_questions = []
        
        for item in questions_data:
            # Lấy câu hỏi từ content
            if isinstance(item.get('content'), str):
                content = json.loads(item['content'])
            else:
                content = item.get('content', {})
            
            question = content.get('question', '')
            if question:
                self.existing_questions.append({
                    'id': item.get('id'),
                    'question': question,
                    'data_type': item.get('data_type'),
                    'content': content
                })
                
                # Tiền xử lý câu hỏi
                preprocessed = self.preprocess_text(question)
                self.preprocessed_questions.append(preprocessed)
        
        if self.preprocessed_questions:
            # Khởi tạo BM25
            self.bm25 = BM25Okapi(self.preprocessed_questions)
            
            # Khởi tạo TF-IDF
            question_texts = [' '.join(words) for words in self.preprocessed_questions]
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(question_texts)
            
            print(f"📚 Updated similarity corpus with {len(self.existing_questions)} questions")
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
        if not self.bm25 or not self.existing_questions:
            return []
        
        # Tiền xử lý câu hỏi mới
        query_tokens = self.preprocess_text(new_question)
        if not query_tokens:
            return []
        
        # Tính BM25 scores
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores về 0-1
        if len(bm25_scores) > 0:
            max_bm25 = max(bm25_scores)
            if max_bm25 > 0:
                bm25_scores = bm25_scores / max_bm25
        
        # Tính TF-IDF cosine similarity (đã ở 0-1)
        query_text = ' '.join(query_tokens)
        query_tfidf = self.tfidf_vectorizer.transform([query_text])
        tfidf_scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        
        # Kết hợp scores (trung bình cộng có trọng số)
        # TF-IDF thường stable hơn cho similarity, nên cho trọng số cao hơn
        combined_scores = (0.3 * bm25_scores + 0.7 * tfidf_scores)
        
        # Lấy top_k scores cao nhất
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if combined_scores[idx] > 0.1:  # Ngưỡng tối thiểu
                results.append((
                    self.existing_questions[idx],
                    float(combined_scores[idx])
                ))
        
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

def test_similarity_checker():
    """Test function cho similarity checker"""
    checker = QuestionSimilarityChecker(similarity_threshold=0.7)
    
    # Test data
    existing_questions = [
        {
            'id': 1,
            'data_type': 'word_matching',
            'content': json.dumps({
                'question': 'Độ tuổi tối thiểu để lái xe mô tô là bao nhiêu?',
                'answer': '16 tuổi'
            })
        },
        {
            'id': 2,
            'data_type': 'concept_understanding', 
            'content': json.dumps({
                'question': 'Thời gian đào tạo lý thuyết cho hạng A1 là bao nhiêu giờ?',
                'answer': '18 giờ'
            })
        }
    ]
    
    checker.update_corpus(existing_questions)
    
    # Test new questions
    test_questions = [
        'Tuổi tối thiểu để được phép lái xe mô tô là gì?',  # Tương đồng cao
        'Quy định về học phí đào tạo lái xe là như thế nào?',  # Khác hoàn toàn
        'Thời gian học lý thuyết hạng A1 bao lâu?'  # Tương đồng trung bình
    ]
    
    for question in test_questions:
        is_dup, similar = checker.is_duplicate(question)
        print(f"\n🔍 Question: {question}")
        print(f"   Duplicate: {is_dup}")
        for sim_q, score in similar:
            print(f"   Similar ({score:.3f}): {sim_q['question']}")

if __name__ == "__main__":
    test_similarity_checker()
