#!/usr/bin/env python3
"""
Coverage Analyzer - Đo độ bao phủ của bộ câu hỏi đối với văn bản gốc
"""

import re
import json
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class CoverageAnalyzer:
    """
    Phân tích độ bao phủ của bộ câu hỏi đối với văn bản pháp luật
    """
    
    def __init__(self, coverage_threshold: float = 0.3):
        """
        Args:
            coverage_threshold: Ngưỡng để xác định unit được bao phủ (0-1)
        """
        self.coverage_threshold = coverage_threshold
        self.text_units = []  # Các đơn vị văn bản (câu/đoạn)
        self.questions = []   # Các câu hỏi đã sinh
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.questions_tfidf = None
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Tiền xử lý văn bản tiếng Việt (tương tự similarity_checker)
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
    
    def split_into_units(self, text: str, unit_type: str = 'sentence') -> List[Dict[str, Any]]:
        """
        Chia văn bản thành các đơn vị (units)
        
        Args:
            text: Văn bản cần chia
            unit_type: Loại đơn vị ('sentence', 'paragraph')
            
        Returns:
            List[Dict]: Danh sách các units với metadata
        """
        units = []
        
        if unit_type == 'sentence':
            # Chia thành câu (điều)
            sentences = re.split(r'[.!?]+', text)
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) > 20:  # Lọc câu quá ngắn
                    units.append({
                        'id': f'sent_{i}',
                        'type': 'sentence',
                        'content': sentence,
                        'length': len(sentence),
                        'tokens': self.preprocess_text(sentence)
                    })
                    
        elif unit_type == 'paragraph':
            # Chia thành đoạn
            paragraphs = text.split('\n\n')
            for i, paragraph in enumerate(paragraphs):
                paragraph = paragraph.strip()
                if len(paragraph) > 50:  # Lọc đoạn quá ngắn
                    units.append({
                        'id': f'para_{i}',
                        'type': 'paragraph', 
                        'content': paragraph,
                        'length': len(paragraph),
                        'tokens': self.preprocess_text(paragraph)
                    })
        
        return units
    
    def prepare_coverage_analysis(self, documents: List[Dict[str, Any]], questions_data: List[Dict[str, Any]], unit_type: str = 'sentence'):
        """
        Chuẩn bị dữ liệu cho phân tích coverage
        
        Args:
            documents: List các documents với content
            questions_data: List các câu hỏi đã sinh
            unit_type: Loại đơn vị để phân tích
        """
        # Bước 1: Chia văn bản thành units
        self.text_units = []
        for doc in documents:
            doc_units = self.split_into_units(doc['content'], unit_type)
            
            # Thêm thông tin document vào mỗi unit
            for unit in doc_units:
                unit['document_id'] = doc.get('id')
                unit['document_title'] = doc.get('title', 'Unknown')
                self.text_units.append(unit)
        
        print(f"📄 Chia thành {len(self.text_units)} {unit_type}s từ {len(documents)} documents")
        
        # Bước 2: Chuẩn bị câu hỏi
        self.questions = []
        for item in questions_data:
            if isinstance(item.get('content'), str):
                content = json.loads(item['content'])
            else:
                content = item.get('content', {})
            
            question = content.get('question', '')
            if question:
                self.questions.append({
                    'id': item.get('id'),
                    'question': question,
                    'data_type': item.get('data_type'),
                    'tokens': self.preprocess_text(question)
                })
        
        print(f"❓ Chuẩn bị {len(self.questions)} câu hỏi")
        
        # Bước 3: Khởi tạo BM25 và TF-IDF cho units
        unit_texts = [' '.join(unit['tokens']) for unit in self.text_units]
        
        if unit_texts:
            # BM25 cho units
            unit_token_lists = [unit['tokens'] for unit in self.text_units]
            self.bm25 = BM25Okapi(unit_token_lists)
            
            # TF-IDF cho questions
            question_texts = [' '.join(q['tokens']) for q in self.questions]
            if question_texts:
                self.tfidf_vectorizer = TfidfVectorizer()
                # Fit trên cả units và questions
                all_texts = unit_texts + question_texts
                self.tfidf_vectorizer.fit(all_texts)
                
                # Transform questions
                self.questions_tfidf = self.tfidf_vectorizer.transform(question_texts)
                
                print("🔍 Khởi tạo BM25 và TF-IDF hoàn thành")
            else:
                print("⚠️ Không có câu hỏi để phân tích")
        else:
            print("⚠️ Không có units để phân tích")
    
    def calculate_unit_coverage(self, unit: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tính độ bao phủ cho một unit cụ thể
        
        Args:
            unit: Unit cần tính coverage
            
        Returns:
            Dict: Thông tin coverage của unit
        """
        if not self.questions or not self.bm25:
            return {
                'is_covered': False,
                'max_similarity': 0.0,
                'best_question': None,
                'similarities': []
            }
        
        unit_tokens = unit['tokens']
        similarities = []
        
        # Tính similarity với tất cả câu hỏi
        for i, question in enumerate(self.questions):
            # BM25 score - truyền query dạng list của tokens
            bm25_score = 0.0
            if len(unit_tokens) > 0:
                bm25_scores = self.bm25.get_scores(unit_tokens)
                # Lấy max score từ tất cả documents
                bm25_score = max(bm25_scores) if len(bm25_scores) > 0 else 0.0
            
            # Normalize BM25 score
            if bm25_score > 0:
                bm25_score = min(bm25_score / 10.0, 1.0)  # Simple normalization
            
            # TF-IDF cosine similarity
            tfidf_score = 0.0
            if self.tfidf_vectorizer and self.questions_tfidf is not None:
                try:
                    unit_text = ' '.join(unit_tokens)
                    unit_tfidf = self.tfidf_vectorizer.transform([unit_text])
                    question_tfidf = self.questions_tfidf[i:i+1]
                    tfidf_score = cosine_similarity(unit_tfidf, question_tfidf)[0][0]
                except:
                    tfidf_score = 0.0
            
            # Kết hợp scores
            combined_score = 0.3 * bm25_score + 0.7 * tfidf_score
            
            similarities.append({
                'question_id': question['id'],
                'question': question['question'],
                'data_type': question['data_type'],
                'bm25_score': float(bm25_score),
                'tfidf_score': float(tfidf_score),
                'combined_score': float(combined_score)
            })
        
        # Tìm similarity cao nhất
        max_similarity = max([s['combined_score'] for s in similarities], default=0.0)
        best_question = max(similarities, key=lambda x: x['combined_score'], default=None)
        
        is_covered = max_similarity >= self.coverage_threshold
        
        return {
            'is_covered': is_covered,
            'max_similarity': max_similarity,
            'best_question': best_question,
            'similarities': sorted(similarities, key=lambda x: x['combined_score'], reverse=True)[:3]  # Top 3
        }
    
    def analyze_coverage(self) -> Dict[str, Any]:
        """
        Phân tích coverage cho tất cả units
        
        Returns:
            Dict: Kết quả phân tích coverage
        """
        if not self.text_units:
            return {
                'total_units': 0,
                'covered_units': 0,
                'coverage_percentage': 0.0,
                'units_analysis': []
            }
        
        print("🔍 Bắt đầu phân tích coverage...")
        
        covered_count = 0
        units_analysis = []
        
        for i, unit in enumerate(self.text_units):
            coverage_info = self.calculate_unit_coverage(unit)
            
            if coverage_info['is_covered']:
                covered_count += 1
            
            unit_analysis = {
                'unit_id': unit['id'],
                'unit_type': unit['type'],
                'document_title': unit['document_title'],
                'content_preview': unit['content'][:100] + '...' if len(unit['content']) > 100 else unit['content'],
                'length': unit['length'],
                **coverage_info
            }
            
            units_analysis.append(unit_analysis)
            
            if (i + 1) % 10 == 0:
                print(f"  📊 Đã phân tích {i + 1}/{len(self.text_units)} units...")
        
        coverage_percentage = (covered_count / len(self.text_units)) * 100
        
        result = {
            'total_units': len(self.text_units),
            'covered_units': covered_count,
            'uncovered_units': len(self.text_units) - covered_count,
            'coverage_percentage': coverage_percentage,
            'threshold_used': self.coverage_threshold,
            'units_analysis': units_analysis
        }
        
        print(f"✅ Hoàn thành phân tích coverage: {coverage_percentage:.1f}% ({covered_count}/{len(self.text_units)} units)")
        
        return result
    
    def get_coverage_summary_by_document(self, coverage_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tóm tắt coverage theo từng document
        """
        doc_stats = {}
        
        for unit in coverage_result['units_analysis']:
            doc_title = unit['document_title']
            
            if doc_title not in doc_stats:
                doc_stats[doc_title] = {
                    'total_units': 0,
                    'covered_units': 0,
                    'uncovered_units': 0,
                    'coverage_percentage': 0.0
                }
            
            doc_stats[doc_title]['total_units'] += 1
            if unit['is_covered']:
                doc_stats[doc_title]['covered_units'] += 1
            else:
                doc_stats[doc_title]['uncovered_units'] += 1
        
        # Tính percentage cho từng doc
        for doc_title, stats in doc_stats.items():
            if stats['total_units'] > 0:
                stats['coverage_percentage'] = (stats['covered_units'] / stats['total_units']) * 100
        
        return doc_stats

def test_coverage_analyzer():
    """Test function cho coverage analyzer"""
    
    # Mock data
    documents = [
        {
            'id': 1,
            'title': 'Luật test',
            'content': '''Điều 1. Quy định chung về giao thông.
            Giao thông đường bộ phải tuân theo luật pháp.
            
            Điều 2. Về độ tuổi lái xe.
            Người lái xe phải đủ 18 tuổi trở lên.
            Đối với xe mô tô thì từ 16 tuổi.'''
        }
    ]
    
    questions_data = [
        {
            'id': 1,
            'data_type': 'word_matching',
            'content': json.dumps({
                'question': 'Độ tuổi tối thiểu để lái xe ô tô là bao nhiêu?',
                'answer': '18 tuổi'
            })
        },
        {
            'id': 2,
            'data_type': 'concept_understanding',
            'content': json.dumps({
                'question': 'Giao thông đường bộ cần tuân theo quy định gì?',
                'answer': 'Tuân theo luật pháp'
            })
        }
    ]
    
    # Test coverage
    analyzer = CoverageAnalyzer(coverage_threshold=0.3)
    analyzer.prepare_coverage_analysis(documents, questions_data, unit_type='sentence')
    
    result = analyzer.analyze_coverage()
    
    print(f"\n📊 COVERAGE ANALYSIS RESULT:")
    print(f"Total units: {result['total_units']}")
    print(f"Covered units: {result['covered_units']}")
    print(f"Coverage: {result['coverage_percentage']:.1f}%")
    
    print(f"\n📋 DETAILED ANALYSIS:")
    for unit in result['units_analysis'][:3]:  # Show first 3
        print(f"Unit: {unit['content_preview']}")
        print(f"  Covered: {unit['is_covered']} (similarity: {unit['max_similarity']:.3f})")
        if unit['best_question']:
            print(f"  Best match: {unit['best_question']['question'][:50]}...")

if __name__ == "__main__":
    test_coverage_analyzer()
