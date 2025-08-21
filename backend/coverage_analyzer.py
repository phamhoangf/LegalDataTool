#!/usr/bin/env python3
"""
Coverage Analyzer - Đo độ bao phủ của bộ câu hỏi đối với văn bản gốc
Uses hybrid search for optimal accuracy
"""

import re
import json
from typing import List, Dict, Any, Tuple, Optional
from hybrid_search import HybridSearchEngine, create_hybrid_search_engine

class CoverageAnalyzer:
    """
    Phân tích độ bao phủ của bộ câu hỏi đối với văn bản pháp luật
    Sử dụng Hybrid Search (BM25 + Semantic)
    """
    
    def __init__(self, 
                 coverage_threshold: float = 0.3,
                 hybrid_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            coverage_threshold: Ngưỡng để xác định unit được bao phủ (0-1)
            hybrid_config: Configuration for hybrid search engine
        """
        self.coverage_threshold = coverage_threshold
        self.should_stop = False  # Flag để dừng phân tích
        
        print(f"🧠 Initializing hybrid coverage analyzer with threshold {coverage_threshold}")
        self.hybrid_engine = create_hybrid_search_engine(hybrid_config)
        
        self.text_units = []  # Các đơn vị văn bản (câu/đoạn)
        self.questions = []   # Các câu hỏi đã sinh
        
    def stop_analysis(self):
        """Dừng quá trình phân tích"""
        self.should_stop = True
        print("🛑 Đã yêu cầu dừng phân tích coverage")
        
    def reset_stop_flag(self):
        """Reset flag dừng để chuẩn bị cho phân tích mới"""
        self.should_stop = False
        
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
            # Chia thành các điều luật theo pattern "Điều X."
            # Sử dụng split approach đơn giản và hiệu quả
            parts = re.split(r'(?=Điều\s+\d+\.)', text, flags=re.IGNORECASE)
            
            article_count = 0
            for i, part in enumerate(parts):
                part = part.strip()
                if len(part) < 30:  # Loại bỏ phần quá ngắn (header, footer)
                    continue
                    
                # Tìm số điều và tên điều
                article_match = re.search(r'Điều\s+(\d+)\.\s*([^\r\n]*)', part, re.IGNORECASE)
                if article_match:
                    article_num = article_match.group(1)
                    article_title = article_match.group(2).strip()
                    article_id = f'dieu_{article_num}'
                    type_label = 'article'
                    article_count += 1
                else:
                    # Phần không có "Điều X." (có thể là phần đầu văn bản)
                    article_id = f'part_{i}'
                    article_title = part[:50].replace('\n', ' ').strip()
                    type_label = 'section'
                
                units.append({
                    'id': article_id,
                    'type': type_label,
                    'content': part,
                    'length': len(part),
                    'tokens': self.preprocess_text(part),
                    'article_number': int(article_match.group(1)) if article_match else None,
                    'article_title': article_title
                })
            
            print(f"📋 Đã chia thành {len(units)} units, trong đó {article_count} điều luật")
                    
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
            questions_data: List các câu hỏi đã sinh với source information
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
        
        # Bước 2: Chuẩn bị câu hỏi với source information
        self.questions = []
        for item in questions_data:
            if isinstance(item.get('content'), str):
                content = json.loads(item['content'])
            else:
                content = item.get('content', {})
            
            question = content.get('question', '')
            sources = content.get('sources', [])  # Extract source information
            
            if question:
                self.questions.append({
                    'id': item.get('id'),
                    'question': question,
                    'data_type': item.get('data_type'),
                    'sources': sources,  # Add source information
                    'tokens': self.preprocess_text(question)
                })
        
        print(f"❓ Chuẩn bị {len(self.questions)} câu hỏi")
        
        # Bước 3: Khởi tạo hybrid search engine
        unit_texts = [' '.join(unit['tokens']) for unit in self.text_units]
        
        if unit_texts and self.questions:
            # Use hybrid search for question-unit matching
            print("🧠 Initializing hybrid search for coverage analysis...")
            self.hybrid_engine.index_documents(unit_texts, list(range(len(self.text_units))))
            print("✅ Hybrid search ready for coverage analysis")
        else:
            print("⚠️ Không có units hoặc questions để phân tích")
    
    def calculate_unit_coverage(self, unit: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tính độ bao phủ cho một unit cụ thể, chỉ tính với questions có sources liên quan
        Sử dụng hybrid search cho accuracy tốt hơn
        
        Args:
            unit: Unit cần tính coverage
            
        Returns:
            Dict: Thông tin coverage của unit
        """
        base_result = {
            'is_covered': False,
            'max_similarity': 0.0,
            'best_question': None,
            'similarities': [],
            'relevant_questions_count': 0
        }
        
        if not self.questions:
            return base_result
        
        unit_tokens = unit['tokens']
        unit_text = ' '.join(unit_tokens)
        unit_doc_title = unit.get('document_title', 'Unknown')
        
        # Filter questions that reference this unit's document as source
        relevant_questions = []
        for i, question in enumerate(self.questions):
            # Check if this unit's document is in the question's sources
            for source in question.get('sources', []):
                # Try different possible field names for document title
                source_title = (source.get('document_title') or 
                              source.get('title') or 
                              source.get('name') or 
                              source.get('article_title', ''))
                
                if source_title and source_title == unit_doc_title:
                    relevant_questions.append((i, question))
                    break
        
        # If no relevant questions, return no coverage
        if not relevant_questions:
            return base_result
        
        similarities = []
        
        # Use hybrid search to compute similarity between unit and relevant questions
        for i, question in relevant_questions:
            try:
                # Compute similarity between unit text and question
                similarity_result = self.hybrid_engine.compute_similarity(unit_text, question['question'])
                combined_score = similarity_result['combined_score']
                
                similarities.append({
                    'question_id': question['id'],
                    'question': question['question'],
                    'data_type': question['data_type'],
                    'bm25_score': similarity_result['bm25_score'],
                    'semantic_score': similarity_result['semantic_score'],
                    'tfidf_score': similarity_result['tfidf_score'],
                    'combined_score': combined_score
                })
            except Exception as e:
                print(f"⚠️ Error computing hybrid similarity: {e}")
                # Fallback to zero score
                similarities.append({
                    'question_id': question['id'],
                    'question': question['question'],
                    'data_type': question['data_type'],
                    'bm25_score': 0.0,
                    'semantic_score': 0.0,
                    'tfidf_score': 0.0,
                    'combined_score': 0.0
                })
        
        # Tìm similarity cao nhất
        max_similarity = max([s['combined_score'] for s in similarities], default=0.0)
        best_question = max(similarities, key=lambda x: x['combined_score'], default=None)
        
        is_covered = max_similarity >= self.coverage_threshold
        
        return {
            'is_covered': is_covered,
            'max_similarity': max_similarity,
            'best_question': best_question,
            'similarities': sorted(similarities, key=lambda x: x['combined_score'], reverse=True)[:3],  # Top 3
            'relevant_questions_count': len(relevant_questions)
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
        
        print("🧠 Bắt đầu phân tích coverage với Hybrid Search (optimized - chỉ tính với relevant questions)...")
        self.reset_stop_flag()  # Reset flag khi bắt đầu
        
        covered_count = 0
        units_analysis = []
        total_relevant_questions = 0
        
        for i, unit in enumerate(self.text_units):
            # Kiểm tra nếu được yêu cầu dừng
            if self.should_stop:
                print(f"🛑 Phân tích bị dừng tại unit {i + 1}/{len(self.text_units)}")
                break
                
            coverage_info = self.calculate_unit_coverage(unit)
            
            if coverage_info['is_covered']:
                covered_count += 1
            
            # Track total relevant questions count
            total_relevant_questions += coverage_info.get('relevant_questions_count', 0)
            
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
                print(f"  📊 Đã phân tích {i + 1}/{len(self.text_units)} units... (relevant questions so far: {total_relevant_questions})")
        
        # Tính coverage dựa trên số unit đã xử lý
        processed_units = len(units_analysis)
        coverage_percentage = (covered_count / processed_units) * 100 if processed_units > 0 else 0
        
        result = {
            'total_units': len(self.text_units),
            'processed_units': processed_units,
            'covered_units': covered_count,
            'uncovered_units': processed_units - covered_count,
            'coverage_percentage': coverage_percentage,
            'threshold_used': self.coverage_threshold,
            'was_stopped': self.should_stop,
            'total_questions': len(self.questions),
            'total_relevant_calculations': total_relevant_questions,
            'optimization_ratio': f"{total_relevant_questions}/{len(self.text_units) * len(self.questions)} ({(total_relevant_questions / (len(self.text_units) * len(self.questions)) * 100):.1f}%)",
            'units_analysis': units_analysis
        }
        
        status_message = "🛑 Đã dừng" if self.should_stop else "✅ Hoàn thành"
        print(f"{status_message} phân tích coverage: {coverage_percentage:.1f}% ({covered_count}/{processed_units} units đã xử lý)")
        
        if processed_units > 0:
            print(f"🚀 Optimization: Tính {total_relevant_questions} similarities thay vì {processed_units * len(self.questions)} (tiết kiệm {((processed_units * len(self.questions) - total_relevant_questions) / (processed_units * len(self.questions)) * 100):.1f}%)")
        
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

        return doc_stats
