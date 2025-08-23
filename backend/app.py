
# ----------- CRAWL LAW DOCUMENT API -----------
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os
import json
import jsonlines
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime
from data_generator import DataGenerator
from coverage_analyzer import CoverageAnalyzer
from file_handler import process_file, validate_file_size, get_supported_formats
from models import db, LegalTopic, LegalDocument, TopicDocument, GeneratedData, LabeledData

# Load .env file từ parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///legal_data.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
CORS(app)
db.init_app(app)

# Initialize data generator with Google API key and similarity threshold
google_api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
data_generator = DataGenerator(api_key=google_api_key, similarity_threshold=0.75)


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/topics', methods=['GET'])
def get_topics():
    """Lấy danh sách chủ đề pháp lý với documents"""
    topics = LegalTopic.query.all()
    result = []
    
    for topic in topics:
        # Lấy tất cả documents liên quan
        documents = db.session.query(LegalDocument).join(TopicDocument).filter(
            TopicDocument.topic_id == topic.id
        ).all()
        
        # Aggregate legal text từ tất cả documents
        legal_text = '\n\n'.join([doc.content for doc in documents])
        
        result.append({
            'id': topic.id,
            'name': topic.name,
            'description': topic.description,
            'legal_text': legal_text,  # Để tương thích với frontend
            'document_count': len(documents),
            'documents': [{'id': doc.id, 'title': doc.title} for doc in documents],
            'created_at': topic.created_at.isoformat()
        })
    
    return jsonify(result)

@app.route('/api/topics', methods=['POST'])
def create_topic():
    """Tạo chủ đề pháp lý mới"""
    data = request.get_json()
    
    topic = LegalTopic(
        name=data['name'],
        description=data.get('description', '')
    )
    
    db.session.add(topic)
    db.session.commit()
    
    return jsonify({
        'id': topic.id,
        'name': topic.name,
        'description': topic.description,
        'document_count': 0,
        'documents': [],
        'message': 'Chủ đề đã được tạo thành công'
    }), 201

@app.route('/api/topics/<int:topic_id>', methods=['PUT'])
def update_topic(topic_id):
    """Cập nhật chủ đề pháp lý"""
    topic = LegalTopic.query.get_or_404(topic_id)
    data = request.get_json()
    
    topic.name = data.get('name', topic.name)
    topic.description = data.get('description', topic.description)
    topic.legal_text = data.get('legal_text', topic.legal_text)
    
    db.session.commit()
    
    return jsonify({
        'id': topic.id,
        'name': topic.name,
        'description': topic.description,
        'legal_text': topic.legal_text,
        'message': 'Chủ đề đã được cập nhật thành công'
    })

@app.route('/api/topics/<int:topic_id>', methods=['DELETE'])
def delete_topic(topic_id):
    """Xóa chủ đề pháp lý"""
    topic = LegalTopic.query.get_or_404(topic_id)
    
    # Xóa tất cả dữ liệu liên quan
    TopicDocument.query.filter_by(topic_id=topic_id).delete()
    GeneratedData.query.filter_by(topic_id=topic_id).delete()
    
    db.session.delete(topic)
    db.session.commit()
    
    return jsonify({
        'message': 'Chủ đề đã được xóa thành công'
    })

    # ----------- ASYNC CRAWL UTILS -----------
TIMEOUT_SECONDS = 20
MAX_CONCURRENT_REQUESTS = 15

async def fetch_and_parse(session, url, semaphore):
    """Tải và phân tích một trang văn bản duy nhất với nhiều selector fallback."""
    async with semaphore:
        try:
            async with session.get(url, timeout=TIMEOUT_SECONDS) as response:
                response.raise_for_status()
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                # Fallback selector logic
                selectors = [
                    ('div', {'itemprop': 'articleBody'}),
                    ('div', {'id': 'content'}),
                    ('article', {}),
                    ('div', {'class': 'main-content'}),
                    ('div', {'class': 'content'}),
                ]
                for tag, attrs in selectors:
                    found = soup.find(tag, attrs=attrs)
                    if found and found.get_text(strip=True):
                        return found.get_text(separator='\n', strip=True), None
                # Nếu không tìm thấy, lấy toàn bộ body
                body = soup.find('body')
                if body and body.get_text(strip=True):
                    return body.get_text(separator='\n', strip=True), "Đã lấy toàn bộ <body> do không tìm thấy selector đặc biệt."
                return None, "Không tìm thấy nội dung phù hợp trên trang (đã thử nhiều selector)."
        except Exception as e:
            return None, f"Lỗi khi crawl: {str(e)}"
@app.route('/api/crawl-law-document', methods=['POST'])
def crawl_law_document():
    """Crawl nội dung văn bản luật từ web, lưu thành document và liên kết với topic nếu có."""
    data = request.get_json()
    url = data.get('url')
    topic_id = data.get('topic_id')
    title = data.get('title')
    if not url:
        return jsonify({'error': 'Thiếu URL'}), 400
    # Crawl nội dung (async)
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        content, crawl_note = loop.run_until_complete(fetch_law_content(url))
    except Exception as e:
        return jsonify({'error': f'Lỗi khi crawl: {str(e)}'}), 500
    if not content:
        return jsonify({'error': crawl_note or 'Không lấy được nội dung từ trang web'}), 400
    # Lưu document
    doc_title = title or url
    document = LegalDocument(
        title=doc_title,
        content=content,
        document_type='law',
        uploaded_by='crawler'
    )
    db.session.add(document)
    db.session.flush()  # Lấy document.id
    # Liên kết với topic nếu có
    if topic_id:
        topic = LegalTopic.query.get(topic_id)
        if topic:
            topic_doc = TopicDocument(
                topic_id=int(topic_id),
                document_id=document.id,
                relevance_score=1.0
            )
            db.session.add(topic_doc)
    db.session.commit()
    return jsonify({
        'message': 'Đã crawl và lưu tài liệu thành công',
        'document_id': document.id,
        'title': document.title,
        'note': crawl_note
    })

async def fetch_law_content(url):
    """Hàm chính để crawl 1 URL, trả về (content, error)"""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        content, error = await fetch_and_parse(session, url, semaphore)
        return content, error
    

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Lấy danh sách tài liệu"""
    documents = LegalDocument.query.all()
    result = []
    
    for doc in documents:
        # Lấy các topic liên kết
        topic_docs = TopicDocument.query.filter_by(document_id=doc.id).all()
        topics = []
        for td in topic_docs:
            topic = LegalTopic.query.get(td.topic_id)
            if topic:
                topics.append({
                    'id': topic.id,
                    'name': topic.name
                })
        
        result.append({
            'id': doc.id,
            'title': doc.title,
            'content': doc.content,
            'document_type': doc.document_type,
            'document_number': doc.document_number,
            'uploaded_at': doc.uploaded_at.isoformat(),
            'created_at': doc.uploaded_at.isoformat(),
            'topics': topics
        })
    
    return jsonify(result)

@app.route('/api/documents', methods=['POST'])
def create_document():
    """Tạo tài liệu mới"""
    data = request.get_json()
    
    document = LegalDocument(
        title=data['title'],
        content=data['content'],
        document_type=data.get('document_type', 'law'),
        document_number=data.get('document_number', ''),
        uploaded_by=data.get('uploaded_by', 'system')
    )
    
    db.session.add(document)
    db.session.commit()
    
    return jsonify({
        'id': document.id,
        'title': document.title,
        'message': 'Tài liệu đã được tạo thành công'
    }), 201

@app.route('/api/topics/<int:topic_id>/documents/<int:document_id>', methods=['POST'])
def link_document_to_topic(topic_id, document_id):
    """Liên kết tài liệu với chủ đề"""
    topic = LegalTopic.query.get_or_404(topic_id)
    document = LegalDocument.query.get_or_404(document_id)
    
    # Kiểm tra đã liên kết chưa
    existing = TopicDocument.query.filter_by(
        topic_id=topic_id, 
        document_id=document_id
    ).first()
    
    if existing:
        return jsonify({'message': 'Tài liệu đã được liên kết với chủ đề này'}), 400
    
    topic_doc = TopicDocument(
        topic_id=topic_id,
        document_id=document_id,
        relevance_score=request.get_json().get('relevance_score', 1.0)
    )
    
    db.session.add(topic_doc)
    db.session.commit()
    
    return jsonify({
        'message': 'Đã liên kết tài liệu với chủ đề thành công'
    }), 201

@app.route('/api/documents/<int:document_id>', methods=['PUT'])
def update_document(document_id):
    """Cập nhật tài liệu"""
    document = LegalDocument.query.get_or_404(document_id)
    data = request.get_json()
    
    if 'title' in data:
        document.title = data['title']
    if 'content' in data:
        document.content = data['content']
    if 'document_type' in data:
        document.document_type = data['document_type']
    
    document.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({
        'id': document.id,
        'title': document.title,
        'message': 'Tài liệu đã được cập nhật thành công'
    })

@app.route('/api/documents/<int:document_id>', methods=['DELETE'])
def delete_document(document_id):
    """Xóa tài liệu"""
    document = LegalDocument.query.get_or_404(document_id)
    
    # Xóa các liên kết với chủ đề trước
    TopicDocument.query.filter_by(document_id=document_id).delete()
    
    # Xóa tài liệu
    db.session.delete(document)
    db.session.commit()
    
    return jsonify({
        'message': 'Đã xóa tài liệu thành công'
    })

@app.route('/api/topics/<int:topic_id>/documents/<int:document_id>', methods=['DELETE'])
def unlink_document_from_topic(topic_id, document_id):
    """Hủy liên kết tài liệu với chủ đề"""
    topic_doc = TopicDocument.query.filter_by(
        topic_id=topic_id, 
        document_id=document_id
    ).first_or_404()
    
    db.session.delete(topic_doc)
    db.session.commit()
    
    return jsonify({
        'message': 'Đã hủy liên kết tài liệu với chủ đề'
    })

@app.route('/api/documents/upload', methods=['POST'])
def upload_document_file():
    """Upload file tài liệu mà không cần liên kết với chủ đề"""
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được tải lên'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tên file không hợp lệ'}), 400

    title = request.form.get('title', file.filename)
    document_type = request.form.get('document_type', 'law')

    # Xử lý file đa định dạng
    file_content = file.read()
    result = process_file(file_content, file.filename)
    if not result.get('success'):
        return jsonify({'error': result.get('error', 'Không thể xử lý file')}), 400

    content = result.get('content', '')

    # Tạo document
    document = LegalDocument(
        title=title,
        content=content,
        document_type=document_type
    )

    db.session.add(document)
    db.session.commit()

    return jsonify({
        'id': document.id,
        'title': document.title,
        'message': 'Tài liệu đã được tải lên thành công',
        'file_type': result.get('file_type'),
        'metadata': result.get('metadata', {})
    }), 201

@app.route('/api/upload', methods=['POST'])
def upload_legal_document():
    """Tải lên văn bản luật và liên kết với chủ đề"""
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được tải lên'}), 400

    file = request.files['file']
    topic_id = request.form.get('topic_id')
    document_title = request.form.get('title', file.filename)
    document_type = request.form.get('document_type', 'law')

    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400

    # Xử lý file đa định dạng
    file_content = file.read()
    result = process_file(file_content, file.filename)
    if not result.get('success'):
        return jsonify({'error': result.get('error', 'Không thể xử lý file')}), 400

    content = result.get('content', '')

    # Tạo document mới
    document = LegalDocument(
        title=document_title,
        content=content,
        document_type=document_type,
        uploaded_by='user'
    )

    db.session.add(document)
    db.session.flush()  # Để có ID

    # Liên kết với topic nếu có
    if topic_id:
        topic = LegalTopic.query.get(topic_id)
        if topic:
            topic_doc = TopicDocument(
                topic_id=int(topic_id),
                document_id=document.id,
                relevance_score=1.0
            )
            db.session.add(topic_doc)

    db.session.commit()

    return jsonify({
        'message': 'File đã được tải lên và liên kết thành công',
        'document_id': document.id,
        'file_type': result.get('file_type'),
        'metadata': result.get('metadata', {})
    })

@app.route('/api/generate', methods=['POST'])
def generate_training_data():
    """Sinh dữ liệu huấn luyện với 4 loại reasoning"""
    data = request.get_json()
    
    topic_id = data.get('topic_id')
    data_type = data.get('data_type')
    num_samples = data.get('num_samples', 10)
    
    # Validate data_type - chỉ chấp nhận 4 loại mới
    valid_types = ['word_matching', 'concept_understanding', 'multi_paragraph_reading', 'multi_hop_reasoning']
    if not topic_id or not data_type:
        return jsonify({'error': 'Thiếu topic_id hoặc data_type'}), 400
    
    if data_type not in valid_types:
        return jsonify({
            'error': f'data_type không hợp lệ. Chỉ chấp nhận: {", ".join(valid_types)}'
        }), 400
    
    topic = LegalTopic.query.get(topic_id)
    if not topic:
        return jsonify({'error': 'Không tìm thấy chủ đề'}), 404
    
    # Lấy tất cả documents liên quan đến topic
    documents = db.session.query(LegalDocument).join(TopicDocument).filter(
        TopicDocument.topic_id == topic_id
    ).all()
    
    if not documents:
        return jsonify({'error': 'Chủ đề chưa có tài liệu pháp luật nào'}), 400
    
    try:
        # Cập nhật similarity corpus với câu hỏi hiện có
        existing_data = GeneratedData.query.filter_by(topic_id=topic_id).all()
        existing_questions = []
        for item in existing_data:
            existing_questions.append({
                'id': item.id,
                'data_type': item.data_type,
                'content': item.content
            })
        
        data_generator.update_similarity_corpus(existing_questions)
        
        # Sử dụng method mới với article-based generation
        generated_samples = data_generator.generate_from_multiple_documents(
            documents, topic.name, data_type, num_samples
        )
        
        # Lưu vào database với metadata chi tiết
        for sample in generated_samples:
            # Extract metadata nếu có
            metadata = sample.pop('metadata', {})
            
            generated_data = GeneratedData(
                topic_id=topic_id,
                data_type=data_type,
                content=json.dumps({
                    **sample,
                    'metadata': metadata
                }, ensure_ascii=False)
            )
            db.session.add(generated_data)
        
        db.session.commit()
        
        # Tạo summary về document distribution
        document_summary = {}
        for sample in generated_samples:
            if 'metadata' in sample and 'source_document' in sample['metadata']:
                doc_name = sample['metadata']['source_document']
                document_summary[doc_name] = document_summary.get(doc_name, 0) + 1
        
        return jsonify({
            'message': f'Đã sinh {len(generated_samples)} mẫu dữ liệu {data_type}',
            'total_samples': len(generated_samples),
            'document_distribution': document_summary,
            'documents_used': [{'title': doc.title, 'id': doc.id} for doc in documents],
            'samples': generated_samples[:5]  # Chỉ hiển thị 5 samples đầu để preview
        })
        
    except Exception as e:
        return jsonify({'error': f'Lỗi khi sinh dữ liệu: {str(e)}'}), 500

@app.route('/api/data/<int:topic_id>', methods=['GET'])
def get_generated_data(topic_id):
    """Lấy dữ liệu đã sinh cho chủ đề"""
    data_type = request.args.get('type')
    
    query = GeneratedData.query.filter_by(topic_id=topic_id)
    if data_type:
        query = query.filter_by(data_type=data_type)
    
    # Sắp xếp theo ID giảm dần để lấy dữ liệu mới nhất trước
    data = query.order_by(GeneratedData.id.desc()).all()
    
    result = []
    for item in data:
        # Kiểm tra xem có label không
        has_label = LabeledData.query.filter_by(generated_data_id=item.id).first() is not None
        
        result.append({
            'id': item.id,
            'data_type': item.data_type,
            'content': json.loads(item.content),
            'created_at': item.created_at.isoformat(),
            'is_labeled': has_label
        })
    
    return jsonify(result)

@app.route('/api/label', methods=['POST'])
def label_data():
    """Gán nhãn cho dữ liệu"""
    data = request.get_json()
    
    data_id = data.get('data_id')
    label = data.get('label')  # 'accept', 'reject', 'modify'
    modified_content = data.get('modified_content')
    notes = data.get('notes', '')
    
    if not data_id or not label:
        return jsonify({'error': 'Thiếu data_id hoặc label'}), 400
    
    # Kiểm tra xem đã có label chưa
    existing_label = LabeledData.query.filter_by(generated_data_id=data_id).first()
    
    if existing_label:
        # Cập nhật label hiện có
        existing_label.label = label
        existing_label.modified_content = modified_content
        existing_label.notes = notes
    else:
        # Tạo label mới
        labeled_data = LabeledData(
            generated_data_id=data_id,
            label=label,
            modified_content=modified_content,
            notes=notes
        )
        db.session.add(labeled_data)
    
    db.session.commit()
    
    return jsonify({'message': 'Nhãn đã được cập nhật thành công'})

@app.route('/api/export/<data_type>', methods=['GET'])
def export_data(data_type):
    """Xuất dữ liệu đã gán nhãn"""
    topic_id = request.args.get('topic_id')
    
    # Query đơn giản: lấy tất cả dữ liệu được chấp nhận hoặc sửa đổi
    query = db.session.query(GeneratedData, LabeledData).join(
        LabeledData, GeneratedData.id == LabeledData.generated_data_id
    ).filter(
        GeneratedData.data_type == data_type,
        LabeledData.label.in_(['accept', 'modify'])
    )
    
    if topic_id:
        query = query.filter(GeneratedData.topic_id == topic_id)
    
    results = query.all()
    
    # Tạo file JSONL
    export_data = []
    for generated, labeled in results:
        content = json.loads(generated.content)
        # Nếu có modified_content, ưu tiên dùng modified
        if labeled.modified_content:
            try:
                modified = json.loads(labeled.modified_content)
                content.update(modified)
            except:
                pass  # Nếu lỗi parse JSON thì dùng content gốc
        export_data.append(content)
    
    # Lưu file
    filename = f"{data_type}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    filepath = os.path.join('data', 'exports', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with jsonlines.open(filepath, mode='w') as writer:
        for item in export_data:
            writer.write(item)
    
    return send_file(filepath, as_attachment=True)

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Thống kê dữ liệu"""
    total_topics = LegalTopic.query.count()
    total_generated = GeneratedData.query.count()
    total_labeled = LabeledData.query.count()
    
    # Thống kê theo loại dữ liệu
    data_type_stats = db.session.query(
        GeneratedData.data_type,
        db.func.count(GeneratedData.id)
    ).group_by(GeneratedData.data_type).all()
    
    # Thống kê nhãn
    label_stats = db.session.query(
        LabeledData.label,
        db.func.count(LabeledData.id)
    ).group_by(LabeledData.label).all()
    
    return jsonify({
        'total_topics': total_topics,
        'total_generated': total_generated,
        'total_labeled': total_labeled,
        'data_type_distribution': dict(data_type_stats),
        'label_distribution': dict(label_stats)
    })

@app.route('/api/topics/<int:topic_id>/coverage', methods=['POST'])
def analyze_topic_coverage(topic_id):
    """Phân tích độ bao phủ của bộ câu hỏi cho một chủ đề"""
    try:
        data = request.get_json()
        unit_type = data.get('unit_type', 'sentence')  # sentence, paragraph
        threshold = data.get('threshold', 0.3)
        
        # Validate unit_type
        if unit_type not in ['sentence', 'paragraph']:
            return jsonify({'error': 'unit_type phải là "sentence" hoặc "paragraph"'}), 400
        
        # Lấy topic và documents
        topic = LegalTopic.query.get_or_404(topic_id)
        documents = db.session.query(LegalDocument).join(TopicDocument).filter(
            TopicDocument.topic_id == topic_id
        ).all()
        
        if not documents:
            return jsonify({'error': 'Không có document nào cho chủ đề này'}), 400
        
        # Chuẩn bị documents data
        documents_data = []
        for doc in documents:
            documents_data.append({
                'id': doc.id,
                'title': doc.title,
                'content': doc.content
            })
        
        # Lấy câu hỏi đã sinh cho topic này
        questions = GeneratedData.query.filter_by(topic_id=topic_id).all()
        questions_data = []
        for q in questions:
            questions_data.append({
                'id': q.id,
                'data_type': q.data_type,
                'content': q.content
            })
        
        if not questions_data:
            return jsonify({'error': 'Không có câu hỏi nào được sinh cho chủ đề này'}), 400
        
        # Phân tích coverage
        analyzer = CoverageAnalyzer(coverage_threshold=threshold)
        analyzer.prepare_coverage_analysis(documents_data, questions_data, unit_type)
        
        coverage_result = analyzer.analyze_coverage()
        doc_summary = analyzer.get_coverage_summary_by_document(coverage_result)
        
        # Thêm thông tin topic
        coverage_result['topic_info'] = {
            'id': topic.id,
            'name': topic.name,
            'description': topic.description
        }
        
        coverage_result['document_summary'] = doc_summary
        coverage_result['analysis_settings'] = {
            'unit_type': unit_type,
            'threshold': threshold,
            'total_documents': len(documents_data),
            'total_questions': len(questions_data)
        }
        
        return jsonify(coverage_result)
    
    except Exception as e:
        return jsonify({'error': f'Lỗi phân tích coverage: {str(e)}'}), 500

@app.route('/api/coverage/batch', methods=['POST'])
def analyze_batch_coverage():
    """Phân tích coverage cho nhiều topics cùng lúc"""
    try:
        data = request.get_json()
        topic_ids = data.get('topic_ids', [])
        unit_type = data.get('unit_type', 'sentence')
        threshold = data.get('threshold', 0.3)
        
        if not topic_ids:
            return jsonify({'error': 'Cần cung cấp danh sách topic_ids'}), 400
        
        results = {}
        
        for topic_id in topic_ids:
            try:
                topic = LegalTopic.query.get(topic_id)
                if not topic:
                    results[topic_id] = {'error': f'Topic {topic_id} không tồn tại'}
                    continue
                
                documents = db.session.query(LegalDocument).join(TopicDocument).filter(
                    TopicDocument.topic_id == topic_id
                ).all()
                
                questions = GeneratedData.query.filter_by(topic_id=topic_id).all()
                
                if not documents or not questions:
                    results[topic_id] = {
                        'error': 'Không có đủ dữ liệu để phân tích',
                        'documents_count': len(documents),
                        'questions_count': len(questions)
                    }
                    continue
                
                # Phân tích coverage cho topic này
                documents_data = [{'id': doc.id, 'title': doc.title, 'content': doc.content} for doc in documents]
                questions_data = [{'id': q.id, 'data_type': q.data_type, 'content': q.content} for q in questions]
                
                analyzer = CoverageAnalyzer(coverage_threshold=threshold)
                analyzer.prepare_coverage_analysis(documents_data, questions_data, unit_type)
                coverage_result = analyzer.analyze_coverage()
                
                # Chỉ lưu summary, không lưu chi tiết units
                results[topic_id] = {
                    'topic_name': topic.name,
                    'total_units': coverage_result['total_units'],
                    'covered_units': coverage_result['covered_units'],
                    'coverage_percentage': coverage_result['coverage_percentage'],
                    'documents_count': len(documents),
                    'questions_count': len(questions)
                }
                
            except Exception as e:
                results[topic_id] = {'error': f'Lỗi phân tích topic {topic_id}: {str(e)}'}
        
        return jsonify({
            'analysis_settings': {
                'unit_type': unit_type,
                'threshold': threshold
            },
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': f'Lỗi phân tích batch coverage: {str(e)}'}), 500

@app.route('/api/system/health', methods=['GET'])
def system_health():
    """Kiểm tra sức khỏe hệ thống và các component"""
    try:
        # Kiểm tra database
        total_topics = LegalTopic.query.count()
        total_documents = LegalDocument.query.count()
        total_generated = GeneratedData.query.count()
        
        # Kiểm tra similarity checker
        similarity_status = "OK" if data_generator.similarity_checker else "NOT_INITIALIZED"
        
        # Kiểm tra coverage analyzer
        try:
            test_analyzer = CoverageAnalyzer()
            coverage_status = "OK"
        except Exception:
            coverage_status = "ERROR"
        
        return jsonify({
            'status': 'healthy',
            'components': {
                'database': {
                    'status': 'OK',
                    'topics': total_topics,
                    'documents': total_documents,
                    'generated_questions': total_generated
                },
                'similarity_checker': {
                    'status': similarity_status,
                    'threshold': getattr(data_generator, 'similarity_threshold', None)
                },
                'coverage_analyzer': {
                    'status': coverage_status
                },
                'data_generator': {
                    'status': 'OK' if data_generator else 'ERROR'
                }
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/system/supported-formats', methods=['GET'])
def get_supported_file_formats():
    """Lấy danh sách các format file được hỗ trợ"""
    try:
        formats = get_supported_formats()
        return jsonify({
            'supported_formats': formats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': f'Lỗi khi lấy danh sách format: {str(e)}'
        }), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)
