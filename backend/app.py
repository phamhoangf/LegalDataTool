from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os
import json
import jsonlines
from datetime import datetime
from data_generator import DataGenerator
from models import db, LegalTopic, GeneratedData, LabeledData

load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///legal_data.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
CORS(app)
db.init_app(app)

# Initialize data generator with Google API key
google_api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
data_generator = DataGenerator(api_key=google_api_key)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/topics', methods=['GET'])
def get_topics():
    """Lấy danh sách chủ đề pháp lý"""
    topics = LegalTopic.query.all()
    return jsonify([{
        'id': topic.id,
        'name': topic.name,
        'description': topic.description,
        'legal_text': topic.legal_text,
        'created_at': topic.created_at.isoformat()
    } for topic in topics])

@app.route('/api/topics', methods=['POST'])
def create_topic():
    """Tạo chủ đề pháp lý mới"""
    data = request.get_json()
    
    topic = LegalTopic(
        name=data['name'],
        description=data.get('description', ''),
        legal_text=data.get('legal_text', '')
    )
    
    db.session.add(topic)
    db.session.commit()
    
    return jsonify({
        'id': topic.id,
        'name': topic.name,
        'description': topic.description,
        'legal_text': topic.legal_text,
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
    GeneratedData.query.filter_by(topic_id=topic_id).delete()
    
    db.session.delete(topic)
    db.session.commit()
    
    return jsonify({
        'message': 'Chủ đề đã được xóa thành công'
    })

@app.route('/api/upload', methods=['POST'])
def upload_legal_document():
    """Tải lên văn bản luật"""
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được tải lên'}), 400
    
    file = request.files['file']
    topic_id = request.form.get('topic_id')
    
    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400
    
    # Đọc nội dung file
    content = file.read().decode('utf-8')
    
    # Cập nhật chủ đề với văn bản luật
    if topic_id:
        topic = LegalTopic.query.get(topic_id)
        if topic:
            topic.legal_text = content
            db.session.commit()
    
    return jsonify({
        'message': 'File đã được tải lên thành công',
        'content_length': len(content)
    })

@app.route('/api/generate', methods=['POST'])
def generate_training_data():
    """Sinh dữ liệu huấn luyện"""
    data = request.get_json()
    
    topic_id = data.get('topic_id')
    data_type = data.get('data_type')  # 'sft', 'cot', 'rlhf'
    num_samples = data.get('num_samples', 10)
    
    if not topic_id or not data_type:
        return jsonify({'error': 'Thiếu topic_id hoặc data_type'}), 400
    
    topic = LegalTopic.query.get(topic_id)
    if not topic:
        return jsonify({'error': 'Không tìm thấy chủ đề'}), 404
    
    try:
        # Sinh dữ liệu dựa trên loại
        if data_type == 'sft':
            generated_samples = data_generator.generate_sft_data(
                topic.legal_text, topic.name, num_samples
            )
        elif data_type == 'cot':
            generated_samples = data_generator.generate_cot_data(
                topic.legal_text, topic.name, num_samples
            )
        elif data_type == 'rlhf':
            generated_samples = data_generator.generate_rlhf_data(
                topic.legal_text, topic.name, num_samples
            )
        else:
            return jsonify({'error': 'Loại dữ liệu không hợp lệ'}), 400
        
        # Lưu vào database
        for sample in generated_samples:
            generated_data = GeneratedData(
                topic_id=topic_id,
                data_type=data_type,
                content=json.dumps(sample, ensure_ascii=False)
            )
            db.session.add(generated_data)
        
        db.session.commit()
        
        return jsonify({
            'message': f'Đã sinh {len(generated_samples)} mẫu dữ liệu {data_type.upper()}',
            'samples': generated_samples
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
    
    data = query.all()
    
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

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)
