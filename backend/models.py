from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class LegalDocument(db.Model):
    """Model cho văn bản luật"""
    __tablename__ = 'legal_documents'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(500), nullable=False)
    content = db.Column(db.Text, nullable=False)
    parsed_structure = db.Column(db.Text)  # JSON string of parsed document structure
    document_type = db.Column(db.String(50))  # 'law', 'decree', 'circular', etc.
    document_number = db.Column(db.String(100))  # Số hiệu văn bản
    issued_date = db.Column(db.Date)
    effective_date = db.Column(db.Date)
    source_url = db.Column(db.String(500))
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    uploaded_by = db.Column(db.String(100))
    
    # Relationship
    topic_documents = db.relationship('TopicDocument', backref='document', lazy=True)
    
    def __repr__(self):
        return f'<LegalDocument {self.title}>'

class LegalTopic(db.Model):
    """Model cho chủ đề pháp lý"""
    __tablename__ = 'legal_topics'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    generated_data = db.relationship('GeneratedData', backref='topic', lazy=True)
    topic_documents = db.relationship('TopicDocument', backref='topic', lazy=True)
    
    def __repr__(self):
        return f'<LegalTopic {self.name}>'

class TopicDocument(db.Model):
    """Model liên kết giữa Topic và Document (many-to-many)"""
    __tablename__ = 'topic_documents'
    
    id = db.Column(db.Integer, primary_key=True)
    topic_id = db.Column(db.Integer, db.ForeignKey('legal_topics.id'), nullable=False)
    document_id = db.Column(db.Integer, db.ForeignKey('legal_documents.id'), nullable=False)
    relevance_score = db.Column(db.Float, default=1.0)  # Độ liên quan 0-1
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    added_by = db.Column(db.String(100))
    
    def __repr__(self):
        return f'<TopicDocument {self.topic_id}-{self.document_id}>'

class GeneratedData(db.Model):
    """Model cho dữ liệu được sinh tự động"""
    __tablename__ = 'generated_data'
    
    id = db.Column(db.Integer, primary_key=True)
    topic_id = db.Column(db.Integer, db.ForeignKey('legal_topics.id'), nullable=False)
    data_type = db.Column(db.String(10), nullable=False)  # 'sft', 'cot', 'rlhf'
    content = db.Column(db.Text, nullable=False)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    labeled_data = db.relationship('LabeledData', backref='generated_data', lazy=True)
    
    def __repr__(self):
        return f'<GeneratedData {self.id} - {self.data_type}>'

class LabeledData(db.Model):
    """Model cho dữ liệu đã được gán nhãn"""
    __tablename__ = 'labeled_data'
    
    id = db.Column(db.Integer, primary_key=True)
    generated_data_id = db.Column(db.Integer, db.ForeignKey('generated_data.id'), nullable=False)
    label = db.Column(db.String(20), nullable=False)  # 'accept', 'reject', 'modify'
    modified_content = db.Column(db.Text)  # JSON string cho nội dung đã sửa
    notes = db.Column(db.Text)
    labeled_by = db.Column(db.String(100))  # ID người gán nhãn
    labeled_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<LabeledData {self.id} - {self.label}>'
