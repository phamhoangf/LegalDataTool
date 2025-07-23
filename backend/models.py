from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class LegalTopic(db.Model):
    """Model cho chủ đề pháp lý"""
    __tablename__ = 'legal_topics'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    legal_text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    generated_data = db.relationship('GeneratedData', backref='topic', lazy=True)
    
    def __repr__(self):
        return f'<LegalTopic {self.name}>'

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
