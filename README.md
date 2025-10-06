# 🏛️ Legal Data Tool

Công cụ xử lý và phân tích dữ liệu văn bản pháp luật Việt Nam với AI

## 📋 Tổng quan

**Legal Data Tool** là hệ thống hoàn chỉnh để xử lý, phân tích và sinh dữ liệu từ văn bản pháp luật Việt Nam. Hệ thống hỗ trợ upload multi-format files, phân tích cấu trúc pháp lý, sinh QA tự động và theo dõi độ bao phủ dữ liệu.

## ✨ Tính năng chính

### 📄 **Xử lý văn bản đa dạng**
- **Upload files**: `.doc`, `.docx`, `.pdf`, `.txt`
- **Parse tự động**: Phân tích cấu trúc điều, khoản, điểm
- **Import CSV**: Tích hợp dữ liệu từ nguồn có sẵn
- **Text processing**: Xử lý encoding tự động

### � **AI-Powered Generation**
- **Multi-LLM support**: Gemini, Qwen, HuggingFace Cloud
- **Batch generation**: Tối ưu API calls
- **Smart sampling**: Monte Carlo với weighted selection
- **Context-aware**: Sinh QA dựa trên cấu trúc pháp lý

### 🔍 **Advanced Analysis**
- **Hybrid search**: Vector + keyword search  
- **Coverage analysis**: Theo dõi độ bao phủ QA
- **Similarity detection**: Lọc trùng lặp thông minh
- **Statistics dashboard**: Báo cáo chi tiết

### 📊 **Data Management**
- **Smart labeling**: Giao diện gán nhãn intuitive
- **Export formats**: JSONL, CSV, structured data
- **Version control**: Theo dõi changes và history
- **Quality assurance**: Validation và quality checks  

## 🚀 Cài đặt

### Phương pháp 1: Quick Start (Khuyến nghị)
```powershell
# Windows PowerShell
.\setup.ps1      # Tự động cài đặt dependencies
.\start.bat      # Khởi chạy cả frontend và backend
```

### Phương pháp 2: Manual Setup
```bash
# 1. Backend Setup
cd backend
pip install -r requirements.txt

# 2. Frontend Setup  
cd ../frontend
npm install

# 3. Environment Configuration
# Tạo .env và thêm API keys:
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_API_KEY=your_gemini_api_key

# 4. Initialize Database
cd ../backend
python -c "from app import app, db; app.app_context().push(); db.create_all()"

# 5. Start Services
python app.py        # Backend: http://localhost:5000
cd ../frontend
npm start           # Frontend: http://localhost:3000
```

## 🏗️ Kiến trúc hệ thống

```
DataTool/
├── backend/                    # 🐍 Flask API Server
│   ├── app.py                    # Main application & routes
│   ├── models.py                 # SQLAlchemy database models  
│   ├── config.py                 # Environment configuration
│   ├── data_generator.py         # 🤖 AI data generation engine
│   ├── document_parsers.py       # 📄 Legal document parsing
│   ├── file_handler.py           # 📁 Multi-format file processing
│   ├── hybrid_search.py          # 🔍 Vector + keyword search
│   ├── similarity_checker.py     # 🎯 Duplicate detection
│   ├── coverage_analyzer.py      # 📊 Coverage analysis
│   ├── vanban_csv.py            # 📋 CSV data integration
│   └── instance/                 # SQLite database storage
│       └── legal_data.db
├── frontend/                   # ⚛️ React UI
│   ├── src/
│   │   ├── components/           # UI Components
│   │   │   ├── HomePage.js         # Dashboard tổng quan
│   │   │   ├── DocumentManagement.js # Quản lý văn bản
│   │   │   ├── DataGeneration.js   # Sinh dữ liệu QA
│   │   │   ├── DataLabeling.js     # Gán nhãn dữ liệu
│   │   │   ├── CoverageDemo.js     # Phân tích coverage
│   │   │   ├── Statistics.js       # Thống kê & báo cáo
│   │   │   └── DataExport.js       # Xuất dữ liệu
│   │   ├── services/             # API integration
│   │   └── hooks/                # React hooks
│   └── public/                   # Static assets
├── data/                       # 💾 Data storage
│   ├── van_ban_phap_luat_async.csv # Source legal documents
│   └── exports/                  # Generated datasets
├── requirements.txt            # Python dependencies
├── setup.ps1                  # Automated setup script
├── start.bat                  # Launch script
└── README.md                  # Documentation
```

## 🎯 Workflow sử dụng

### 1. 🏠 **Dashboard Overview**
- Xem tổng quan hệ thống
- Theo dõi thống kê documents, QA pairs
- Quick access các tính năng chính

### 2. 📁 **Document Management**
- **Upload files**: Drag & drop `.doc`, `.docx`, `.pdf`, `.txt`
- **CSV integration**: Import từ nguồn dữ liệu có sẵn
- **Auto parsing**: Hệ thống tự động phân tích cấu trúc
- **View structure**: Xem hierarchy điều/khoản/điểm

### 3. 🤖 **Data Generation**
- **Select documents**: Chọn văn bản cần sinh QA
- **Configure parameters**: 
  - Số lượng câu hỏi (5-10 per batch)
  - Focus areas (specific articles/sections)
  - Question types (factual/analytical/procedural)
- **AI processing**: Multi-LLM generation với context
- **Real-time monitoring**: Track progress và quality

### 4. 🏷️ **Data Labeling**
- **Review interface**: User-friendly labeling UI
- **Quality control**: Accept/Reject/Edit generated QA
- **Bulk operations**: Mass labeling với filters
- **Export ready data**: Chỉ export data đã được label

### 5. 📊 **Coverage Analysis**
- **Document coverage**: Xem % văn bản đã có QA
- **Gap analysis**: Identify sections cần thêm QA
- **Quality metrics**: Similarity scores, duplicates
- **Visual reports**: Charts và heatmaps

### 6. 📤 **Data Export**
- **Multiple formats**: JSONL, CSV, structured JSON
- **Filter options**: By document, date, quality score
- **Preview before export**: Kiểm tra data trước khi tải
- **Batch download**: Export large datasets

## 🚀 Quick Start Guide

```bash
# 1. Launch system
.\start.bat

# 2. Access web interface
# Frontend: http://localhost:3000
# Backend API: http://localhost:5000

# 3. Upload your first document
# Go to Document Management → Upload File

# 4. Generate QA pairs
# Go to Data Generation → Select Document → Generate

# 5. Label and export
# Go to Data Labeling → Review → Export Data
```

## 📋 Data Formats & Examples

### QA Pairs (Question-Answer)
```json
{
  "id": "qa_001",
  "question": "Mức phạt tiền đối với hành vi lái xe không có giấy phép là bao nhiêu?",
  "answer": "Theo Nghị định 100/2019, mức phạt từ 8.000.000đ đến 12.000.000đ",
  "source": {
    "document_title": "Nghị định 100/2019/NĐ-CP",
    "unit_path": "Điều 6, Khoản 1, Điểm a",
    "unit_id": "dieu_6_khoan_1_diem_a"
  },
  "metadata": {
    "generated_at": "2025-09-19T10:30:00Z",
    "model": "gemini-1.5-pro",
    "similarity_score": 0.95,
    "labeled": true,
    "label_status": "approved"
  }
}
```

### Document Structure
```json
{
  "document_id": "doc_001",
  "title": "Luật Giao thông đường bộ 2008",
  "parsed_structure": {
    "articles": [
      {
        "article_number": "1",
        "title": "Phạm vi điều chỉnh",
        "content": "Luật này quy định về...",
        "clauses": [
          {
            "clause_number": "1", 
            "content": "Giao thông đường bộ bao gồm...",
            "points": []
          }
        ]
      }
    ]
  },
  "articles_count": 85,
  "file_type": "docx",
  "uploaded_at": "2025-09-19T10:00:00Z"
}
```

### Export Formats

**JSONL for Training:**
```jsonl
{"instruction": "Question 1", "output": "Answer 1", "source": "Article 1"}
{"instruction": "Question 2", "output": "Answer 2", "source": "Article 2"}
```

**CSV for Analysis:**
```csv
id,question,answer,document,article,similarity_score,label_status
1,"Question 1","Answer 1","Doc 1","Article 1",0.95,approved
2,"Question 2","Answer 2","Doc 1","Article 2",0.87,pending
```

## 🔌 API Reference

### Document Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/documents` | List all documents |
| `POST` | `/api/documents` | Create new document |
| `POST` | `/api/documents/upload` | Upload file (multi-format) |
| `GET` | `/api/documents/{id}` | Get document details |
| `DELETE` | `/api/documents/{id}` | Delete document |

### Data Generation
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/generate` | Generate QA pairs |
| `GET` | `/api/qa-pairs` | List generated QA pairs |
| `GET` | `/api/qa-pairs/{id}` | Get QA pair details |
| `PUT` | `/api/qa-pairs/{id}/label` | Update QA pair label |

### Analysis & Export
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/coverage` | Get coverage analysis |
| `GET` | `/api/statistics` | System statistics |
| `POST` | `/api/export` | Export data (JSONL/CSV) |
| `GET` | `/api/search/documents` | Search documents |
| `GET` | `/api/search/vanban` | Search CSV data |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | API health check |
| `GET` | `/api/supported-formats` | Supported file formats |

## ⚙️ Configuration

### Environment Variables
Create `.env` file in project root:
```bash
# AI Model Configuration
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Application Settings
FLASK_ENV=development
DATABASE_URL=sqlite:///legal_data.db
SECRET_KEY=your_secret_key_here

# File Upload Settings
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216  # 16MB

# CORS Settings
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### AI Model Setup
**Gemini API:**
1. Visit [Google AI Studio](https://aistudio.google.com)
2. Create new API key
3. Add to `.env` file

**HuggingFace Cloud (Optional):**
- Configure in `test_ngrok.py` for cloud inference
- Used when local GPU unavailable

## 🔧 Technical Architecture

### Backend Components
```
🐍 Flask Application (app.py)
├── 📊 Models (models.py)
│   ├── LegalDocument
│   ├── QAPair  
│   ├── LegalTopic
│   └── TopicDocument
├── 🤖 AI Engine (data_generator.py)
│   ├── Multi-LLM support
│   ├── Batch generation
│   ├── Monte Carlo sampling
│   └── Context management
├── 📄 Document Processing
│   ├── file_handler.py (multi-format)
│   ├── document_parsers.py (structure)
│   └── vanban_csv.py (CSV integration)
├── 🔍 Analysis Engine
│   ├── hybrid_search.py (vector + keyword)
│   ├── similarity_checker.py (duplicates)
│   └── coverage_analyzer.py (gap analysis)
└── 💾 Database (SQLite)
    └── legal_data.db
```

### Frontend Architecture
```
⚛️ React Application
├── 🏠 HomePage (dashboard)
├── 📁 DocumentManagement (CRUD)
├── 🤖 DataGeneration (AI-powered)
├── 🏷️ DataLabeling (manual review)
├── � CoverageDemo (analysis)
├── 📈 Statistics (reports)
├── 📤 DataExport (download)
└── 🔧 Services (API integration)
```

### Key Features
- **Multi-format Support**: `.doc`, `.docx`, `.pdf`, `.txt`
- **Smart Parsing**: Legal structure recognition
- **Batch Processing**: Optimized AI generation
- **Real-time Analysis**: Coverage và similarity
- **Responsive UI**: Modern React interface

## 🧪 Testing & Development

### Initialize Sample Data
```bash
cd backend
python -c "from app import app, db; app.app_context().push(); db.create_all()"
```

### Test AI Models
```bash
# Test Gemini API
python test_ngrok.py

# Check supported file formats
python -c "from file_handler import get_supported_formats; print(get_supported_formats())"
```

### Development Tools
```bash
# Database reset (if needed)
rm instance/legal_data.db
python -c "from app import app, db; app.app_context().push(); db.create_all()"

# Check system status
curl http://localhost:5000/api/health

# View database contents
python -c "from app import app, db, LegalDocument; app.app_context().push(); print(LegalDocument.query.count())"
```

## 🔧 Troubleshooting

### Common Issues

**🚫 Backend startup fails:**
```bash
# Install dependencies
pip install -r requirements.txt

# Check Python version (requires 3.8+)
python --version

# Verify database
python -c "from app import app, db; app.app_context().push(); db.create_all()"
```

**🚫 Frontend startup fails:**
```bash
cd frontend
npm install --force
npm audit fix
npm start
```

**🚫 File upload errors:**
- Check file format support (`.doc` requires `docx2txt`)
- Verify file size (< 16MB)
- Check upload folder permissions

**🚫 AI generation fails:**
- Verify `GOOGLE_API_KEY` in `.env`
- Check API quota at [Google AI Studio](https://aistudio.google.com)
- Review backend logs for specific errors

**🚫 Database issues:**
```bash
# Check database connection
python -c "from app import app, db; app.app_context().push(); print('DB connected:', db.engine.url)"

# Reset if corrupted
rm instance/legal_data.db
python -c "from app import app, db; app.app_context().push(); db.create_all()"
```

## 📚 Documentation

### Project Structure Details
- **Backend**: Flask REST API with SQLAlchemy ORM
- **Frontend**: React SPA with Ant Design UI
- **Database**: SQLite for development, PostgreSQL for production
- **AI Models**: Multi-provider support (Gemini, Qwen, HuggingFace)

### Development Roadmap
- [ ] PostgreSQL production support
- [ ] User authentication & authorization  
- [ ] Advanced analytics dashboard
- [ ] Batch file processing
- [ ] API rate limiting
- [ ] Export scheduling

## 📄 License

MIT License - Free for educational and research purposes.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## Link Data
https://huggingface.co/datasets/phamhoangf/legal_generated_data

## 📞 Support

For issues and questions:
- Create GitHub issue
- Check documentation
- Review troubleshooting guide

---

**Legal Data Tool** - Empowering legal document analysis with AI 
