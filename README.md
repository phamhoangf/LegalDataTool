# LegalSLM Data Generation Tool

Tool tạo dữ liệu finetune SFT/CoT/RLHF cho mô hình LegalSLM

## Tính năng chính

 **Quản lý chủ đề pháp lý** - Tạo và quản lý các chủ đề luật khác nhau  
 **Tải lên văn bản luật** - Hỗ trợ file text và paste trực tiếp  
 **Sinh dữ liệu tự động** - Tạo dữ liệu SFT, CoT, RLHF bằng AI  
 **Gán nhãn thông minh** - Giao diện thân thiện cho chuyên gia luật  
 **Xuất dữ liệu chuẩn** - File .jsonl sẵn sàng cho huấn luyện  
 **Thống kê chi tiết** - Theo dõi tiến độ và chất lượng dữ liệu  

## Cài đặt nhanh

### Phương pháp 1: Automatic Setup (Khuyến nghị)
```bash
# Chạy script setup tự động
.\setup.ps1

# Khởi chạy ứng dụng
.\start.bat
```

### Phương pháp 2: Manual Setup
```bash
# 1. Cài đặt Backend (Flask)
pip install -r requirements.txt

# 2. Cài đặt Frontend (React)  
cd frontend
npm install
cd ..

# 3. Cấu hình environment
# Sửa file .env và thêm GOOGLE_API_KEY

# 4. Tạo dữ liệu mẫu (tùy chọn)
cd backend
python create_sample_data.py
cd ..

# 5. Khởi chạy Backend
cd backend  
python app.py &
cd ..

# 6. Khởi chạy Frontend
cd frontend
npm start
```

## Cấu trúc dự án

```
DataTool/
├── backend/                 # Flask API Server
│   ├── app.py                 # Main application
│   ├── models.py              # Database models  
│   ├── data_generator.py      # AI data generation
│   ├── config.py              # Configuration
│   └── create_sample_data.py  # Sample data script
├── frontend/               # React UI
│   ├── src/components/        # UI Components
│   ├── src/services/          # API Services
│   └── public/                # Static files
├── data/                   # Data storage
│   └── exports/               # Generated .jsonl files
├── requirements.txt        # Python dependencies
├── setup.ps1              # Setup script
├── start.bat              # Start script
└── README.md              # This file
```

## Sử dụng

### Bước 1: Test Google AI
```bash
cd backend
python test_google_ai.py
```

### Bước 2: Khởi chạy ứng dụng
```bash
.\start.bat
```
- **Backend:** http://localhost:5000
- **Frontend:** http://localhost:3000

### Bước 3: Workflow cơ bản

1. ** Quản lý chủ đề**
   - Tạo chủ đề pháp lý mới (VD: "Giấy phép lái xe")
   - Tải lên văn bản luật liên quan
   - Thêm mô tả chi tiết

2. ** Sinh dữ liệu**
   - Chọn chủ đề đã có văn bản
   - Chọn loại dữ liệu: SFT / CoT / RLHF
   - Thiết lập số lượng mẫu (5-20 mẫu/lần)
   - Nhấn "Sinh dữ liệu"

3. ** Gán nhãn**
   - Duyệt từng mẫu dữ liệu đã sinh
   - Chọn: Chấp nhận / Từ chối / Sửa đổi
   - Thêm ghi chú và chỉnh sửa nếu cần

4. ** Thống kê & Kiểm tra**
   - Xem tiến độ hoàn thành
   - Phân tích chất lượng dữ liệu
   - Kiểm tra phân bố các loại nhãn

5. ** Xuất dữ liệu**
   - Chọn chủ đề và loại dữ liệu
   - Preview dữ liệu sẽ xuất
   - Tải file .jsonl hoàn chỉnh

## Các loại dữ liệu

### SFT (Supervised Fine-Tuning)
```json
{
  "instruction": "Thời hạn GPLX hạng A1 là bao lâu?",
  "output": "Theo Thông tư 12/2017, GPLX hạng A1 có giá trị không thời hạn."
}
```

### CoT (Chain-of-Thought)  
```json
{
  "instruction": "Người 17 tuổi có được thi GPLX không?",
  "reasoning_steps": [
    "Bước 1: Xem điều kiện dự thi",
    "Bước 2: Theo Thông tư 12, tuổi tối thiểu thi A1 là 18",
    "Bước 3: So sánh 17 < 18"
  ],
  "final_answer": "Không, chưa đủ điều kiện về tuổi"
}
```

### RLHF (Human Feedback)
```json
{
  "prompt": "Tư vấn thủ tục đổi GPLX hết hạn",
  "response_a": "Hướng dẫn đầy đủ, chính xác...",
  "response_b": "Hướng dẫn thiếu sót, không rõ ràng...",
  "preferred": "A"
}
```

## API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/api/health` | Kiểm tra trạng thái API |
| `GET` | `/api/topics` | Lấy danh sách chủ đề |
| `POST` | `/api/topics` | Tạo chủ đề mới |
| `POST` | `/api/upload` | Tải lên văn bản luật |
| `POST` | `/api/generate` | Sinh dữ liệu huấn luyện |
| `GET` | `/api/data/{id}` | Lấy dữ liệu đã sinh |
| `POST` | `/api/label` | Gán nhãn dữ liệu |
| `GET` | `/api/export/{type}` | Xuất dữ liệu |
| `GET` | `/api/stats` | Thống kê tổng quan |

## Cấu hình

### Environment Variables (.env)
```bash
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_API_KEY=your_google_api_key_here
FLASK_ENV=development
DATABASE_URL=sqlite:///legal_data.db
```

### Cấu hình Google AI
1. Truy cập Google AI Studio: https://aistudio.google.com
2. Tạo API key mới
3. Thêm key vào file `.env`

## Demo & Test

```bash
# Tạo dữ liệu mẫu để test
cd backend
python create_sample_data.py
```

Dữ liệu mẫu bao gồm:
- 1 chủ đề "Giấy phép lái xe"
- Văn bản Luật Giao thông đường bộ
- 3 mẫu SFT, 1 mẫu CoT, 1 mẫu RLHF
- Labels mẫu để test workflow

## Troubleshooting

### Lỗi thường gặp:

**Backend không khởi chạy:**
```bash
# Kiểm tra Python dependencies
pip install -r requirements.txt

# Kiểm tra database
python -c "from backend.app import app, db; app.app_context().push(); db.create_all()"
```

**Frontend không khởi chạy:**
```bash
cd frontend
npm install --force
npm start
```

**Không sinh được dữ liệu:**
- Kiểm tra GOOGLE_API_KEY trong `.env`
- Kiểm tra quota Google AI Studio
- Xem logs trong terminal backend

**Database lỗi:**
```bash
# Reset database
rm backend/legal_data.db
cd backend
python create_sample_data.py
```

## License

MIT License - Tự do sử dụng cho mục đích học tập và nghiên cứu.

## Đóng góp


---

# LegalDataTool
