from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core import Document
import os

def load_cleaned_text(file_path):
    """Load text đã cleaned từ file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def chunk_text_with_sentence_splitter(text, chunk_size=512, chunk_overlap=50):
    """
    Chunk text sử dụng SentenceSplitter của LlamaIndex
    
    Args:
        text: Text cần chunk
        chunk_size: Kích thước tối đa của chunk (tokens)
        chunk_overlap: Số tokens overlap giữa các chunk
    """
    # Khởi tạo SentenceSplitter
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" ",  # Phân tách bằng space
    )
    
    # Tạo Document từ text
    document = Document(text=text)
    
    # Split thành chunks
    chunks = splitter.split_text(text)
    
    return chunks

def chunk_text_advanced(text, chunk_size=512, chunk_overlap=50):

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" ",
        # Các separator cho tiếng Việt
        secondary_chunking_regex=r'[.!?][\s]+',  # Phân tách theo câu
    )
    
    chunks = splitter.split_text(text)
    return chunks

def save_chunks_to_files(chunks, output_dir="./Data/chunks"):
    """Lưu chunks vào các file riêng biệt"""
    # Tạo thư mục nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    for i, chunk in enumerate(chunks):
        output_path = os.path.join(output_dir, f"chunk_{i+1:03d}.txt")
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(chunk)
    
    print(f"Đã lưu {len(chunks)} chunks vào thư mục: {output_dir}")

def save_chunks_to_single_file(chunks, output_path="./Data/all_chunks.txt"):
    """Lưu tất cả chunks vào 1 file với separator"""
    with open(output_path, 'w', encoding='utf-8') as file:
        for i, chunk in enumerate(chunks):
            file.write(f"=== CHUNK {i+1} ===\n")
            file.write(chunk)
            file.write(f"\n\n{'='*50}\n\n")
    
    print(f"Đã lưu {len(chunks)} chunks vào: {output_path}")

def analyze_chunks(chunks):
    """Phân tích thông tin về chunks"""
    total_chunks = len(chunks)
    chunk_lengths = [len(chunk) for chunk in chunks]
    avg_length = sum(chunk_lengths) / total_chunks if total_chunks > 0 else 0
    
    print(f"Tổng số chunks: {total_chunks}")
    print(f"Độ dài trung bình: {avg_length:.2f} ký tự")
    print(f"Chunk ngắn nhất: {min(chunk_lengths)} ký tự")
    print(f"Chunk dài nhất: {max(chunk_lengths)} ký tự")
    
    return {
        'total_chunks': total_chunks,
        'avg_length': avg_length,
        'min_length': min(chunk_lengths),
        'max_length': max(chunk_lengths)
    }

# def chunk_multiple_files(input_dir="./Data", pattern="*_cleaned.txt"):
#     """Chunk nhiều file cleaned text"""
#     import glob
    
#     files = glob.glob(os.path.join(input_dir, pattern))
#     all_chunks = []
    
#     for file_path in files:
#         print(f"Processing: {file_path}")
#         text = load_cleaned_text(file_path)
#         chunks = chunk_text_with_sentence_splitter(text)
        
#         # Thêm metadata cho chunks
#         filename = os.path.basename(file_path).replace('_cleaned.txt', '')
#         chunks_with_metadata = [f"[FILE: {filename}]\n{chunk}" for chunk in chunks]
#         all_chunks.extend(chunks_with_metadata)
    
#     return all_chunks


if __name__ == "__main__":
    # Load text đã cleaned
    cleaned_text_path = "./Data/Quyết định dự đoán_cleaned.txt"
    
    if os.path.exists(cleaned_text_path):
        # Load text
        text = load_cleaned_text(cleaned_text_path)
        print(f"Loaded text length: {len(text)} characters")
        
        # Chunk text với các setting khác nhau
        print("\n" + "="*50)
        print("CHUNKING WITH DIFFERENT SETTINGS:")
        
        # Setting 1: Chunk nhỏ
        chunks_small = chunk_text_with_sentence_splitter(text, chunk_size=256, chunk_overlap=25)
        print(f"\nSmall chunks (256 tokens):")
        analyze_chunks(chunks_small)
        
        # Setting 2: Chunk vừa
        chunks_medium = chunk_text_with_sentence_splitter(text, chunk_size=512, chunk_overlap=50)
        print(f"\nMedium chunks (512 tokens):")
        analyze_chunks(chunks_medium)
        
        # Setting 3: Chunk lớn
        chunks_large = chunk_text_with_sentence_splitter(text, chunk_size=1024, chunk_overlap=100)
        print(f"\nLarge chunks (1024 tokens):")
        analyze_chunks(chunks_large)
        
        # Lưu chunks (sử dụng medium setting)
        print("\n" + "="*50)
        print("SAVING CHUNKS:")
        
        # Lưu vào files riêng biệt
        save_chunks_to_files(chunks_medium, "./Data/chunks")
        
        # Lưu vào 1 file
        save_chunks_to_single_file(chunks_medium, "./Data/all_chunks.txt")
        
        # Preview chunks
        print("\n" + "="*50)
        print("PREVIEW FIRST 3 CHUNKS:")
        for i, chunk in enumerate(chunks_medium[:3]):
            print(f"\n--- CHUNK {i+1} ---")
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    
    else:
        print(f"File không tồn tại: {cleaned_text_path}")
        print("Hãy chạy extract.py trước để tạo file cleaned text")