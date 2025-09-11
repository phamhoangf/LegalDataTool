import React, { useState, useEffect } from 'react';
import {
  Card,
  Button,
  Select,
  Table,
  Space,
  message,
  Tag,
  Alert,
  Descriptions,
  Row,
  Col,
  Statistic,
  Divider
} from 'antd';
import {
  DownloadOutlined,
  FileTextOutlined,
  CheckCircleOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import apiService from '../services/api';

const { Option } = Select;const DataExport = () => {
  const [topics, setTopics] = useState([]);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [dataType, setDataType] = useState('word_matching');
  const [dataPreview, setDataPreview] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    loadTopics();
    loadStats();
  }, []);

  useEffect(() => {
    if (selectedTopic) {
      loadDataPreview();
    }
  }, [selectedTopic, dataType]);

  const loadTopics = async () => {
    try {
      const response = await apiService.getTopics();
      // Lấy tất cả topics, không filter
      setTopics(response.data);
    } catch (error) {
      message.error('Không thể tải danh sách chủ đề');
    }
  };

  const loadStats = async () => {
    try {
      const response = await apiService.getStatistics();
      setStats(response.data);
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  };

  const loadDataPreview = async () => {
    try {
      setLoading(true);
      const response = await apiService.getGeneratedData(selectedTopic, dataType);
      // Lấy tất cả dữ liệu đã có nhãn để preview
      const labeledData = response.data.filter(item => item.is_labeled);
      setDataPreview(labeledData);
    } catch (error) {
      message.error('Không thể tải dữ liệu preview');
      console.error('Error loading preview:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    try {
      setExporting(true);
      
      const response = await apiService.exportData(dataType, selectedTopic);
      
      // Tạo URL để download file
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      
      // Tạo tên file
      const topicName = selectedTopic 
        ? topics.find(t => t.id === selectedTopic)?.name.replace(/\s+/g, '_')
        : 'all_topics';
      const timestamp = new Date().toISOString().slice(0, 19).replace(/[:-]/g, '');
      const filename = `${dataType}_${topicName}_${timestamp}.jsonl`;
      
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      message.success(`Đã xuất file: ${filename}`);
    } catch (error) {
      message.error('Không thể xuất dữ liệu');
    } finally {
      setExporting(false);
    }
  };

  const renderPreviewContent = (item) => {
    const content = typeof item.content === 'string' ? JSON.parse(item.content) : item.content;
    
    const truncateText = (text, maxLength = 100) => {
      return text && text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    };
    
    switch (item.data_type) {
      case 'word_matching':
        return (
          <div style={{ maxWidth: '400px', wordWrap: 'break-word' }}>
            <strong>Câu hỏi:</strong> {truncateText(content.question, 80)}<br />
            <strong>Câu trả lời:</strong> {truncateText(content.answer, 80)}<br />
            {content.metadata?.source_document && (
              <><strong>Nguồn:</strong> {truncateText(content.metadata.source_document, 50)}<br /></>
            )}
          </div>
        );
      case 'concept_understanding':
        return (
          <div style={{ maxWidth: '400px', wordWrap: 'break-word' }}>
            <strong>Câu hỏi:</strong> {truncateText(content.question, 80)}<br />
            <strong>Câu trả lời:</strong> {truncateText(content.answer, 80)}<br />
            <strong>Giải thích:</strong> {truncateText(content.explanation, 60)}<br />
            {content.metadata?.source_documents && (
              <><strong>Nguồn:</strong> {truncateText(content.metadata.source_documents.join(', '), 50)}<br /></>
            )}
          </div>
        );
      case 'multi_paragraph_reading':
        return (
          <div style={{ maxWidth: '400px', wordWrap: 'break-word' }}>
            <strong>Câu hỏi:</strong> {truncateText(content.question, 80)}<br />
            <strong>Câu trả lời:</strong> {truncateText(content.answer, 80)}<br />
            <strong>Lý do:</strong> {truncateText(content.reasoning, 60)}<br />
            {content.metadata?.source_documents && (
              <><strong>Nguồn:</strong> {truncateText(content.metadata.source_documents.join(', '), 50)}<br /></>
            )}
          </div>
        );
      case 'multi_hop_reasoning':
        return (
          <div style={{ maxWidth: '400px', wordWrap: 'break-word' }}>
            <strong>Câu hỏi:</strong> {truncateText(content.question, 80)}<br />
            <strong>Câu trả lời:</strong> {truncateText(content.answer, 80)}<br />
            <strong>Các bước:</strong> {truncateText(content.reasoning_steps?.join(' → '), 60)}<br />
            {content.metadata?.source_documents && (
              <><strong>Nguồn:</strong> {truncateText(content.metadata.source_documents.join(', '), 50)}<br /></>
            )}
          </div>
        );
      default:
        return <div style={{ maxWidth: '400px', wordWrap: 'break-word' }}>{truncateText(JSON.stringify(content), 100)}</div>;
    }
  };

  const previewColumns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 80,
    },
    {
      title: 'Loại',
      dataIndex: 'data_type',
      key: 'data_type',
      width: 80,
      render: (type) => (
        <Tag color={getDataTypeColor(type)}>
          {type.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Nội dung',
      key: 'content',
      render: (_, record) => renderPreviewContent(record),
    },
    {
      title: 'Trạng thái',
      key: 'status',
      width: 120,
      render: (_, record) => (
        <Tag color="green">
          <CheckCircleOutlined /> Đã duyệt
        </Tag>
      ),
    },
  ];

  const getExportableCount = () => {
    return dataPreview.length;
  };

  return (
    <div>
      <div className="page-header">
        <h1>Xuất Dữ Liệu Huấn Luyện</h1>
        <p>Xuất dữ liệu đã được gán nhãn thành file .jsonl để huấn luyện mô hình</p>
      </div>

      {/* Thống kê tổng quan */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Tổng Dữ Liệu Đã Sinh"
              value={stats?.total_generated || 0}
              prefix={<FileTextOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Đã Gán Nhãn"
              value={stats?.total_labeled || 0}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Sẵn Sàng Xuất"
              value={stats?.label_distribution?.accept || 0}
              prefix={<DownloadOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Cấu hình xuất */}
      <Card title="Cấu Hình Xuất Dữ Liệu" style={{ marginBottom: 24 }}>
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={12}>
              <div>
                <label style={{ display: 'block', marginBottom: 8, fontWeight: 'bold' }}>
                  Chọn Chủ Đề:
                </label>
                <div style={{ display: 'flex', gap: 8 }}>
                  <Select
                    placeholder="Tất cả chủ đề hoặc chọn chủ đề cụ thể"
                    style={{ flex: 1 }}
                    value={selectedTopic}
                    onChange={setSelectedTopic}
                    allowClear
                  >
                    {topics.map(topic => (
                      <Option key={topic.id} value={topic.id}>
                        {topic.name}
                      </Option>
                    ))}
                  </Select>
                  <Button 
                    onClick={loadTopics}
                    size="small"
                  >
                    🔄
                  </Button>
                </div>
              </div>
            </Col>
            
            <Col xs={24} sm={12}>
              <div>
                <label style={{ display: 'block', marginBottom: 8, fontWeight: 'bold' }}>
                  Loại Dữ Liệu:
                </label>
                <Select
                  style={{ width: '100%' }}
                  value={dataType}
                  onChange={setDataType}
                >
                  <Option value="word_matching">Word Matching - Khớp từ khóa</Option>
                  <Option value="concept_understanding">Concept Understanding - Hiểu khái niệm</Option>
                  <Option value="multi_paragraph_reading">Multi Paragraph Reading - Đọc hiểu đa đoạn</Option>
                  <Option value="multi_hop_reasoning">Multi Hop Reasoning - Suy luận đa bước</Option>
                </Select>
              </div>
            </Col>
          </Row>

          <Alert
            message="Lưu ý về định dạng xuất"
            description={
              <div>
                <p>Dữ liệu sẽ được xuất theo định dạng JSONL (JSON Lines), mỗi dòng là một object JSON.</p>
                <p>Chỉ các mẫu dữ liệu đã được gán nhãn "Chấp nhận" hoặc "Đã sửa" mới được xuất.</p>
                <p>File có thể được sử dụng trực tiếp với các framework huấn luyện như Transformers, OpenAI Fine-tuning, v.v.</p>
              </div>
            }
            type="info"
            showIcon
          />

          {selectedTopic && (
            <Descriptions title="Thông Tin Chủ Đề" bordered>
              <Descriptions.Item label="Tên chủ đề">
                {topics.find(t => t.id === selectedTopic)?.name}
              </Descriptions.Item>
              <Descriptions.Item label="Mô tả">
                {topics.find(t => t.id === selectedTopic)?.description}
              </Descriptions.Item>
              <Descriptions.Item label="Dữ liệu có thể xuất">
                <Tag color="green">{getExportableCount()} mẫu</Tag>
              </Descriptions.Item>
            </Descriptions>
          )}

          <div style={{ textAlign: 'center', marginTop: 16 }}>
            <Space>
              <Button
                onClick={() => selectedTopic && loadDataPreview()}
                disabled={!selectedTopic}
              >
                🔄 Tải lại preview
              </Button>
              
              <Button
                type="primary"
                size="large"
                icon={<DownloadOutlined />}
                loading={exporting}
                onClick={handleExport}
                disabled={getExportableCount() === 0}
              >
                {exporting ? 'Đang Xuất...' : `Xuất ${getExportableCount()} Mẫu Dữ Liệu`}
              </Button>
            </Space>
          </div>
        </Space>
      </Card>

      {/* Preview dữ liệu */}
      {selectedTopic && (
        <Card 
          title={`Preview Dữ Liệu Sẽ Xuất (${getExportableCount()} mẫu)`}
          extra={
            <Button 
              onClick={loadDataPreview}
              icon={<InfoCircleOutlined />}
            >
              Tải Lại
            </Button>
          }
        >
          {getExportableCount() > 0 ? (
            <Table
              columns={previewColumns}
              dataSource={dataPreview}
              rowKey="id"
              loading={loading}
              pagination={{ pageSize: 10 }}
              size="small"
              scroll={{ x: 800 }}
            />
          ) : (
            <div style={{ textAlign: 'center', padding: 40, color: '#999' }}>
              {selectedTopic 
                ? 'Chưa có dữ liệu nào được gán nhãn "Chấp nhận" cho chủ đề này'
                : 'Vui lòng chọn chủ đề để xem preview'
              }
            </div>
          )}
        </Card>
      )}

      {/* Hướng dẫn sử dụng */}
      <Card title="Hướng Dẫn Sử Dụng File Xuất" style={{ marginTop: 24 }}>
        <div>
          <h4>Định dạng Word Matching:</h4>
          <pre style={{ 
            background: '#f5f5f5', 
            padding: 12, 
            borderRadius: 4,
            overflowX: 'auto',
            whiteSpace: 'pre-wrap',
            wordWrap: 'break-word',
            fontSize: '12px'
          }}>
{`{"question": "Độ tuổi tối thiểu để lái xe ô tô là bao nhiêu?", "answer": "18 tuổi", "metadata": {"source_document": "Luật Giao thông"}}
{"question": "Ai có thể cấp giấy phép lái xe?", "answer": "Cơ quan có thẩm quyền", "metadata": {"source_document": "Nghị định 12"}}`}
          </pre>

          <h4>Định dạng Concept Understanding:</h4>
          <pre style={{ 
            background: '#f5f5f5', 
            padding: 12, 
            borderRadius: 4,
            overflowX: 'auto',
            whiteSpace: 'pre-wrap',
            wordWrap: 'break-word',
            fontSize: '12px'
          }}>
{`{"question": "Khái niệm giao thông đường bộ là gì?", "answer": "Hoạt động di chuyển người, hàng hóa bằng phương tiện giao thông", "explanation": "Định nghĩa chi tiết về giao thông đường bộ", "metadata": {"source_documents": ["Luật Giao thông", "Nghị định 12"]}}
{"question": "Nguyên tắc cấp giấy phép lái xe?", "answer": "Đúng tuổi, đủ sức khỏe, có kiến thức", "explanation": "Giải thích về các điều kiện cấp phép", "metadata": {"source_documents": ["Luật Giao thông"]}}`}
          </pre>

          <h4>Định dạng Multi Paragraph Reading:</h4>
          <pre style={{ 
            background: '#f5f5f5', 
            padding: 12, 
            borderRadius: 4,
            overflowX: 'auto',
            whiteSpace: 'pre-wrap',
            wordWrap: 'break-word',
            fontSize: '12px'
          }}>
{`{"question": "So sánh điều kiện cấp GPLX hạng A và hạng B", "answer": "Hạng A: từ 16 tuổi, Hạng B: từ 18 tuổi", "reasoning": "Dựa vào nhiều điều luật khác nhau", "metadata": {"source_documents": ["Luật Giao thông", "Nghị định 12", "Thông tư 58"]}}
{"question": "Quy trình đào tạo và sát hạch lái xe", "answer": "Đào tạo lý thuyết → thực hành → sát hạch", "reasoning": "Tổng hợp từ các quy định về đào tạo", "metadata": {"source_documents": ["Nghị định 12", "Thông tư 58"]}}`}
          </pre>

          <h4>Định dạng Multi Hop Reasoning:</h4>
          <pre style={{ 
            background: '#f5f5f5', 
            padding: 12, 
            borderRadius: 4,
            overflowX: 'auto',
            whiteSpace: 'pre-wrap',
            wordWrap: 'break-word',
            fontSize: '12px'
          }}>
{`{"question": "Một người 17 tuổi muốn lái xe cần làm gì?", "answer": "Chờ đủ 18 tuổi hoặc học hạng A1", "reasoning_steps": ["Kiểm tra độ tuổi", "Xem loại xe muốn lái", "Tìm quy định phù hợp"], "metadata": {"source_documents": ["Luật Giao thông", "Nghị định 12", "Thông tư 58"]}}
{"question": "Chi phí và thời gian hoàn tất giấy phép lái xe B1", "answer": "Khoảng 3-6 tháng, chi phí 8-12 triệu", "reasoning_steps": ["Tính thời gian đào tạo", "Cộng thời gian chờ sát hạch", "Tổng hợp chi phí các khâu"], "metadata": {"source_documents": ["Nghị định 12", "Thông tư 58", "Quyết định phí"]}}`}
          </pre>

          <Divider />
          
          <Alert
            message="Sử dụng với Python"
            description={
              <pre style={{ 
                margin: 0,
                overflowX: 'auto',
                whiteSpace: 'pre-wrap',
                wordWrap: 'break-word',
                fontSize: '12px'
              }}>
{`import jsonlines

# Đọc file JSONL cho Word Matching
with jsonlines.open('word_matching_data.jsonl') as reader:
    for obj in reader:
        print(f"Q: {obj['question']}")
        print(f"A: {obj['answer']}")
        
# Đọc file JSONL cho Multi Hop Reasoning  
with jsonlines.open('multi_hop_reasoning_data.jsonl') as reader:
    for obj in reader:
        print(f"Q: {obj['question']}")
        print(f"Steps: {' → '.join(obj['reasoning_steps'])}")
        print(f"A: {obj['answer']}")

# Hoặc với pandas
import pandas as pd
df = pd.read_json('word_matching_data.jsonl', lines=True)`}
              </pre>
            }
            type="info"
          />
        </div>
      </Card>
    </div>
  );
};

const getDataTypeColor = (type) => {
  switch (type) {
    case 'word_matching': return 'blue';
    case 'concept_understanding': return 'green';
    case 'multi_paragraph_reading': return 'purple';
    case 'multi_hop_reasoning': return 'orange';
    default: return 'default';
  }
};

export default DataExport;
