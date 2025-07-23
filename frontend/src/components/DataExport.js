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

const { Option } = Select;

const DataExport = () => {
  const [topics, setTopics] = useState([]);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [dataType, setDataType] = useState('sft');
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
    
    switch (item.data_type) {
      case 'sft':
        return (
          <div>
            <strong>Instruction:</strong> {content.instruction}<br />
            <strong>Output:</strong> {content.output}
          </div>
        );
      case 'cot':
        return (
          <div>
            <strong>Instruction:</strong> {content.instruction}<br />
            <strong>Steps:</strong> {content.reasoning_steps?.join(' → ')}<br />
            <strong>Answer:</strong> {content.final_answer}
          </div>
        );
      case 'rlhf':
        return (
          <div>
            <strong>Prompt:</strong> {content.prompt}<br />
            <strong>Response A:</strong> {content.response_a?.substring(0, 100)}...<br />
            <strong>Response B:</strong> {content.response_b?.substring(0, 100)}...<br />
            <strong>Preferred:</strong> {content.preferred}
          </div>
        );
      default:
        return JSON.stringify(content);
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
                  <Option value="sft">SFT - Supervised Fine-Tuning</Option>
                  <Option value="cot">CoT - Chain-of-Thought</Option>
                  <Option value="rlhf">RLHF - Human Feedback</Option>
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
          <h4>Định dạng SFT:</h4>
          <pre style={{ background: '#f5f5f5', padding: 12, borderRadius: 4 }}>
{`{"instruction": "Câu hỏi", "output": "Câu trả lời"}
{"instruction": "Câu hỏi khác", "output": "Câu trả lời khác"}`}
          </pre>

          <h4>Định dạng CoT:</h4>
          <pre style={{ background: '#f5f5f5', padding: 12, borderRadius: 4 }}>
{`{"instruction": "Câu hỏi", "reasoning_steps": ["Bước 1", "Bước 2"], "final_answer": "Kết luận"}
{"instruction": "Câu hỏi khác", "reasoning_steps": ["Bước A", "Bước B"], "final_answer": "Kết luận khác"}`}
          </pre>

          <h4>Định dạng RLHF:</h4>
          <pre style={{ background: '#f5f5f5', padding: 12, borderRadius: 4 }}>
{`{"prompt": "Câu hỏi", "response_a": "Phản hồi A", "response_b": "Phản hồi B", "preferred": "A"}
{"prompt": "Câu hỏi khác", "response_a": "Phản hồi A", "response_b": "Phản hồi B", "preferred": "B"}`}
          </pre>

          <Divider />
          
          <Alert
            message="Sử dụng với Python"
            description={
              <pre style={{ margin: 0 }}>
{`import jsonlines

# Đọc file JSONL
with jsonlines.open('sft_data.jsonl') as reader:
    for obj in reader:
        print(obj['instruction'], obj['output'])

# Hoặc với pandas
import pandas as pd
df = pd.read_json('sft_data.jsonl', lines=True)`}
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
    case 'sft': return 'blue';
    case 'cot': return 'purple';
    case 'rlhf': return 'orange';
    default: return 'default';
  }
};

export default DataExport;
