import React, { useState, useEffect } from 'react';
import {
  Card,
  Select,
  Button,
  InputNumber,
  Radio,
  Space,
  Alert,
  Spin,
  message,
  Divider,
  Progress,
  List,
  Tag
} from 'antd';
import {
  PlayCircleOutlined,
  ReloadOutlined,
  EyeOutlined
} from '@ant-design/icons';
import apiService from '../services/api';

const { Option } = Select;

const DataGeneration = () => {
  const [topics, setTopics] = useState([]);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [dataType, setDataType] = useState('sft');
  const [numSamples, setNumSamples] = useState(10);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [generatedData, setGeneratedData] = useState([]);

  useEffect(() => {
    loadTopics();
  }, []);

  useEffect(() => {
    if (selectedTopic) {
      loadGeneratedData(selectedTopic);
    }
  }, [selectedTopic]);

  const loadTopics = async () => {
    try {
      setLoading(true);
      const response = await apiService.getTopics();
      // Chỉ hiển thị topics có văn bản để sinh dữ liệu
      const topicsWithText = response.data.filter(topic => topic.legal_text);
      setTopics(topicsWithText);
      
      if (response.data.length > topicsWithText.length) {
        message.info(`Có ${response.data.length - topicsWithText.length} chủ đề chưa có văn bản luật`);
      }
    } catch (error) {
      message.error('Không thể tải danh sách chủ đề');
    } finally {
      setLoading(false);
    }
  };

  const loadGeneratedData = async (topicId) => {
    try {
      const response = await apiService.getGeneratedData(topicId);
      setGeneratedData(response.data);
    } catch (error) {
      console.error('Error loading generated data:', error);
    }
  };

  const handleGenerate = async () => {
    if (!selectedTopic) {
      message.warning('Vui lòng chọn chủ đề');
      return;
    }

    try {
      setGenerating(true);
      
      const response = await apiService.generateData({
        topic_id: selectedTopic,
        data_type: dataType,
        num_samples: numSamples
      });

      message.success(response.data.message);
      loadGeneratedData(selectedTopic);
    } catch (error) {
      message.error('Không thể sinh dữ liệu. Vui lòng thử lại.');
    } finally {
      setGenerating(false);
    }
  };

  const getDataTypeDescription = (type) => {
    switch (type) {
      case 'sft':
        return {
          title: 'SFT (Supervised Fine-Tuning)',
          description: 'Tạo cặp instruction-output đơn giản để huấn luyện mô hình trả lời câu hỏi pháp lý',
          example: 'Instruction: "Thời hạn GPLX hạng A1 là bao lâu?"\nOutput: "Theo Thông tư 12/2017, GPLX hạng A1 có giá trị không thời hạn."'
        };
      case 'cot':
        return {
          title: 'CoT (Chain-of-Thought)',
          description: 'Tạo dữ liệu với các bước suy luận rõ ràng để mô hình học cách phân tích từng bước',
          example: 'Instruction: "Người 17 tuổi có được thi GPLX không?"\nReasoning: Bước 1 → Bước 2 → Bước 3\nAnswer: "Không"'
        };
      case 'rlhf':
        return {
          title: 'RLHF (Reinforcement Learning from Human Feedback)',
          description: 'Tạo cặp câu trả lời A/B để con người đánh giá và cải thiện chất lượng mô hình',
          example: 'Prompt: "Tư vấn thủ tục đổi GPLX"\nResponse A: Đầy đủ, chính xác\nResponse B: Thiếu sót'
        };
      default:
        return { title: '', description: '', example: '' };
    }
  };

  const renderDataPreview = (item) => {
    const content = typeof item.content === 'string' ? JSON.parse(item.content) : item.content;
    
    switch (item.data_type) {
      case 'sft':
        return (
          <div>
            <p><strong>Instruction:</strong> {content.instruction}</p>
            <p><strong>Output:</strong> {content.output}</p>
          </div>
        );
      case 'cot':
        return (
          <div>
            <p><strong>Instruction:</strong> {content.instruction}</p>
            <p><strong>Reasoning Steps:</strong></p>
            <ul className="reasoning-steps">
              {content.reasoning_steps?.map((step, idx) => (
                <li key={idx}>{step}</li>
              ))}
            </ul>
            <p><strong>Final Answer:</strong> {content.final_answer}</p>
          </div>
        );
      case 'rlhf':
        return (
          <div>
            <p><strong>Prompt:</strong> {content.prompt}</p>
            <div className="response-comparison">
              <div>
                <strong>Response A:</strong>
                <p>{content.response_a}</p>
              </div>
              <div>
                <strong>Response B:</strong>
                <p>{content.response_b}</p>
              </div>
            </div>
            <p><strong>Preferred:</strong> <Tag color="green">Response {content.preferred}</Tag></p>
          </div>
        );
      default:
        return <pre>{JSON.stringify(content, null, 2)}</pre>;
    }
  };

  const dataTypeInfo = getDataTypeDescription(dataType);

  return (
    <div>
      <div className="page-header">
        <h1>Sinh Dữ Liệu Huấn Luyện</h1>
        <p>Tạo dữ liệu huấn luyện tự động từ văn bản pháp luật</p>
      </div>

      <Card title="Cấu Hình Sinh Dữ Liệu" style={{ marginBottom: 24 }}>
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          {/* Chọn chủ đề */}
          <div>
            <label style={{ display: 'block', marginBottom: 8, fontWeight: 'bold' }}>
              Chọn Chủ Đề:
            </label>
            <Select
              placeholder="Chọn chủ đề có văn bản luật"
              style={{ width: '100%' }}
              value={selectedTopic}
              onChange={setSelectedTopic}
              loading={loading}
            >
              {topics.map(topic => (
                <Option key={topic.id} value={topic.id}>
                  {topic.name} - {topic.description}
                </Option>
              ))}
            </Select>
          </div>

          {/* Chọn loại dữ liệu */}
          <div>
            <label style={{ display: 'block', marginBottom: 8, fontWeight: 'bold' }}>
              Loại Dữ Liệu:
            </label>
            <Radio.Group value={dataType} onChange={e => setDataType(e.target.value)}>
              <Space direction="vertical">
                <Radio value="sft">SFT - Supervised Fine-Tuning</Radio>
                <Radio value="cot">CoT - Chain-of-Thought</Radio>
                <Radio value="rlhf">RLHF - Human Feedback</Radio>
              </Space>
            </Radio.Group>
          </div>

          {/* Số lượng mẫu */}
          <div>
            <label style={{ display: 'block', marginBottom: 8, fontWeight: 'bold' }}>
              Số Lượng Mẫu:
            </label>
            <InputNumber
              min={1}
              max={50}
              value={numSamples}
              onChange={setNumSamples}
              style={{ width: 120 }}
            />
            <span style={{ marginLeft: 8, color: '#666' }}>
              (Khuyến nghị: 5-20 mẫu cho mỗi lần sinh)
            </span>
          </div>

          {/* Mô tả loại dữ liệu */}
          <Alert
            message={dataTypeInfo.title}
            description={
              <div>
                <p>{dataTypeInfo.description}</p>
                <Divider />
                <p><strong>Ví dụ:</strong></p>
                <pre style={{ whiteSpace: 'pre-wrap', fontSize: '12px' }}>
                  {dataTypeInfo.example}
                </pre>
              </div>
            }
            type="info"
            showIcon
          />

          {/* Nút sinh dữ liệu */}
          <div style={{ textAlign: 'center' }}>
            <Button
              type="primary"
              size="large"
              icon={<PlayCircleOutlined />}
              loading={generating}
              onClick={handleGenerate}
              disabled={!selectedTopic}
            >
              {generating ? 'Đang Sinh Dữ Liệu...' : 'Sinh Dữ Liệu'}
            </Button>
          </div>
        </Space>
      </Card>

      {/* Hiển thị dữ liệu đã sinh */}
      {selectedTopic && (
        <Card
          title={`Dữ Liệu Đã Sinh (${generatedData.length} mẫu)`}
          extra={
            <Button
              icon={<ReloadOutlined />}
              onClick={() => loadGeneratedData(selectedTopic)}
            >
              Tải Lại
            </Button>
          }
        >
          {generatedData.length > 0 ? (
            <List
              dataSource={generatedData}
              renderItem={(item, index) => (
                <List.Item key={item.id}>
                  <Card
                    size="small"
                    title={
                      <Space>
                        <Tag color={getDataTypeColor(item.data_type)}>
                          {item.data_type.toUpperCase()}
                        </Tag>
                        <span>Mẫu #{index + 1}</span>
                        {item.is_labeled && <Tag color="green">Đã gán nhãn</Tag>}
                      </Space>
                    }
                    style={{ width: '100%' }}
                  >
                    {renderDataPreview(item)}
                  </Card>
                </List.Item>
              )}
              pagination={{ pageSize: 5 }}
            />
          ) : (
            <div style={{ textAlign: 'center', padding: 40, color: '#999' }}>
              Chưa có dữ liệu nào được sinh ra cho chủ đề này
            </div>
          )}
        </Card>
      )}
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

export default DataGeneration;
