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
  const [dataType, setDataType] = useState('word_matching');
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
      
      // Filter topics có documents hoặc legal_text
      const readyTopics = response.data.filter(topic => 
        (topic.document_count && topic.document_count > 0) || 
        (topic.legal_text && topic.legal_text.trim().length > 0)
      );
      
      setTopics(readyTopics);
      
      if (response.data.length > readyTopics.length) {
        const missingCount = response.data.length - readyTopics.length;
        message.info(
          `Có ${missingCount} chủ đề chưa có tài liệu. Hãy thêm tài liệu ở trang "Quản lý Chủ đề" trước khi sinh dữ liệu.`
        );
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
      case 'word_matching':
        return {
          title: 'Word Matching',
          description: 'Câu hỏi đơn giản có thể trả lời bằng tìm kiếm từ khóa trực tiếp trong văn bản',
          example: 'Question: "Độ tuổi tối thiểu để thi GPLX hạng A1 là bao nhiêu?"\nAnswer: "18 tuổi"\nEvidence: "Điều 60: Hạng A1: đủ 18 tuổi"'
        };
      case 'concept_understanding':
        return {
          title: 'Concept Understanding',
          description: 'Yêu cầu hiểu ý nghĩa các khái niệm và thuật ngữ pháp lý để trả lời',
          example: 'Question: "Thế nào là vi phạm về GPLX?"\nAnswer: "Vi phạm bao gồm lái xe khi không có GPLX, GPLX hết hạn..."\nConcepts: ["vi phạm", "GPLX", "hành vi bị cấm"]'
        };
      case 'multi_paragraph_reading':
        return {
          title: 'Multi-Paragraph Reading',
          description: 'Cần đọc và tổng hợp thông tin từ nhiều đoạn văn khác nhau',
          example: 'Question: "Quy trình cấp đổi GPLX như thế nào?"\nAnswer: "Gồm 3 bước: nộp hồ sơ, kiểm tra, cấp mới"\nParagraphs: ["Điều về hồ sơ", "Điều về kiểm tra", "Điều về cấp phát"]'
        };
      case 'multi_hop_reasoning':
        return {
          title: 'Multi-Hop Reasoning',
          description: 'Phức tạp nhất, cần nhiều bước suy luận logic liên tiếp để trả lời',
          example: 'Question: "Người nước ngoài muốn lái xe tại VN cần làm gì?"\nReasoning: ["Xác định loại GPLX", "Kiểm tra hiệp định", "Thủ tục chuyển đổi", "Điều kiện cư trú"]\nAnswer: "Tùy thuộc vào quốc tịch và loại GPLX..."'
        };
      // Backward compatibility
      case 'sft':
        return {
          title: 'SFT (Word Matching)',
          description: 'Tạo cặp instruction-output đơn giản (tương đương Word Matching)',
          example: 'Instruction: "Thời hạn GPLX hạng A1 là bao lâu?"\nOutput: "Theo Thông tư 12/2017, GPLX hạng A1 có giá trị không thời hạn."'
        };
      case 'cot':
        return {
          title: 'CoT (Concept Understanding)',
          description: 'Tạo dữ liệu với hiểu biết khái niệm (tương đương Concept Understanding)',
          example: 'Instruction: "Người 17 tuổi có được thi GPLX không?"\nConcepts: Độ tuổi tối thiểu\nAnswer: "Không"'
        };
      case 'rlhf':
        return {
          title: 'RLHF (Multi-Hop Reasoning)',
          description: 'Tạo dữ liệu phức tạp (tương đương Multi-Hop Reasoning)',
          example: 'Prompt: "Tư vấn thủ tục đổi GPLX"\nReasoning: Nhiều bước\nAnswer: Kết luận'
        };
      default:
        return { title: '', description: '', example: '' };
    }
  };

  const renderDataPreview = (item) => {
    const content = typeof item.content === 'string' ? JSON.parse(item.content) : item.content;
    
    // Hiển thị format đơn giản: chỉ 3 trường (question, answer, difficulty)
    const getColorByType = (type) => {
      switch (type) {
        case 'word_matching': return 'blue';
        case 'concept_understanding': return 'green';
        case 'multi_paragraph_reading': return 'orange';
        case 'multi_hop_reasoning': return 'red';
        case 'sft': return 'blue';
        case 'cot': return 'green';
        case 'rlhf': return 'red';
        default: return 'default';
      }
    };

    const getDisplayType = (type) => {
      switch (type) {
        case 'word_matching': return 'Word Matching';
        case 'concept_understanding': return 'Concept Understanding';
        case 'multi_paragraph_reading': return 'Multi-Paragraph Reading';
        case 'multi_hop_reasoning': return 'Multi-Hop Reasoning';
        case 'sft': return 'SFT (Legacy)';
        case 'cot': return 'CoT (Legacy)';
        case 'rlhf': return 'RLHF (Legacy)';
        default: return type;
      }
    };

    return (
      <div>
        <p><strong>Question:</strong> {content.question || content.instruction || content.prompt}</p>
        <p><strong>Answer:</strong> {content.answer || content.final_answer || content.output}</p>
        <Tag color={getColorByType(item.data_type)}>
          {content.difficulty || 'Unknown'} - {getDisplayType(item.data_type)}
        </Tag>
      </div>
    );
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
              placeholder="Chọn chủ đề có tài liệu"
              style={{ width: '100%' }}
              value={selectedTopic}
              onChange={setSelectedTopic}
              loading={loading}
              optionLabelProp="label"
            >
              {topics.map(topic => (
                <Option 
                  key={topic.id} 
                  value={topic.id}
                  label={topic.name}
                >
                  <div>
                    <div><strong>{topic.name}</strong></div>
                    <div style={{ fontSize: '12px', color: '#666' }}>
                      {topic.document_count > 0 
                        ? `${topic.document_count} tài liệu` 
                        : 'Văn bản trực tiếp'
                      } • {topic.description?.substring(0, 50)}...
                    </div>
                  </div>
                </Option>
              ))}
            </Select>
            
            {selectedTopic && (
              <div style={{ 
                marginTop: 8, 
                padding: 12, 
                background: '#f0f9ff', 
                border: '1px solid #bae6fd', 
                borderRadius: 6,
                fontSize: '13px'
              }}>
                {(() => {
                  const topic = topics.find(t => t.id === selectedTopic);
                  return topic ? (
                    <div>
                      <strong>📑 Nguồn tài liệu:</strong>
                      {topic.document_count > 0 ? (
                        <div style={{ marginTop: 4 }}>
                          {topic.documents.map(doc => (
                            <div key={doc.id}>• {doc.title}</div>
                          ))}
                        </div>
                      ) : (
                        <span> Văn bản được nhập trực tiếp</span>
                      )}
                    </div>
                  ) : null;
                })()}
              </div>
            )}
          </div>

          {/* Chọn loại dữ liệu */}
          <div>
            <label style={{ display: 'block', marginBottom: 8, fontWeight: 'bold' }}>
              Loại Dữ Liệu:
            </label>
            <Radio.Group value={dataType} onChange={e => setDataType(e.target.value)}>
              <Space direction="vertical">
                <Radio value="word_matching">
                  <strong>Word Matching</strong> - Tìm từ khóa đơn giản
                  <div style={{ fontSize: '12px', color: '#666', marginLeft: 20 }}>
                    Câu hỏi có thể trả lời bằng tìm kiếm trực tiếp trong văn bản
                  </div>
                </Radio>
                <Radio value="concept_understanding">
                  <strong>Concept Understanding</strong> - Hiểu khái niệm pháp lý
                  <div style={{ fontSize: '12px', color: '#666', marginLeft: 20 }}>
                    Yêu cầu hiểu ý nghĩa các thuật ngữ và khái niệm pháp luật
                  </div>
                </Radio>
                <Radio value="multi_paragraph_reading">
                  <strong>Multi-Paragraph Reading</strong> - Đọc nhiều đoạn văn
                  <div style={{ fontSize: '12px', color: '#666', marginLeft: 20 }}>
                    Cần tập hợp thông tin từ nhiều đoạn văn khác nhau
                  </div>
                </Radio>
                <Radio value="multi_hop_reasoning">
                  <strong>Multi-Hop Reasoning</strong> - Suy luận nhiều bước
                  <div style={{ fontSize: '12px', color: '#666', marginLeft: 20 }}>
                    Phức tạp nhất, cần nhiều bước suy luận logic liên tiếp
                  </div>
                </Radio>
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
