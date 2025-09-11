import React, { useState, useEffect } from 'react';
import {
  Card,
  Select,
  Button,
  Radio,
  Input,
  Space,
  message,
  List,
  Tag,
  Modal,
  Form,
  Divider,
  Alert,
  Progress
} from 'antd';
import {
  CheckOutlined,
  CloseOutlined,
  EditOutlined,
  EyeOutlined,
  FilterOutlined
} from '@ant-design/icons';
import apiService from '../services/api';

const { Option } = Select;
const { TextArea } = Input;

const DataLabeling = () => {
  const [topics, setTopics] = useState([]);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [dataList, setDataList] = useState([]);
  const [filteredData, setFilteredData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [selectedItem, setSelectedItem] = useState(null);
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterDataType, setFilterDataType] = useState('all');
  const [form] = Form.useForm();

  useEffect(() => {
    loadTopics();
  }, []);

  useEffect(() => {
    if (selectedTopic) {
      loadDataForLabeling(selectedTopic);
    }
  }, [selectedTopic]);

  useEffect(() => {
    applyFilters();
  }, [dataList, filterStatus, filterDataType]);

  const loadTopics = async () => {
    try {
      const response = await apiService.getTopics();
      // Lấy tất cả topics, không filter
      setTopics(response.data);
    } catch (error) {
      message.error('Không thể tải danh sách chủ đề');
    }
  };

  const loadDataForLabeling = async (topicId) => {
    try {
      setLoading(true);
      const response = await apiService.getGeneratedData(topicId);
      setDataList(response.data);
    } catch (error) {
      message.error('Không thể tải dữ liệu');
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  const applyFilters = () => {
    let filtered = [...dataList];

    if (filterStatus !== 'all') {
      if (filterStatus === 'labeled') {
        filtered = filtered.filter(item => item.is_labeled);
      } else if (filterStatus === 'unlabeled') {
        filtered = filtered.filter(item => !item.is_labeled);
      }
    }

    if (filterDataType !== 'all') {
      filtered = filtered.filter(item => item.data_type === filterDataType);
    }

    setFilteredData(filtered);
  };

  const handleQuickLabel = async (dataId, label) => {
    try {
      await apiService.labelData({
        data_id: dataId,
        label: label,
        notes: `Quick label: ${label}`
      });
      
      message.success(`Đã ${getLabelText(label).toLowerCase()} mẫu dữ liệu`);
      loadDataForLabeling(selectedTopic);
    } catch (error) {
      message.error('Không thể gán nhãn');
    }
  };

  const handleDetailedLabel = (item) => {
    setSelectedItem(item);
    form.resetFields();
    setModalVisible(true);
  };

  const handleSubmitLabel = async (values) => {
    try {
      const payload = {
        data_id: selectedItem.id,
        label: values.label,
        notes: values.notes || '',
      };

      if (values.label === 'modify' && values.modified_content) {
        // Parse and update the content
        const originalContent = typeof selectedItem.content === 'string' 
          ? JSON.parse(selectedItem.content) 
          : selectedItem.content;
        
        const modifiedContent = {
          ...originalContent,
          ...values.modified_content
        };
        
        payload.modified_content = JSON.stringify(modifiedContent);
      }

      await apiService.labelData(payload);
      message.success('Gán nhãn thành công!');
      setModalVisible(false);
      loadDataForLabeling(selectedTopic);
    } catch (error) {
      message.error('Không thể gán nhãn');
    }
  };

  const renderDataContent = (item) => {
    const content = typeof item.content === 'string' ? JSON.parse(item.content) : item.content;
    
    // Render sources information nếu có
    const renderSources = (sources) => {
      if (!sources || !Array.isArray(sources) || sources.length === 0) {
        return null;
      }
      
      // Group sources by document để hiển thị rõ ràng hơn
      const sourcesByDoc = sources.reduce((acc, source) => {
        const docTitle = source.document_title;
        if (!acc[docTitle]) {
          acc[docTitle] = [];
        }
        acc[docTitle].push(source);
        return acc;
      }, {});
      
      return (
        <div className="sources-info" style={{ marginTop: 12, padding: 8, backgroundColor: '#f5f5f5', borderRadius: 4 }}>
          <strong>📚 Nguồn tham chiếu ({sources.length} điều):</strong>
          {Object.entries(sourcesByDoc).map(([docTitle, docSources], docIndex) => (
            <div key={docIndex} style={{ marginTop: 6 }}>
              <div style={{ fontWeight: '500', color: '#333', fontSize: '0.9em' }}>
                📄 {docTitle}
              </div>
              {docSources.map((source, sourceIndex) => {
                // Trích xuất thông tin từ unit_path (format: "Document > Điều X > Khoản Y > Điểm Z")
                // hoặc fallback về article_number/article_title cho data cũ
                let displayText = '';
                
                if (source.unit_path) {
                  // Format mới: trích xuất từ unit_path, chỉ lấy từ "Điều" trở đi
                  const pathParts = source.unit_path.split(' > ');
                  const dieuIndex = pathParts.findIndex(part => part.includes('Điều'));
                  
                  if (dieuIndex !== -1) {
                    const relevantParts = pathParts.slice(dieuIndex);
                    let displayParts = [relevantParts[0]]; // Điều X
                    
                    // Thêm Khoản nếu có và không phải N/A
                    if (relevantParts.length >= 2 && !relevantParts[1].includes('N/A')) {
                      displayParts.push(relevantParts[1]);
                    }
                    
                    // Thêm Điểm nếu có và không phải N/A
                    if (relevantParts.length >= 3 && !relevantParts[2].includes('N/A')) {
                      displayParts.push(relevantParts[2]);
                    }
                    
                    displayText = displayParts.join(' > ');
                  } else {
                    displayText = source.unit_path;
                  }
                } else if (source.article_number) {
                  // Format cũ: fallback
                  displayText = `Điều ${source.article_number}`;
                  if (source.article_title) {
                    displayText += `: ${source.article_title}`;
                  }
                } else {
                  displayText = 'N/A';
                }
                
                return (
                  <div key={sourceIndex} style={{ marginTop: 2, marginLeft: 16, fontSize: '0.85em' }}>
                    <Tag color="blue" size="small">{displayText}</Tag>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      );
    };
    
    switch (item.data_type) {
      case 'word_matching':
        return (
          <div className="data-item">
            <div className="instruction-text">
              <strong>Question:</strong> {content.question}
            </div>
            <div className="output-text">
              <strong>Answer:</strong> {content.answer}
            </div>
            {renderSources(content.sources)}
          </div>
        );
      case 'concept_understanding':
        return (
          <div className="data-item">
            <div className="instruction-text">
              <strong>Question:</strong> {content.question}
            </div>
            <div className="output-text">
              <strong>Answer:</strong> {content.answer}
            </div>
            {renderSources(content.sources)}
          </div>
        );
      case 'multi_paragraph_reading':
        return (
          <div className="data-item">
            <div className="instruction-text">
              <strong>Question:</strong> {content.question}
            </div>
            <div className="output-text">
              <strong>Answer:</strong> {content.answer}
            </div>
            {renderSources(content.sources)}
          </div>
        );
      case 'multi_hop_reasoning':
        return (
          <div className="data-item">
            <div className="instruction-text">
              <strong>Question:</strong> {content.question}
            </div>
            <div className="output-text">
              <strong>Answer:</strong> {content.answer}
            </div>
            {renderSources(content.sources)}
          </div>
        );

      default:
        return (
          <div className="data-item">
            <pre style={{whiteSpace: 'pre-wrap', wordWrap: 'break-word'}}>
              {JSON.stringify(content, null, 2)}
            </pre>
            {renderSources(content.sources)}
          </div>
        );
    }
  };

  const renderModifyForm = () => {
    if (!selectedItem) return null;

    const content = typeof selectedItem.content === 'string' 
      ? JSON.parse(selectedItem.content) 
      : selectedItem.content;

    switch (selectedItem.data_type) {
      case 'word_matching':
        return (
          <div>
            <Form.Item name={['modified_content', 'question']} label="Question">
              <TextArea rows={2} defaultValue={content.question} />
            </Form.Item>
            <Form.Item name={['modified_content', 'answer']} label="Answer">
              <TextArea rows={3} defaultValue={content.answer} />
            </Form.Item>
          </div>
        );
      case 'concept_understanding':
        return (
          <div>
            <Form.Item name={['modified_content', 'question']} label="Question">
              <TextArea rows={2} defaultValue={content.question} />
            </Form.Item>
            <Form.Item name={['modified_content', 'answer']} label="Answer">
              <TextArea rows={3} defaultValue={content.answer} />
            </Form.Item>
          </div>
        );
      case 'multi_paragraph_reading':
        return (
          <div>
            <Form.Item name={['modified_content', 'question']} label="Question">
              <TextArea rows={2} defaultValue={content.question} />
            </Form.Item>
            <Form.Item name={['modified_content', 'answer']} label="Answer">
              <TextArea rows={3} defaultValue={content.answer} />
            </Form.Item>
          </div>
        );
      case 'multi_hop_reasoning':
        return (
          <div>
            <Form.Item name={['modified_content', 'question']} label="Question">
              <TextArea rows={2} defaultValue={content.question} />
            </Form.Item>
            <Form.Item name={['modified_content', 'answer']} label="Answer">
              <TextArea rows={4} defaultValue={content.answer} />
            </Form.Item>
          </div>
        );
      default:
        return (
          <Form.Item name="modified_content" label="Modified Content">
            <TextArea 
              rows={8} 
              defaultValue={JSON.stringify(content, null, 2)}
              placeholder="Edit as JSON..."
            />
          </Form.Item>
        );
    }
  };

  const getStatusProgress = () => {
    if (dataList.length === 0) return 0;
    const labeledCount = dataList.filter(item => item.is_labeled).length;
    return Math.round((labeledCount / dataList.length) * 100);
  };

  return (
    <div>
      <div className="page-header">
        <h1>Gán Nhãn Dữ Liệu</h1>
        <p>Duyệt và gán nhãn cho dữ liệu đã sinh ra</p>
      </div>

      {/* Filters */}
      <Card title="Bộ Lọc" style={{ marginBottom: 24 }}>
        <Space wrap>
          <div>
            <label>Chủ đề:</label>
            <Select
              placeholder="Chọn chủ đề"
              style={{ width: 200, marginLeft: 8 }}
              value={selectedTopic}
              onChange={setSelectedTopic}
            >
              {topics.map(topic => (
                <Option key={topic.id} value={topic.id}>
                  {topic.name}
                </Option>
              ))}
            </Select>
            <Button 
              onClick={loadTopics}
              style={{ marginLeft: 8 }}
              size="small"
            >
              🔄
            </Button>
          </div>

          <div>
            <label>Trạng thái:</label>
            <Select
              style={{ width: 150, marginLeft: 8 }}
              value={filterStatus}
              onChange={setFilterStatus}
            >
              <Option value="all">Tất cả</Option>
              <Option value="unlabeled">Chưa gán nhãn</Option>
              <Option value="labeled">Đã gán nhãn</Option>
            </Select>
          </div>

          <div>
            <label>Loại dữ liệu:</label>
            <Select
              style={{ width: 180, marginLeft: 8 }}
              value={filterDataType}
              onChange={setFilterDataType}
            >
              <Option value="all">Tất cả</Option>
              <Option value="word_matching">Word Matching</Option>
              <Option value="concept_understanding">Concept Understanding</Option>
              <Option value="multi_paragraph_reading">Multi-Paragraph Reading</Option>
              <Option value="multi_hop_reasoning">Multi-Hop Reasoning</Option>
            </Select>
          </div>
        </Space>

        <div style={{ marginTop: 16 }}>
          <Button 
            onClick={() => selectedTopic && loadDataForLabeling(selectedTopic)}
            disabled={!selectedTopic}
          >
            🔄 Tải lại dữ liệu
          </Button>
        </div>

        {selectedTopic && (
          <div style={{ marginTop: 16 }}>
            <Progress
              percent={getStatusProgress()}
              status="active"
              format={(percent) => `${percent}% hoàn thành`}
            />
          </div>
        )}
      </Card>

      {/* Data List */}
      {selectedTopic && (
        <Card title={`Dữ liệu cần gán nhãn (${filteredData.length} mẫu)`}>
          {filteredData.length > 0 ? (
            <List
              dataSource={filteredData}
              loading={loading}
              renderItem={(item, index) => (
                <List.Item key={item.id}>
                  <Card
                    size="small"
                    style={{ width: '100%' }}
                    className={`data-card ${item.is_labeled ? 'labeled' : ''}`}
                    title={
                      <Space>
                        <Tag color={getDataTypeColor(item.data_type)}>
                          {item.data_type.toUpperCase()}
                        </Tag>
                        <span>Mẫu #{index + 1}</span>
                        {item.is_labeled && <Tag color="green">Đã gán nhãn</Tag>}
                      </Space>
                    }
                  >
                    {renderDataContent(item)}
                    
                    <div className="label-actions">
                      <Space>
                        <Button
                          type="primary"
                          icon={<CheckOutlined />}
                          size="small"
                          onClick={() => handleQuickLabel(item.id, 'accept')}
                          disabled={item.is_labeled}
                        >
                          Chấp nhận
                        </Button>
                        <Button
                          danger
                          icon={<CloseOutlined />}
                          size="small"
                          onClick={() => handleQuickLabel(item.id, 'reject')}
                          disabled={item.is_labeled}
                        >
                          Từ chối
                        </Button>
                        <Button
                          icon={<EditOutlined />}
                          size="small"
                          onClick={() => handleDetailedLabel(item)}
                        >
                          Sửa & Gán nhãn
                        </Button>
                      </Space>
                    </div>
                  </Card>
                </List.Item>
              )}
              pagination={{ pageSize: 5 }}
            />
          ) : (
            <div style={{ textAlign: 'center', padding: 40, color: '#999' }}>
              {dataList.length === 0 
                ? 'Chưa có dữ liệu nào cho chủ đề này'
                : 'Không có dữ liệu phù hợp với bộ lọc'
              }
            </div>
          )}
        </Card>
      )}

      {/* Detail Label Modal */}
      <Modal
        title="Gán Nhãn Chi Tiết"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
        width={800}
      >
        {selectedItem && (
          <Form
            form={form}
            layout="vertical"
            onFinish={handleSubmitLabel}
          >
            <Alert
              message="Dữ liệu gốc"
              description={renderDataContent(selectedItem)}
              type="info"
              style={{ 
                marginBottom: 16,
                wordWrap: 'break-word',
                overflowWrap: 'break-word',
                whiteSpace: 'pre-wrap'
              }}
            />

            <Form.Item
              name="label"
              label="Quyết định"
              rules={[{ required: true, message: 'Vui lòng chọn quyết định!' }]}
            >
              <Radio.Group>
                <Radio value="accept">Chấp nhận (dữ liệu tốt)</Radio>
                <Radio value="reject">Từ chối (dữ liệu kém)</Radio>
                <Radio value="modify">Sửa đổi (cần chỉnh sửa)</Radio>
              </Radio.Group>
            </Form.Item>

            <Form.Item noStyle shouldUpdate={(prev, curr) => prev.label !== curr.label}>
              {({ getFieldValue }) => {
                return getFieldValue('label') === 'modify' ? (
                  <div>
                    <Divider>Chỉnh sửa nội dung</Divider>
                    {renderModifyForm()}
                  </div>
                ) : null;
              }}
            </Form.Item>

            <Form.Item name="notes" label="Ghi chú">
              <TextArea rows={3} placeholder="Ghi chú về quyết định của bạn..." />
            </Form.Item>

            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit">
                  Xác nhận
                </Button>
                <Button onClick={() => setModalVisible(false)}>
                  Hủy
                </Button>
              </Space>
            </Form.Item>
          </Form>
        )}
      </Modal>
    </div>
  );
};

const getDataTypeColor = (type) => {
  switch (type) {
    case 'word_matching': return 'blue';
    case 'concept_understanding': return 'purple';
    case 'multi_paragraph_reading': return 'orange';
    case 'multi_hop_reasoning': return 'red';
    default: return 'default';
  }
};

const getLabelText = (label) => {
  switch (label) {
    case 'accept': return 'Chấp nhận';
    case 'reject': return 'Từ chối';
    case 'modify': return 'Sửa đổi';
    default: return label;
  }
};

export default DataLabeling;
