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
    
    switch (item.data_type) {
      case 'sft':
        return (
          <div className="data-item">
            <div className="instruction-text">
              <strong>Instruction:</strong> {content.instruction}
            </div>
            <div className="output-text">
              <strong>Output:</strong> {content.output}
            </div>
          </div>
        );
      case 'cot':
        return (
          <div className="data-item">
            <div className="instruction-text">
              <strong>Instruction:</strong> {content.instruction}
            </div>
            <div>
              <strong>Reasoning Steps:</strong>
              <ol className="reasoning-steps">
                {content.reasoning_steps?.map((step, idx) => (
                  <li key={idx}>{step}</li>
                ))}
              </ol>
            </div>
            <div className="output-text">
              <strong>Final Answer:</strong> {content.final_answer}
            </div>
          </div>
        );
      case 'rlhf':
        return (
          <div className="data-item">
            <div className="instruction-text">
              <strong>Prompt:</strong> {content.prompt}
            </div>
            <div className="response-comparison">
              <div className="response-option">
                <strong>Response A:</strong>
                <p>{content.response_a}</p>
              </div>
              <div className="response-option">
                <strong>Response B:</strong>
                <p>{content.response_b}</p>
              </div>
            </div>
            <div>
              <strong>Preferred:</strong> <Tag color="green">Response {content.preferred}</Tag>
            </div>
          </div>
        );
      default:
        return <pre>{JSON.stringify(content, null, 2)}</pre>;
    }
  };

  const renderModifyForm = () => {
    if (!selectedItem) return null;

    const content = typeof selectedItem.content === 'string' 
      ? JSON.parse(selectedItem.content) 
      : selectedItem.content;

    switch (selectedItem.data_type) {
      case 'sft':
        return (
          <div>
            <Form.Item name={['modified_content', 'instruction']} label="Instruction">
              <TextArea rows={2} defaultValue={content.instruction} />
            </Form.Item>
            <Form.Item name={['modified_content', 'output']} label="Output">
              <TextArea rows={4} defaultValue={content.output} />
            </Form.Item>
          </div>
        );
      case 'cot':
        return (
          <div>
            <Form.Item name={['modified_content', 'instruction']} label="Instruction">
              <TextArea rows={2} defaultValue={content.instruction} />
            </Form.Item>
            <Form.Item name={['modified_content', 'reasoning_steps']} label="Reasoning Steps">
              <TextArea 
                rows={4} 
                defaultValue={content.reasoning_steps?.join('\n')}
                placeholder="Mỗi bước trên một dòng"
              />
            </Form.Item>
            <Form.Item name={['modified_content', 'final_answer']} label="Final Answer">
              <TextArea rows={2} defaultValue={content.final_answer} />
            </Form.Item>
          </div>
        );
      case 'rlhf':
        return (
          <div>
            <Form.Item name={['modified_content', 'prompt']} label="Prompt">
              <TextArea rows={2} defaultValue={content.prompt} />
            </Form.Item>
            <Form.Item name={['modified_content', 'response_a']} label="Response A">
              <TextArea rows={3} defaultValue={content.response_a} />
            </Form.Item>
            <Form.Item name={['modified_content', 'response_b']} label="Response B">
              <TextArea rows={3} defaultValue={content.response_b} />
            </Form.Item>
            <Form.Item name={['modified_content', 'preferred']} label="Preferred">
              <Radio.Group defaultValue={content.preferred}>
                <Radio value="A">Response A</Radio>
                <Radio value="B">Response B</Radio>
              </Radio.Group>
            </Form.Item>
          </div>
        );
      default:
        return null;
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
              style={{ width: 120, marginLeft: 8 }}
              value={filterDataType}
              onChange={setFilterDataType}
            >
              <Option value="all">Tất cả</Option>
              <Option value="sft">SFT</Option>
              <Option value="cot">CoT</Option>
              <Option value="rlhf">RLHF</Option>
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
              style={{ marginBottom: 16 }}
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
    case 'sft': return 'blue';
    case 'cot': return 'purple';
    case 'rlhf': return 'orange';
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
