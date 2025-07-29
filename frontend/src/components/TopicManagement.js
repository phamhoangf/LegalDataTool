import React, { useState, useEffect } from 'react';
import {
  Card,
  Button,
  Table,
  Modal,
  Form,
  Input,
  Upload,
  message,
  Space,
  Tag,
  Divider,
  Popconfirm,
  Tooltip,
  Select,
  Typography,
  Tabs
} from 'antd';
import {
  PlusOutlined,
  UploadOutlined,
  FileTextOutlined,
  EyeOutlined,
  DeleteOutlined,
  LinkOutlined
} from '@ant-design/icons';
import apiService from '../services/api';

const { TextArea } = Input;
const { Dragger } = Upload;
const { Text } = Typography;
const { TabPane } = Tabs;

const TopicManagement = () => {
  const [topics, setTopics] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [uploadModalVisible, setUploadModalVisible] = useState(false);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [form] = Form.useForm();

  useEffect(() => {
    loadTopics();
    loadDocuments();
  }, []);

  const loadTopics = async () => {
    try {
      setLoading(true);
      const response = await apiService.getTopics();
      console.log('Topics loaded:', response.data);
      setTopics(response.data);
    } catch (error) {
      message.error('Không thể tải danh sách chủ đề');
    } finally {
      setLoading(false);
    }
  };

  const loadDocuments = async () => {
    try {
      const response = await apiService.getDocuments();
      setDocuments(response.data);
    } catch (error) {
      console.error('Không thể tải danh sách tài liệu');
    }
  };

  const handleCreateTopic = async (values) => {
    try {
      await apiService.createTopic(values);
      message.success('Tạo chủ đề thành công!');
      setModalVisible(false);
      form.resetFields();
      loadTopics();
    } catch (error) {
      message.error('Không thể tạo chủ đề');
    }
  };

  const handleDeleteTopic = async (topicId) => {
    try {
      await apiService.deleteTopic(topicId);
      message.success('Xóa chủ đề thành công!');
      loadTopics();
    } catch (error) {
      message.error('Không thể xóa chủ đề');
    }
  };

  const handleUploadDocument = async (file, topicId, documentTitle = null) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('topic_id', topicId);
    formData.append('title', documentTitle || file.name);
    formData.append('document_type', 'law');

    try {
      await apiService.uploadDocument(formData);
      message.success('Tải lên tài liệu thành công!');
      setUploadModalVisible(false);
      loadTopics();
      loadDocuments(); // Cập nhật danh sách tài liệu
    } catch (error) {
      message.error('Không thể tải lên tài liệu');
    }
  };

  const handleCreateDocumentFromText = async (values) => {
    try {
      // Tạo document từ text
      const docResponse = await apiService.createDocument({
        title: values.title || `Văn bản cho ${selectedTopic.name}`,
        content: values.legal_text,
        document_type: 'law'
      });

      // Link với topic
      await apiService.linkDocumentToTopic(
        selectedTopic.id, 
        docResponse.data.id,
        { relevance_score: 1.0 }
      );

      message.success('Tạo và liên kết tài liệu thành công!');
      setUploadModalVisible(false);
      loadTopics();
      loadDocuments(); // Cập nhật danh sách tài liệu
    } catch (error) {
      message.error('Không thể tạo tài liệu');
    }
  };

  const handleLinkExistingDocument = async (values) => {
    try {
      await apiService.linkDocumentToTopic(
        selectedTopic.id,
        values.document_id,
        { relevance_score: 1.0 }
      );

      message.success('Liên kết tài liệu thành công!');
      setUploadModalVisible(false);
      loadTopics();
      loadDocuments(); // Cập nhật danh sách tài liệu
    } catch (error) {
      message.error('Không thể liên kết tài liệu');
    }
  };

  const uploadProps = {
    name: 'file',
    multiple: false,
    accept: '.txt,.pdf,.doc,.docx',
    beforeUpload: (file) => {
      if (selectedTopic) {
        handleUploadDocument(file, selectedTopic.id);
      }
      return false; // Prevent default upload
    },
    onDrop(e) {
      console.log('Dropped files', e.dataTransfer.files);
    },
  };

  const columns = [
    {
      title: 'Tên Chủ Đề',
      dataIndex: 'name',
      key: 'name',
      render: (text) => <strong>{text}</strong>,
    },
    {
      title: 'Mô Tả',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
    {
      title: 'Tài Liệu',
      key: 'documents',
      render: (_, record) => {
        const docCount = record.document_count || 0;
        return (
          <div>
            <Button
              type="link"
              size="small"
              disabled={docCount === 0}
              onClick={() => {
                if (docCount > 0) {
                  Modal.info({
                    title: `Tài liệu liên kết - ${record.name}`,
                    width: 600,
                    content: (
                      <div>
                        <p>Chủ đề này có {docCount} tài liệu liên kết:</p>
                        <div style={{ marginTop: 16 }}>
                          {record.documents && record.documents.map(doc => (
                            <Tag key={doc.id} color="blue" style={{ marginBottom: 8, marginRight: 8, display: 'block', width: 'fit-content' }}>
                              <FileTextOutlined /> {doc.title}
                            </Tag>
                          ))}
                        </div>
                      </div>
                    ),
                  });
                }
              }}
            >
              <Tag color={docCount > 0 ? 'green' : 'orange'}>
                {docCount} tài liệu
              </Tag>
            </Button>
          </div>
        );
      },
    },
    {
      title: 'Trạng Thái',
      key: 'status',
      render: (_, record) => {
        const hasDocuments = record.document_count > 0;
        const hasLegalText = record.legal_text && record.legal_text.trim().length > 0;
        return (
          <Tag color={hasDocuments || hasLegalText ? 'green' : 'orange'}>
            {hasDocuments || hasLegalText ? 'Sẵn sàng' : 'Chưa sẵn sàng'}
          </Tag>
        );
      },
    },
    {
      title: 'Ngày Tạo',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date) => new Date(date).toLocaleDateString('vi-VN'),
    },
    {
      title: 'Thao Tác',
      key: 'actions',
      width: 120,
      render: (_, record) => (
        <Space size="small">
          <Tooltip title="Tải Văn Bản">
            <Button
              icon={<UploadOutlined />}
              size="small"
              onClick={() => {
                setSelectedTopic(record);
                setUploadModalVisible(true);
              }}
            />
          </Tooltip>
          
          <Tooltip title="Xem Chi Tiết">
            <Button
              icon={<EyeOutlined />}
              size="small"
              onClick={() => {
                Modal.info({
                  title: `Chi tiết: ${record.name}`,
                  width: 700,
                  content: (
                    <div>
                      <p><strong>Mô tả:</strong> {record.description}</p>
                      
                      <Divider orientation="left">Tài liệu ({record.document_count || 0})</Divider>
                      {record.documents && record.documents.length > 0 ? (
                        <div style={{ marginBottom: 16 }}>
                          {record.documents.map((doc, index) => (
                            <Tag key={doc.id} color="blue" style={{ marginBottom: 4 }}>
                              <FileTextOutlined /> {doc.title}
                            </Tag>
                          ))}
                        </div>
                      ) : (
                        <p style={{ color: '#999', fontStyle: 'italic' }}>
                          Chưa có tài liệu nào được liên kết
                        </p>
                      )}
                      
                      {record.legal_text && record.legal_text.trim().length > 0 && (
                        <div>
                          <Divider orientation="left">Nội dung văn bản</Divider>
                          <div style={{ 
                            maxHeight: 300, 
                            overflow: 'auto',
                            background: '#f5f5f5',
                            padding: 12,
                            borderRadius: 4,
                            fontSize: '13px',
                            lineHeight: '1.5'
                          }}>
                            {record.legal_text.substring(0, 2000)}
                            {record.legal_text.length > 2000 && '...'}
                          </div>
                        </div>
                      )}
                    </div>
                  ),
                });
              }}
            />
          </Tooltip>
          
          <Popconfirm
            title="Xóa chủ đề"
            description="Bạn có chắc chắn muốn xóa chủ đề này? Tất cả dữ liệu liên quan sẽ bị xóa."
            onConfirm={() => handleDeleteTopic(record.id)}
            okText="Xóa"
            cancelText="Hủy"
            okType="danger"
          >
            <Tooltip title="Xóa Chủ Đề">
              <Button
                icon={<DeleteOutlined />}
                size="small"
                danger
              />
            </Tooltip>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <div className="page-header">
        <h1>Quản Lý Chủ Đề Pháp Lý</h1>
        <p>Tạo và quản lý các chủ đề để sinh dữ liệu huấn luyện</p>
      </div>

      <Card
        title="Danh Sách Chủ Đề"
        extra={
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => setModalVisible(true)}
          >
            Tạo Chủ Đề Mới
          </Button>
        }
      >
        <Table
          columns={columns}
          dataSource={topics}
          rowKey="id"
          loading={loading}
          pagination={{ pageSize: 10 }}
        />
      </Card>

      {/* Modal tạo chủ đề mới */}
      <Modal
        title="Tạo Chủ Đề Mới"
        open={modalVisible}
        onCancel={() => {
          setModalVisible(false);
          form.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateTopic}
        >
          <Form.Item
            name="name"
            label="Tên Chủ Đề"
            rules={[{ required: true, message: 'Vui lòng nhập tên chủ đề!' }]}
          >
            <Input placeholder="Ví dụ: Giấy phép lái xe" />
          </Form.Item>

          <Form.Item
            name="description"
            label="Mô Tả"
            rules={[{ required: true, message: 'Vui lòng nhập mô tả!' }]}
          >
            <TextArea
              rows={4}
              placeholder="Mô tả chi tiết về chủ đề này..."
            />
          </Form.Item>

          <div style={{ 
            background: '#f0f9ff', 
            border: '1px solid #bae6fd', 
            borderRadius: 6, 
            padding: 12, 
            marginBottom: 16 
          }}>
            <p style={{ margin: 0, color: '#0369a1' }}>
              💡 <strong>Lưu ý:</strong> Sau khi tạo chủ đề, bạn có thể tải lên các tài liệu pháp luật liên quan 
              bằng nút "Tải Văn Bản" ở bảng danh sách.
            </p>
          </div>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                Tạo Chủ Đề
              </Button>
              <Button onClick={() => {
                setModalVisible(false);
                form.resetFields();
              }}>
                Hủy
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Modal tải lên văn bản */}
      <Modal
        title={`Thêm Tài Liệu cho: ${selectedTopic?.name}`}
        open={uploadModalVisible}
        onCancel={() => setUploadModalVisible(false)}
        footer={null}
        width={800}
      >
        <Tabs defaultActiveKey="upload" type="card">
          <TabPane tab="📁 Tải Lên File" key="upload">
            <div style={{ padding: 16 }}>
              <Dragger {...uploadProps} style={{ padding: 20 }}>
                <p className="ant-upload-drag-icon">
                  <FileTextOutlined style={{ fontSize: 48, color: '#1890ff' }} />
                </p>
                <p className="ant-upload-text">
                  Nhấp hoặc kéo thả file văn bản vào đây
                </p>
                <p className="ant-upload-hint">
                  Hỗ trợ file .txt, .pdf, .doc, .docx. File sẽ được tự động đặt tên theo tên file.
                </p>
              </Dragger>
            </div>
          </TabPane>

          <TabPane tab="✏️ Tạo Từ Văn Bản" key="create">
            <div style={{ padding: 16 }}>
              <Form
                layout="vertical"
                onFinish={handleCreateDocumentFromText}
              >
                <Form.Item
                  name="title"
                  label="Tiêu Đề Tài Liệu"
                  rules={[{ required: true, message: 'Vui lòng nhập tiêu đề!' }]}
                >
                  <Input 
                    placeholder="VD: Luật Giao thông đường bộ 2008 - Điều 60"
                  />
                </Form.Item>
                
                <Form.Item
                  name="legal_text"
                  label="Nội Dung Văn Bản"
                  rules={[{ required: true, message: 'Vui lòng nhập nội dung!' }]}
                >
                  <TextArea
                    rows={8}
                    placeholder="Paste nội dung văn bản luật vào đây..."
                  />
                </Form.Item>

                <Form.Item>
                  <Space>
                    <Button type="primary" htmlType="submit">
                      Tạo Tài Liệu
                    </Button>
                    <Button onClick={() => setUploadModalVisible(false)}>
                      Hủy
                    </Button>
                  </Space>
                </Form.Item>
              </Form>
            </div>
          </TabPane>

          <TabPane tab="🔗 Liên Kết Tài Liệu Có Sẵn" key="link">
            <div style={{ padding: 16 }}>
              <Form
                layout="vertical"
                onFinish={handleLinkExistingDocument}
              >
                <Form.Item
                  name="document_id"
                  label="Chọn Tài Liệu"
                  rules={[{ required: true, message: 'Vui lòng chọn tài liệu!' }]}
                >
                  <Select
                    placeholder="Chọn tài liệu từ danh sách có sẵn"
                    optionFilterProp="children"
                    showSearch
                    filterOption={(input, option) =>
                      option.children.toLowerCase().indexOf(input.toLowerCase()) >= 0
                    }
                  >
                    {documents
                      .filter(doc => {
                        // Lọc ra những tài liệu chưa được liên kết với chủ đề này
                        const isLinked = selectedTopic?.documents?.some(d => d.id === doc.id);
                        return !isLinked;
                      })
                      .map(doc => (
                        <Select.Option key={doc.id} value={doc.id}>
                          <div>
                            <strong>{doc.title}</strong>
                            <br />
                            <Text type="secondary" style={{ fontSize: '12px' }}>
                              {doc.content ? `${doc.content.length.toLocaleString()} ký tự` : 'Không có nội dung'}
                            </Text>
                          </div>
                        </Select.Option>
                      ))
                    }
                  </Select>
                </Form.Item>

                {documents.filter(doc => {
                  const isLinked = selectedTopic?.documents?.some(d => d.id === doc.id);
                  return !isLinked;
                }).length === 0 && (
                  <div style={{ 
                    textAlign: 'center', 
                    color: '#999', 
                    padding: 20,
                    background: '#f9f9f9',
                    borderRadius: 6,
                    margin: '16px 0'
                  }}>
                    <FileTextOutlined style={{ fontSize: 24, marginBottom: 8 }} />
                    <p>Tất cả tài liệu đã được liên kết với chủ đề này</p>
                    <p style={{ fontSize: '12px' }}>
                      Hãy tạo tài liệu mới hoặc sử dụng tab khác
                    </p>
                  </div>
                )}

                <Form.Item>
                  <Space>
                    <Button 
                      type="primary" 
                      htmlType="submit"
                      disabled={documents.filter(doc => {
                        const isLinked = selectedTopic?.documents?.some(d => d.id === doc.id);
                        return !isLinked;
                      }).length === 0}
                    >
                      Liên Kết Tài Liệu
                    </Button>
                    <Button onClick={() => setUploadModalVisible(false)}>
                      Hủy
                    </Button>
                  </Space>
                </Form.Item>
              </Form>
            </div>
          </TabPane>
        </Tabs>
      </Modal>
    </div>
  );
};

export default TopicManagement;
