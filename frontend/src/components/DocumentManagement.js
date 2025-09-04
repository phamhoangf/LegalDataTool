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
  Typography
} from 'antd';
import {
  PlusOutlined,
  UploadOutlined,
  FileTextOutlined,
  EyeOutlined,
  DeleteOutlined,
  EditOutlined,
  LinkOutlined
} from '@ant-design/icons';
import apiService from '../services/api';

const { TextArea } = Input;
const { Dragger } = Upload;
const { Text } = Typography;

const DocumentManagement = () => {
  const [documents, setDocuments] = useState([]);
  const [topics, setTopics] = useState([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [linkModalVisible, setLinkModalVisible] = useState(false);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [searchText, setSearchText] = useState('');
  const [form] = Form.useForm();
  const [editForm] = Form.useForm();
  const [linkForm] = Form.useForm();

  useEffect(() => {
    loadDocuments();
    loadTopics();
  }, []);

  const loadDocuments = async () => {
    try {
      setLoading(true);
      const response = await apiService.getDocuments();
      console.log('Documents loaded:', response.data);
      setDocuments(response.data);
    } catch (error) {
      message.error('Không thể tải danh sách tài liệu');
    } finally {
      setLoading(false);
    }
  };

  const loadTopics = async () => {
    try {
      const response = await apiService.getTopics();
      setTopics(response.data);
    } catch (error) {
      console.error('Không thể tải danh sách chủ đề');
    }
  };

  const handleCreateDocument = async (values) => {
    try {
      await apiService.createDocument(values);
      message.success('Tạo tài liệu thành công!');
      setModalVisible(false);
      form.resetFields();
      loadDocuments();
    } catch (error) {
      message.error('Không thể tạo tài liệu');
    }
  };

  const handleUpdateDocument = async (values) => {
    try {
      await apiService.updateDocument(selectedDocument.id, values);
      message.success('Cập nhật tài liệu thành công!');
      setEditModalVisible(false);
      editForm.resetFields();
      setSelectedDocument(null);
      loadDocuments();
    } catch (error) {
      message.error('Không thể cập nhật tài liệu');
    }
  };

  const handleDeleteDocument = async (documentId) => {
    try {
      await apiService.deleteDocument(documentId);
      message.success('Xóa tài liệu thành công!');
      loadDocuments();
    } catch (error) {
      message.error('Không thể xóa tài liệu');
    }
  };

  const showDocumentDetails = async (document) => {
    try {
      setLoading(true);
      // Lấy chi tiết đầy đủ của document
      const response = await apiService.getDocumentDetails(document.id);
      setSelectedDocument({
        ...document,
        fullContent: response.data.content || document.content,
        topics: response.data.topics || document.topics || []
      });
      setSearchText('');
      setDetailModalVisible(true);
    } catch (error) {
      // Fallback nếu API không có endpoint getDocumentDetails
      setSelectedDocument({
        ...document,
        fullContent: document.content,
        topics: document.topics || []
      });
      setSearchText('');
      setDetailModalVisible(true);
    } finally {
      setLoading(false);
    }
  };

  const highlightSearchText = (text, searchText) => {
    if (!searchText || !text) return text;
    const regex = new RegExp(`(${searchText})`, 'gi');
    return text.replace(regex, '<mark style="background-color: #ffeb3b;">$1</mark>');
  };

  const handleUploadDocument = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', file.name);
    formData.append('document_type', 'law');

    try {
      await apiService.uploadDocumentFile(formData);
      message.success('Tải lên tài liệu thành công!');
      loadDocuments();
    } catch (error) {
      message.error('Không thể tải lên tài liệu');
    }
  };

  const handleLinkToTopics = async (values) => {
    try {
      const { topic_ids } = values;
      for (const topicId of topic_ids) {
        await apiService.linkDocumentToTopic(topicId, selectedDocument.id, {
          relevance_score: 1.0
        });
      }
      message.success('Liên kết tài liệu với chủ đề thành công!');
      setLinkModalVisible(false);
      linkForm.resetFields();
      setSelectedDocument(null);
      loadDocuments();
    } catch (error) {
      message.error('Không thể liên kết tài liệu');
    }
  };

  const uploadProps = {
    name: 'file',
    multiple: false,
    accept: '.txt,.pdf,.doc,.docx',
    beforeUpload: (file) => {
      handleUploadDocument(file);
      return false; // Prevent default upload
    },
    onDrop(e) {
      console.log('Dropped files', e.dataTransfer.files);
    },
  };

  const columns = [
    {
      title: 'Tiêu Đề',
      dataIndex: 'title',
      key: 'title',
      render: (text) => <strong>{text}</strong>,
    },
    {
      title: 'Loại Tài Liệu',
      dataIndex: 'document_type',
      key: 'document_type',
      render: (type) => (
        <Tag color={type === 'law' ? 'blue' : 'green'}>
          {type === 'law' ? 'Văn bản pháp luật' : 'Tài liệu khác'}
        </Tag>
      ),
    },
    {
      title: 'Độ Dài',
      key: 'content_length',
      render: (_, record) => {
        const length = record.content ? record.content.length : 0;
        return <Text type="secondary">{length.toLocaleString()} ký tự</Text>;
      },
    },
    {
      title: 'Chủ Đề Liên Kết',
      key: 'linked_topics',
      render: (_, record) => {
        const topicCount = record.topics ? record.topics.length : 0;
        if (topicCount > 0) {
          return (
            <Button
              type="link"
              size="small"
              onClick={() => {
                Modal.info({
                  title: `Chủ đề liên kết - ${record.title}`,
                  width: 500,
                  content: (
                    <div>
                      <p>Tài liệu này được liên kết với {topicCount} chủ đề:</p>
                      <div style={{ marginTop: 16 }}>
                        {record.topics.map(topic => (
                          <Tag key={topic.id} color="purple" style={{ marginBottom: 8, marginRight: 8 }}>
                            {topic.name}
                          </Tag>
                        ))}
                      </div>
                    </div>
                  ),
                });
              }}
            >
              <Tag color="purple">{topicCount} chủ đề</Tag>
            </Button>
          );
        }
        return <Text type="secondary" italic>Chưa liên kết</Text>;
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
      width: 160,
      render: (_, record) => (
        <Space size="small">
          <Tooltip title="Xem Chi Tiết">
            <Button
              icon={<EyeOutlined />}
              size="small"
              onClick={() => showDocumentDetails(record)}
            />
          </Tooltip>
          
          <Tooltip title="Chỉnh Sửa">
            <Button
              icon={<EditOutlined />}
              size="small"
              onClick={() => {
                setSelectedDocument(record);
                editForm.setFieldsValue({
                  title: record.title,
                  content: record.content,
                  document_type: record.document_type
                });
                setEditModalVisible(true);
              }}
            />
          </Tooltip>
          
          <Tooltip title="Liên Kết Chủ Đề">
            <Button
              icon={<LinkOutlined />}
              size="small"
              onClick={() => {
                setSelectedDocument(record);
                setLinkModalVisible(true);
              }}
            />
          </Tooltip>
          
          <Popconfirm
            title="Xóa tài liệu"
            description="Bạn có chắc chắn muốn xóa tài liệu này?"
            onConfirm={() => handleDeleteDocument(record.id)}
            okText="Xóa"
            cancelText="Hủy"
            okType="danger"
          >
            <Tooltip title="Xóa Tài Liệu">
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
        <h1>Quản Lý Tài Liệu Pháp Lý</h1>
        <p>Tạo, quản lý và tổ chức các tài liệu pháp luật</p>
      </div>

      <Card
        title="Danh Sách Tài Liệu"
        extra={
          <Space>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setModalVisible(true)}
            >
              Tạo Tài Liệu Mới
            </Button>
          </Space>
        }
      >
        <div style={{ marginBottom: 16 }}>
          <h4>📁 Tải Lên Tài Liệu</h4>
          <Dragger {...uploadProps} style={{ padding: 16 }}>
            <p className="ant-upload-drag-icon">
              <FileTextOutlined style={{ fontSize: 32, color: '#1890ff' }} />
            </p>
            <p className="ant-upload-text">
              Nhấp hoặc kéo thả file vào đây để tải lên
            </p>
            <p className="ant-upload-hint">
              Hỗ trợ file .txt, .pdf, .doc, .docx
            </p>
          </Dragger>
        </div>

        <Table
          columns={columns}
          dataSource={documents}
          rowKey="id"
          loading={loading}
          pagination={{ pageSize: 10 }}
        />
      </Card>

      {/* Modal tạo tài liệu mới */}
      <Modal
        title="Tạo Tài Liệu Mới"
        open={modalVisible}
        onCancel={() => {
          setModalVisible(false);
          form.resetFields();
        }}
        footer={null}
        width={700}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateDocument}
        >
          <Form.Item
            name="title"
            label="Tiêu Đề Tài Liệu"
            rules={[{ required: true, message: 'Vui lòng nhập tiêu đề!' }]}
          >
            <Input placeholder="VD: Luật Giao thông đường bộ 2008 - Điều 60" />
          </Form.Item>

          <Form.Item
            name="document_type"
            label="Loại Tài Liệu"
            rules={[{ required: true, message: 'Vui lòng chọn loại tài liệu!' }]}
            initialValue="law"
          >
            <Select>
              <Select.Option value="law">Văn bản pháp luật</Select.Option>
              <Select.Option value="other">Tài liệu khác</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="content"
            label="Nội Dung Tài Liệu"
            rules={[{ required: true, message: 'Vui lòng nhập nội dung!' }]}
          >
            <TextArea
              rows={10}
              placeholder="Paste nội dung tài liệu vào đây..."
            />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                Tạo Tài Liệu
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

      {/* Modal chỉnh sửa tài liệu */}
      <Modal
        title="Chỉnh Sửa Tài Liệu"
        open={editModalVisible}
        onCancel={() => {
          setEditModalVisible(false);
          editForm.resetFields();
          setSelectedDocument(null);
        }}
        footer={null}
        width={700}
      >
        <Form
          form={editForm}
          layout="vertical"
          onFinish={handleUpdateDocument}
        >
          <Form.Item
            name="title"
            label="Tiêu Đề Tài Liệu"
            rules={[{ required: true, message: 'Vui lòng nhập tiêu đề!' }]}
          >
            <Input placeholder="VD: Luật Giao thông đường bộ 2008 - Điều 60" />
          </Form.Item>

          <Form.Item
            name="document_type"
            label="Loại Tài Liệu"
            rules={[{ required: true, message: 'Vui lòng chọn loại tài liệu!' }]}
          >
            <Select>
              <Select.Option value="law">Văn bản pháp luật</Select.Option>
              <Select.Option value="other">Tài liệu khác</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="content"
            label="Nội Dung Tài Liệu"
            rules={[{ required: true, message: 'Vui lòng nhập nội dung!' }]}
          >
            <TextArea
              rows={10}
              placeholder="Paste nội dung tài liệu vào đây..."
            />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                Cập Nhật
              </Button>
              <Button onClick={() => {
                setEditModalVisible(false);
                editForm.resetFields();
                setSelectedDocument(null);
              }}>
                Hủy
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Modal liên kết với chủ đề */}
      <Modal
        title={`Liên Kết Tài Liệu: ${selectedDocument?.title}`}
        open={linkModalVisible}
        onCancel={() => {
          setLinkModalVisible(false);
          linkForm.resetFields();
          setSelectedDocument(null);
        }}
        footer={null}
        width={500}
      >
        <Form
          form={linkForm}
          layout="vertical"
          onFinish={handleLinkToTopics}
        >
          <Form.Item
            name="topic_ids"
            label="Chọn Chủ Đề"
            rules={[{ required: true, message: 'Vui lòng chọn ít nhất một chủ đề!' }]}
          >
            <Select
              mode="multiple"
              placeholder="Chọn các chủ đề để liên kết"
              optionFilterProp="children"
            >
              {topics.map(topic => (
                <Select.Option key={topic.id} value={topic.id}>
                  {topic.name}
                </Select.Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                Liên Kết
              </Button>
              <Button onClick={() => {
                setLinkModalVisible(false);
                linkForm.resetFields();
                setSelectedDocument(null);
              }}>
                Hủy
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Modal chi tiết tài liệu cải tiến */}
      <Modal
        title={
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingRight: 40 }}>
            <span>📄 Chi tiết tài liệu: {selectedDocument?.title}</span>
            <Input.Search
              placeholder="Tìm kiếm trong nội dung..."
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              style={{ width: 300, marginRight: 10 }}
            />
          </div>
        }
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        width="60%"
        style={{ top: 60, paddingBottom: 0 }}
        bodyStyle={{ maxHeight: 'calc(100vh - 200px)', overflow: 'hidden' }}
        footer={[
          <Button key="close" onClick={() => setDetailModalVisible(false)}>
            Đóng
          </Button>
        ]}
      >
        {selectedDocument && (
          <div>
            {/* Thông tin metadata */}
            <div style={{ 
              background: '#f0f2f5', 
              padding: 16, 
              borderRadius: 8,
              marginBottom: 16 
            }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 12 }}>
                <div>
                  <Text strong>📋 Tiêu đề:</Text>
                  <br />
                  <Text>{selectedDocument.title}</Text>
                </div>
                <div>
                  <Text strong>📂 Loại tài liệu:</Text>
                  <br />
                  <Tag color={selectedDocument.document_type === 'law' ? 'blue' : 'green'}>
                    {selectedDocument.document_type === 'law' ? 'Văn bản pháp luật' : 'Tài liệu khác'}
                  </Tag>
                </div>
                <div>
                  <Text strong>📏 Độ dài:</Text>
                  <br />
                  <Text>{selectedDocument.fullContent?.length?.toLocaleString() || 0} ký tự</Text>
                </div>
                <div>
                  <Text strong>⏰ Ngày tải lên:</Text>
                  <br />
                  <Text>
                    {selectedDocument.uploaded_at 
                      ? new Date(selectedDocument.uploaded_at).toLocaleString('vi-VN')
                      : 'Không rõ'
                    }
                  </Text>
                </div>
              </div>

              {/* Thông tin bổ sung */}
              <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 12 }}>
                {selectedDocument.document_number && (
                  <div>
                    <Text strong>🔢 Số hiệu:</Text>
                    <br />
                    <Text>{selectedDocument.document_number}</Text>
                  </div>
                )}
                {selectedDocument.effective_date && (
                  <div>
                    <Text strong>📅 Ngày hiệu lực:</Text>
                    <br />
                    <Text>{new Date(selectedDocument.effective_date).toLocaleDateString('vi-VN')}</Text>
                  </div>
                )}
                {selectedDocument.source_url && (
                  <div>
                    <Text strong>🔗 Nguồn:</Text>
                    <br />
                    <a href={selectedDocument.source_url} target="_blank" rel="noopener noreferrer">
                      {selectedDocument.source_url}
                    </a>
                  </div>
                )}
              </div>
            </div>

            {/* Chủ đề liên kết */}
            {selectedDocument.topics && selectedDocument.topics.length > 0 && (
              <div style={{ marginBottom: 16 }}>
                <Text strong>🏷️ Chủ đề liên kết:</Text>
                <div style={{ marginTop: 8 }}>
                  {selectedDocument.topics.map(topic => (
                    <Tag key={topic.id} color="purple" style={{ marginBottom: 4 }}>
                      {topic.name}
                    </Tag>
                  ))}
                </div>
              </div>
            )}

            {/* Nội dung tài liệu */}
            {selectedDocument.fullContent ? (
              <div>
                <Text strong>📝 Nội dung tài liệu:</Text>
                <div style={{ 
                  background: '#fafafa',
                  border: '1px solid #d9d9d9',
                  borderRadius: 8,
                  marginTop: 8,
                  maxHeight: '50vh',
                  overflow: 'auto'
                }}>
                  <div style={{ 
                    padding: 16,
                    fontSize: '14px',
                    lineHeight: '1.6',
                    fontFamily: 'monospace'
                  }}>
                    <div
                      dangerouslySetInnerHTML={{
                        __html: highlightSearchText(selectedDocument.fullContent, searchText)
                      }}
                    />
                  </div>
                </div>

                {/* Thống kê nội dung */}
                <div style={{ 
                  marginTop: 12,
                  padding: 12,
                  background: '#e6f7ff',
                  borderRadius: 6,
                  fontSize: '12px'
                }}>
                  <Space>
                    <Text>📊 Thống kê:</Text>
                    <Text>{selectedDocument.fullContent.split(' ').filter(word => word.trim()).length} từ</Text>
                    <Text>{selectedDocument.fullContent.split('\n').length} dòng</Text>
                    <Text>{selectedDocument.fullContent.split(/[.!?]+/).filter(s => s.trim()).length} câu</Text>
                    {searchText && (
                      <Text style={{ color: '#1890ff' }}>
                        {(selectedDocument.fullContent.match(new RegExp(searchText, 'gi')) || []).length} kết quả tìm kiếm
                      </Text>
                    )}
                  </Space>
                </div>
              </div>
            ) : (
              <div style={{ 
                textAlign: 'center', 
                padding: 60,
                background: '#fafafa',
                borderRadius: 8,
                color: '#999' 
              }}>
                <FileTextOutlined style={{ fontSize: 48, marginBottom: 16 }} />
                <p style={{ fontSize: 16, marginBottom: 8 }}>Không có nội dung</p>
                <p style={{ fontSize: 14 }}>
                  Tài liệu này không có nội dung để hiển thị
                </p>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default DocumentManagement;
