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
  Popconfirm
} from 'antd';
import {
  PlusOutlined,
  UploadOutlined,
  FileTextOutlined,
  EyeOutlined,
  DeleteOutlined
} from '@ant-design/icons';
import apiService from '../services/api';

const { TextArea } = Input;
const { Dragger } = Upload;

const TopicManagement = () => {
  const [topics, setTopics] = useState([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [uploadModalVisible, setUploadModalVisible] = useState(false);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [form] = Form.useForm();

  useEffect(() => {
    loadTopics();
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

  const handleUploadDocument = async (file, topicId) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('topic_id', topicId);

    try {
      await apiService.uploadDocument(formData);
      message.success('Tải lên văn bản thành công!');
      setUploadModalVisible(false);
      loadTopics();
    } catch (error) {
      message.error('Không thể tải lên văn bản');
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
      title: 'Trạng Thái',
      key: 'status',
      render: (_, record) => {
        console.log('Topic record:', record);
        const hasLegalText = record.legal_text && record.legal_text.trim().length > 0;
        return (
          <Tag color={hasLegalText ? 'green' : 'orange'}>
            {hasLegalText ? 'Đã có văn bản' : 'Chưa có văn bản'}
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
      render: (_, record) => (
        <Space>
          <Button
            icon={<UploadOutlined />}
            onClick={() => {
              setSelectedTopic(record);
              setUploadModalVisible(true);
            }}
          >
            Tải Văn Bản
          </Button>
          <Button
            icon={<EyeOutlined />}
            onClick={() => {
              Modal.info({
                title: `Chi tiết: ${record.name}`,
                width: 600,
                content: (
                  <div>
                    <p><strong>Mô tả:</strong> {record.description}</p>
                    {record.legal_text && (
                      <div>
                        <p><strong>Văn bản luật:</strong></p>
                        <div style={{ 
                          maxHeight: 300, 
                          overflow: 'auto',
                          background: '#f5f5f5',
                          padding: 12,
                          borderRadius: 4
                        }}>
                          {record.legal_text.substring(0, 1000)}
                          {record.legal_text.length > 1000 && '...'}
                        </div>
                      </div>
                    )}
                  </div>
                ),
              });
            }}
          >
            Xem Chi Tiết
          </Button>
          <Popconfirm
            title="Xóa chủ đề"
            description="Bạn có chắc chắn muốn xóa chủ đề này? Tất cả dữ liệu liên quan sẽ bị xóa."
            onConfirm={() => handleDeleteTopic(record.id)}
            okText="Xóa"
            cancelText="Hủy"
            okType="danger"
          >
            <Button
              icon={<DeleteOutlined />}
              danger
            >
              Xóa
            </Button>
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

          <Form.Item
            name="legal_text"
            label="Văn Bản Luật (Tùy chọn)"
          >
            <TextArea
              rows={6}
              placeholder="Có thể paste văn bản luật trực tiếp hoặc tải lên file sau..."
            />
          </Form.Item>

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
        title={`Tải Văn Bản cho: ${selectedTopic?.name}`}
        open={uploadModalVisible}
        onCancel={() => setUploadModalVisible(false)}
        footer={null}
        width={600}
      >
        <Dragger {...uploadProps} style={{ padding: 20 }}>
          <p className="ant-upload-drag-icon">
            <FileTextOutlined style={{ fontSize: 48, color: '#1890ff' }} />
          </p>
          <p className="ant-upload-text">
            Nhấp hoặc kéo thả file văn bản vào đây
          </p>
          <p className="ant-upload-hint">
            Hỗ trợ file .txt, .pdf, .doc, .docx
          </p>
        </Dragger>

        <Divider>Hoặc</Divider>

        <Form
          layout="vertical"
          onFinish={(values) => {
            // Update topic với text content
            handleCreateTopic({
              ...selectedTopic,
              legal_text: values.legal_text
            });
          }}
        >
          <Form.Item
            name="legal_text"
            label="Paste Văn Bản Trực Tiếp"
          >
            <TextArea
              rows={8}
              placeholder="Paste nội dung văn bản luật vào đây..."
            />
          </Form.Item>

          <Form.Item>
            <Button type="primary" htmlType="submit">
              Lưu Văn Bản
            </Button>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default TopicManagement;
