import React, { useState, useEffect } from 'react';
import { Select, Modal, Button, Card, Typography, message, Spin, Tag } from 'antd';
import { FileTextOutlined } from '@ant-design/icons';
import apiService from '../services/api';

const { Option } = Select;
const { Text, Paragraph } = Typography;

const VanBanDropdown = ({ onDocumentImported }) => {
  const [documents, setDocuments] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [previewDoc, setPreviewDoc] = useState(null);
  const [showPreview, setShowPreview] = useState(false);
  const [importing, setImporting] = useState(false);

  useEffect(() => {
    // Load documents when component mounts
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    setLoading(true);
    try {
      const response = await apiService.searchVanBanDocuments('', 1, 500); // Load 500 documents
      if (response.data.results) {
        const docsWithId = response.data.results
          .filter(doc => (doc.article_count || 0) > 0) // Skip documents with 0 articles
          .map((doc) => ({
            id: doc.id, // Keep original ID from backend
            title: doc.title || `Tài liệu ${doc.id}`,
            content_length: doc.content_length || 0,
            article_count: doc.article_count || 0,
            preview: doc.preview || ''
          }));
        setDocuments(docsWithId);
      } else {
        throw new Error('Không có dữ liệu văn bản');
      }
    } catch (error) {
      console.error('Error loading documents:', error);
      message.error('Lỗi khi tải danh sách văn bản: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDocumentSelect = (docId) => {
    setSelectedDocId(docId);
  };

  const handlePreviewDocument = async () => {
    if (!selectedDocId && selectedDocId !== 0) return; // Allow index 0
    
    try {
      const response = await apiService.previewVanBanDocument(selectedDocId);
      if (response.data && response.data.title) {
        setPreviewDoc({
          id: selectedDocId,
          title: response.data.title,
          preview: response.data.preview,
          full_length: response.data.full_length,
          index: response.data.index
        });
        setShowPreview(true);
      } else {
        throw new Error(response.data?.error || 'Lỗi khi xem trước văn bản');
      }
    } catch (error) {
      console.error('Error fetching document preview:', error);
      message.error('Lỗi khi tải preview văn bản: ' + error.message);
    }
  };

  const handleImportConfirm = async () => {
    if (!previewDoc) return;
    
    setImporting(true);
    try {
      const response = await apiService.importVanBanDocument(previewDoc.id);
      
      if (response.data && response.data.id) {
        message.success(response.data.message || 'Văn bản đã được thêm thành công!');
        setShowPreview(false);
        setPreviewDoc(null);
        setSelectedDocId(null);
        if (onDocumentImported) {
          onDocumentImported({
            id: response.data.id,
            title: response.data.title,
            content_length: response.data.content_length
          });
        }
      } else {
        throw new Error(response.data?.error || 'Có lỗi xảy ra khi import');
      }
    } catch (error) {
      console.error('Error importing document:', error);
      if (error.response?.status === 409) {
        message.warning('Tài liệu đã tồn tại trong hệ thống');
      } else {
        message.error('Có lỗi xảy ra khi thêm văn bản: ' + (error.response?.data?.error || error.message));
      }
    } finally {
      setImporting(false);
    }
  };

  return (
    <>
      <Card 
        title={
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <FileTextOutlined />
            Chọn văn bản từ danh sách có sẵn
          </div>
        }
        style={{ marginBottom: 16 }}
      >
        <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
          <Select
            placeholder="Chọn văn bản pháp luật từ danh sách..."
            style={{ flex: 1 }}
            value={selectedDocId}
            onSelect={handleDocumentSelect}
            loading={loading}
            allowClear
            showSearch
            filterOption={(input, option) => {
              const doc = documents.find(d => d.id === option.value);
              return doc ? doc.title.toLowerCase().indexOf(input.toLowerCase()) >= 0 : false;
            }}
            notFoundContent={loading ? <Spin size="small" /> : 'Không tìm thấy văn bản'}
          >
            {documents.map((doc) => (
              <Option key={doc.id} value={doc.id}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: '500' }}>{doc.title}</span>
                  <Tag color="blue" size="small">
                    {doc.article_count || 0} điều
                  </Tag>
                </div>
              </Option>
            ))}
          </Select>
          
          <Button 
            type="primary" 
            onClick={handlePreviewDocument}
            disabled={!selectedDocId && selectedDocId !== 0}
            style={{ flexShrink: 0 }}
          >
            Chọn
          </Button>
        </div>
        
        <div style={{ marginTop: 8, color: '#999', fontSize: '12px' }}>
          Chọn văn bản từ danh sách dropdown ({documents.length} văn bản có sẵn)
        </div>
      </Card>

      <Modal
        title="Xác nhận thêm văn bản"
        open={showPreview}
        onCancel={() => {
          setShowPreview(false);
          setPreviewDoc(null);
          setSelectedDocId(null);
        }}
        width={800}
        footer={[
          <Button key="cancel" onClick={() => {
            setShowPreview(false);
            setPreviewDoc(null);
            setSelectedDocId(null);
          }}>
            Hủy
          </Button>,
          <Button 
            key="confirm" 
            type="primary"
            loading={importing}
            onClick={handleImportConfirm}
          >
            {importing ? 'Đang thêm...' : 'Xác nhận thêm'}
          </Button>
        ]}
      >
        {previewDoc && (
          <div>
            <Card size="small" style={{ marginBottom: 16 }}>
              <Text strong style={{ fontSize: '16px' }}>
                {previewDoc.title}
              </Text>
            </Card>

            <div>
              <Text strong>Nội dung preview:</Text>
              <Card 
                size="small" 
                style={{ 
                  marginTop: 8, 
                  maxHeight: '300px', 
                  overflow: 'auto',
                  background: '#fafafa' 
                }}
              >
                <Paragraph style={{ 
                  margin: 0, 
                  fontSize: '13px', 
                  lineHeight: '1.4',
                  whiteSpace: 'pre-wrap'
                }}>
                  {previewDoc.preview}
                </Paragraph>
              </Card>
            </div>
          </div>
        )}
      </Modal>
    </>
  );
};

export default VanBanDropdown;
