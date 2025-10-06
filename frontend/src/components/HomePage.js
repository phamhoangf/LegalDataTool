import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Timeline, Alert, Spin, Button } from 'antd';
import {
  FolderOutlined,
  DatabaseOutlined,
  TagsOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import apiService from '../services/api';

const HomePage = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadStatistics();
  }, []);

  const loadStatistics = async () => {
    try {
      setLoading(true);
      const response = await apiService.getStatistics();
      setStats(response.data);
      setError(null);
    } catch (err) {
      setError('Không thể tải thống kê. Vui lòng kiểm tra kết nối API.');
      console.error('Error loading statistics:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <p style={{ marginTop: 16 }}>Đang tải dữ liệu...</p>
      </div>
    );
  }

  if (error) {
    return <Alert message="Lỗi" description={error} type="error" showIcon />;
  }

  return (
    <div>
      <div className="page-header">
        <h1>Dashboard - Tool Tạo Dữ Liệu LegalSLM</h1>
        <p>Chào mừng đến với công cụ tạo dữ liệu finetune cho mô hình LegalSLM</p>
        <Button onClick={loadStatistics} style={{ marginTop: 8 }}>
          🔄 Cập nhật thống kê
        </Button>
      </div>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Tổng Chủ Đề"
              value={stats?.total_topics || 0}
              prefix={<FolderOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Dữ Liệu Đã Sinh"
              value={stats?.total_generated || 0}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Đã Gán Nhãn"
              value={stats?.total_labeled || 0}
              prefix={<TagsOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Tỷ Lệ Hoàn Thành"
              value={stats?.total_generated > 0 ? Math.round((stats?.total_labeled / stats?.total_generated) * 100) : 0}
              suffix="%"
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card title="Phân Bố Loại Dữ Liệu" bordered={false}>
            {stats?.data_type_distribution ? (
              <div>
                {Object.entries(stats.data_type_distribution).map(([type, count]) => (
                  <div key={type} style={{ marginBottom: 8 }}>
                    <strong>{type.toUpperCase()}:</strong> {count} mẫu
                  </div>
                ))}
              </div>
            ) : (
              <p>Chưa có dữ liệu</p>
            )}
          </Card>
        </Col>
        
        <Col xs={24} lg={12}>
          <Card title="Trạng Thái Gán Nhãn" bordered={false}>
            {stats?.label_distribution ? (
              <div>
                {Object.entries(stats.label_distribution).map(([label, count]) => (
                  <div key={label} style={{ marginBottom: 8 }}>
                    <strong>{getLabelText(label)}:</strong> {count} mẫu
                  </div>
                ))}
              </div>
            ) : (
              <p>Chưa có dữ liệu gán nhãn</p>
            )}
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col span={24}>
          <Card title="Quy Trình Sử Dụng" bordered={false}>
            <Timeline
              items={[
                {
                  color: 'green',
                  icon: <FolderOutlined />,
                  children: 'Tạo chủ đề pháp lý và tải lên văn bản luật liên quan',
                },
                {
                  color: 'blue',
                  icon: <DatabaseOutlined />,
                  children: 'Sinh dữ liệu huấn luyện (Word Matching, Concept Understanding, Multi Paragraph Reading, Multi Hop Reasoning) dựa trên văn bản luật',
                },
                {
                  color: 'purple',
                  icon: <TagsOutlined />,
                  children: 'Chuyên gia luật duyệt và gán nhãn cho dữ liệu đã sinh',
                },
                {
                  color: 'orange',
                  icon: <CheckCircleOutlined />,
                  children: 'Xuất dữ liệu đã được gán nhãn thành file .jsonl để huấn luyện',
                },
              ]}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col span={24}>
          <Alert
            message="Hướng Dẫn Nhanh"
            description={
              <div>
                <p><strong>Bước 1:</strong> Vào "Quản lý chủ đề" để tạo chủ đề mới và tải lên văn bản luật</p>
                <p><strong>Bước 2:</strong> Vào "Sinh dữ liệu" để tạo các mẫu dữ liệu huấn luyện</p>
                <p><strong>Bước 3:</strong> Vào "Duyệt dữ liệu" để duyệt và chỉnh sửa dữ liệu</p>
                <p><strong>Bước 4:</strong> Vào "Xuất dữ liệu" để tải về file .jsonl hoàn chỉnh</p>
              </div>
            }
            type="info"
            showIcon
          />
        </Col>
      </Row>
    </div>
  );
};

const getLabelText = (label) => {
  switch (label) {
    case 'accept': return 'Chấp nhận';
    case 'reject': return 'Từ chối';
    case 'modify': return 'Đã sửa';
    default: return label;
  }
};

export default HomePage;
