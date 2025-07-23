import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Table,
  Tag,
  Spin,
  Alert
} from 'antd';
import {
  BarChartOutlined,
  PieChartOutlined,
  TrophyOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import apiService from '../services/api';

const Statistics = () => {
  const [stats, setStats] = useState(null);
  const [topics, setTopics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [statsResponse, topicsResponse] = await Promise.all([
        apiService.getStatistics(),
        apiService.getTopics()
      ]);
      
      setStats(statsResponse.data);
      setTopics(topicsResponse.data);
      setError(null);
    } catch (err) {
      setError('Không thể tải dữ liệu thống kê');
      console.error('Error loading statistics:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <p style={{ marginTop: 16 }}>Đang tải thống kê...</p>
      </div>
    );
  }

  if (error) {
    return <Alert message="Lỗi" description={error} type="error" showIcon />;
  }

  const completionRate = stats?.total_generated > 0 
    ? Math.round((stats.total_labeled / stats.total_generated) * 100)
    : 0;

  const topicColumns = [
    {
      title: 'Chủ đề',
      dataIndex: 'name',
      key: 'name',
      render: (text) => <strong>{text}</strong>,
    },
    {
      title: 'Mô tả',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
    {
      title: 'Trạng thái',
      key: 'status',
      render: (_, record) => (
        <Tag color={record.legal_text ? 'green' : 'orange'}>
          {record.legal_text ? 'Có văn bản' : 'Chưa có văn bản'}
        </Tag>
      ),
    },
    {
      title: 'Ngày tạo',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date) => new Date(date).toLocaleDateString('vi-VN'),
    },
  ];

  return (
    <div>
      <div className="page-header">
        <h1>Thống Kê & Báo Cáo</h1>
        <p>Tổng quan về tiến độ và chất lượng dữ liệu</p>
      </div>

      {/* Thống kê tổng quan */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Tổng Chủ Đề"
              value={stats?.total_topics || 0}
              prefix={<BarChartOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Dữ Liệu Đã Sinh"
              value={stats?.total_generated || 0}
              prefix={<PieChartOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Đã Gán Nhãn"
              value={stats?.total_labeled || 0}
              prefix={<TrophyOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Tỷ Lệ Hoàn Thành"
              value={completionRate}
              suffix="%"
              prefix={<ClockCircleOutlined />}
              valueStyle={{ 
                color: completionRate >= 80 ? '#3f8600' : completionRate >= 50 ? '#1890ff' : '#cf1322' 
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* Tiến độ chi tiết */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={12}>
          <Card title="Tiến Độ Gán Nhãn" bordered={false}>
            <Progress
              type="circle"
              percent={completionRate}
              format={(percent) => `${percent}%`}
              strokeColor={{
                '0%': '#108ee9',
                '100%': '#87d068',
              }}
              size={150}
            />
            <div style={{ textAlign: 'center', marginTop: 16 }}>
              <p>
                <strong>{stats?.total_labeled || 0}</strong> / <strong>{stats?.total_generated || 0}</strong> mẫu đã hoàn thành
              </p>
            </div>
          </Card>
        </Col>
        
        <Col xs={24} lg={12}>
          <Card title="Phân Bố Loại Dữ Liệu" bordered={false}>
            {stats?.data_type_distribution ? (
              <div>
                {Object.entries(stats.data_type_distribution).map(([type, count]) => {
                  const percentage = stats.total_generated > 0 
                    ? Math.round((count / stats.total_generated) * 100) 
                    : 0;
                  
                  return (
                    <div key={type} style={{ marginBottom: 16 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                        <span><strong>{type.toUpperCase()}</strong></span>
                        <span>{count} mẫu ({percentage}%)</span>
                      </div>
                      <Progress 
                        percent={percentage} 
                        showInfo={false}
                        strokeColor={getDataTypeColor(type)}
                      />
                    </div>
                  );
                })}
              </div>
            ) : (
              <p style={{ textAlign: 'center', color: '#999' }}>Chưa có dữ liệu</p>
            )}
          </Card>
        </Col>
      </Row>

      {/* Phân bố nhãn */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card title="Phân Bố Kết Quả Gán Nhãn" bordered={false}>
            {stats?.label_distribution ? (
              <Row gutter={[16, 16]}>
                {Object.entries(stats.label_distribution).map(([label, count]) => {
                  const percentage = stats.total_labeled > 0 
                    ? Math.round((count / stats.total_labeled) * 100) 
                    : 0;
                  
                  return (
                    <Col xs={24} sm={8} key={label}>
                      <Card size="small" style={{ textAlign: 'center' }}>
                        <Statistic
                          title={getLabelText(label)}
                          value={count}
                          suffix={`(${percentage}%)`}
                          valueStyle={{ color: getLabelColor(label) }}
                        />
                        <Progress
                          percent={percentage}
                          showInfo={false}
                          strokeColor={getLabelColor(label)}
                          size="small"
                        />
                      </Card>
                    </Col>
                  );
                })}
              </Row>
            ) : (
              <div style={{ textAlign: 'center', padding: 40, color: '#999' }}>
                Chưa có dữ liệu gán nhãn
              </div>
            )}
          </Card>
        </Col>
      </Row>

      {/* Danh sách chủ đề */}
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card title="Danh Sách Chủ Đề" bordered={false}>
            <Table
              columns={topicColumns}
              dataSource={topics}
              rowKey="id"
              pagination={{ pageSize: 10 }}
              size="small"
            />
          </Card>
        </Col>
      </Row>

      {/* Recommendations */}
      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col span={24}>
          <Card title="Khuyến Nghị" bordered={false}>
            <div>
              {completionRate < 30 && (
                <Alert
                  message="Tỷ lệ gán nhãn thấp"
                  description="Nên tăng cường gán nhãn để có đủ dữ liệu huấn luyện chất lượng."
                  type="warning"
                  showIcon
                  style={{ marginBottom: 16 }}
                />
              )}
              
              {(stats?.total_topics || 0) < 3 && (
                <Alert
                  message="Số lượng chủ đề ít"
                  description="Nên tạo thêm nhiều chủ đề khác nhau để đa dạng hóa dữ liệu huấn luyện."
                  type="info"
                  showIcon
                  style={{ marginBottom: 16 }}
                />
              )}
              
              {completionRate >= 80 && (
                <Alert
                  message="Tiến độ tốt!"
                  description="Có thể bắt đầu xuất dữ liệu để huấn luyện mô hình."
                  type="success"
                  showIcon
                  style={{ marginBottom: 16 }}
                />
              )}
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

const getDataTypeColor = (type) => {
  switch (type) {
    case 'sft': return '#1890ff';
    case 'cot': return '#722ed1';
    case 'rlhf': return '#fa8c16';
    default: return '#d9d9d9';
  }
};

const getLabelText = (label) => {
  switch (label) {
    case 'accept': return 'Chấp Nhận';
    case 'reject': return 'Từ Chối';
    case 'modify': return 'Đã Sửa';
    default: return label;
  }
};

const getLabelColor = (label) => {
  switch (label) {
    case 'accept': return '#52c41a';
    case 'reject': return '#ff4d4f';
    case 'modify': return '#1890ff';
    default: return '#d9d9d9';
  }
};

export default Statistics;
