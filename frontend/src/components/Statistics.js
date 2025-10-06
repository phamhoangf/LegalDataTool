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
  Alert,
  Button,
  Select,
  Modal,
  Tabs,
  List,
  Typography,
  Collapse,
  InputNumber,
  Tooltip,
  message
} from 'antd';
import {
  BarChartOutlined,
  PieChartOutlined,
  TrophyOutlined,
  ClockCircleOutlined,
  AimOutlined,
  FileTextOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';
import apiService from '../services/api';

const { Title, Text } = Typography;
const { Panel } = Collapse;
const { TabPane } = Tabs;

const Statistics = () => {
  const [stats, setStats] = useState(null);
  const [topics, setTopics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Coverage analysis states
  const [coverageModalVisible, setCoverageModalVisible] = useState(false);
  const [coverageLoading, setCoverageLoading] = useState(false);
  const [coverageData, setCoverageData] = useState(null);
  const [selectedTopicId, setSelectedTopicId] = useState(null);
  const [analyzingTopicId, setAnalyzingTopicId] = useState(null); // Track topic đang analyze
  const [coverageSettings, setCoverageSettings] = useState({
    unit_type: 'sentence',
    threshold: 0.35
  });

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async (forceRefresh = false) => {
    try {
      setLoading(true);
      
      // Thêm timestamp để force refresh cache nếu cần
      const cacheBuster = forceRefresh ? `?_t=${Date.now()}` : '';
      
      const [statsResponse, topicsResponse] = await Promise.all([
        fetch(`/api/stats${cacheBuster}`).then(res => res.json()),
        apiService.getTopics()
      ]);
      
      setStats(statsResponse);
      setTopics(topicsResponse.data);
      setError(null);
    } catch (err) {
      setError('Không thể tải dữ liệu thống kê');
      console.error('Error loading statistics:', err);
    } finally {
      setLoading(false);
    }
  };

  // Expose refresh function để các component khác có thể gọi
  const refreshStats = () => {
    loadData(true);
  };

  // Listen for refresh events từ other components
  useEffect(() => {
    const handleRefreshStats = () => {
      refreshStats();
    };
    
    window.addEventListener('refreshStats', handleRefreshStats);
    return () => window.removeEventListener('refreshStats', handleRefreshStats);
  }, []);

  const analyzeCoverage = async (topicId) => {
    try {
      setCoverageLoading(true);
      setAnalyzingTopicId(topicId);
      
      const response = await apiService.analyzeCoverage(topicId, coverageSettings);
      const data = response.data;
      
      setCoverageData(data);
      setSelectedTopicId(topicId);
      setCoverageModalVisible(true);
      message.success('Phân tích coverage hoàn thành');
      
    } catch (err) {
      console.error('Error analyzing coverage:', err);
      message.error('Lỗi phân tích coverage: ' + (err.response?.data?.error || err.message));
    } finally {
      setCoverageLoading(false);
      setAnalyzingTopicId(null);
    }
  };

  const stopCoverageAnalysis = async (topicId) => {
    try {
      await apiService.stopCoverageAnalysis(topicId);
      message.success('Đã yêu cầu dừng phân tích coverage');
      setAnalyzingTopicId(null); // Reset state khi dừng thành công
      setCoverageLoading(false);
    } catch (err) {
      console.error('Error stopping coverage analysis:', err);
      message.error('Lỗi dừng phân tích: ' + (err.response?.data?.error || err.message));
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
    {
      title: 'Hành động',
      key: 'actions',
      render: (_, record) => (
        <div>
          {coverageLoading && analyzingTopicId === record.id ? (
            <div>
              <Button
                type="primary"
                size="small"
                loading={true}
                style={{ marginRight: 8 }}
              >
                Đang phân tích...
              </Button>
              <Button
                type="default"
                size="small"
                danger
                onClick={() => stopCoverageAnalysis(record.id)}
              >
                🛑 Dừng
              </Button>
            </div>
          ) : (
            <Tooltip title="Phân tích độ bao phủ">
              <Button
                type="primary"
                size="small"
                icon={<AimOutlined />}
                onClick={() => analyzeCoverage(record.id)}
              >
                Coverage
              </Button>
            </Tooltip>
          )}
        </div>
      ),
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
        <Col xs={24} lg={16}>
          <Card title="Danh Sách Chủ Đề" size="small">
            <Table
              columns={topicColumns}
              dataSource={topics}
              rowKey="id"
              size="small"
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="Cài Đặt Coverage Analysis" size="small" style={{ marginBottom: 16 }}>
            <div style={{ marginBottom: 16 }}>
              <Text strong>Loại Unit:</Text>
              <Select
                style={{ width: '100%', marginTop: 8 }}
                value={coverageSettings.unit_type}
                onChange={(value) => setCoverageSettings({...coverageSettings, unit_type: value})}
              >
                <Select.Option value="sentence">Điều/Câu (Sentence)</Select.Option>
                <Select.Option value="paragraph">Đoạn (Paragraph)</Select.Option>
              </Select>
            </div>
            <div>
              <Text strong>Ngưỡng Similarity:</Text>
              <InputNumber
                style={{ width: '100%', marginTop: 8 }}
                min={0.1}
                max={1.0}
                step={0.1}
                value={coverageSettings.threshold}
                onChange={(value) => setCoverageSettings({...coverageSettings, threshold: value})}
              />
              <Text type="secondary" style={{ fontSize: '12px' }}>
                (0.1 - 1.0, càng cao càng khó bao phủ)
              </Text>
            </div>
          </Card>
        </Col>
      </Row>
      
      {/* Coverage Analysis và Tiến độ */}
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

      {/* Coverage Analysis Modal */}
      <Modal
        title="Phân Tích Độ Bao Phủ (Coverage Analysis)"
        visible={coverageModalVisible}
        onCancel={() => setCoverageModalVisible(false)}
        width={1000}
        footer={[
          <Button key="close" onClick={() => setCoverageModalVisible(false)}>
            Đóng
          </Button>
        ]}
      >
        {coverageData && (
          <div>
            {/* Header Info */}
            <Row gutter={16} style={{ marginBottom: 24 }}>
              <Col span={8}>
                <Card size="small">
                  <Statistic
                    title="Tổng Units"
                    value={coverageData.total_units}
                    prefix={<FileTextOutlined />}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small">
                  <Statistic
                    title="Units Được Bao Phủ"
                    value={coverageData.covered_units}
                    prefix={<CheckCircleOutlined />}
                    valueStyle={{ color: '#3f8600' }}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small">
                  <Statistic
                    title="Tỷ Lệ Bao Phủ"
                    value={coverageData.coverage_percentage.toFixed(1)}
                    suffix="%"
                    prefix={<TrophyOutlined />}
                    valueStyle={{ 
                      color: coverageData.coverage_percentage >= 70 ? '#3f8600' : '#cf1322' 
                    }}
                  />
                </Card>
              </Col>
            </Row>

            {/* Progress */}
            <div style={{ marginBottom: 24 }}>
              <Title level={5}>Tiến Độ Bao Phủ</Title>
              <Progress
                percent={Number(coverageData.coverage_percentage.toFixed(1))}
                status={coverageData.coverage_percentage >= 70 ? 'success' : 'normal'}
                strokeColor={coverageData.coverage_percentage >= 70 ? '#52c41a' : '#1890ff'}
              />
            </div>

            {/* Settings Info */}
            <Card size="small" style={{ marginBottom: 16, backgroundColor: '#f6f6f6' }}>
              <Text strong>Cài đặt phân tích: </Text>
              <Tag color="blue">
                Loại unit: {coverageData.analysis_settings?.unit_type === 'sentence' ? 'Điều/Câu' : 'Đoạn'}
              </Tag>
              <Tag color="orange">Ngưỡng: {coverageData.threshold_used}</Tag>
              <Tag color="green">Documents: {coverageData.analysis_settings?.total_documents}</Tag>
              <Tag color="purple">Câu hỏi: {coverageData.analysis_settings?.total_questions}</Tag>
              
              {coverageData.was_stopped && (
                <Tag color="red">🛑 Đã dừng tại {coverageData.processed_units}/{coverageData.total_units} units</Tag>
              )}
              {coverageData.processed_units && coverageData.processed_units !== coverageData.total_units && !coverageData.was_stopped && (
                <Tag color="orange">⚠️ Xử lý {coverageData.processed_units}/{coverageData.total_units} units</Tag>
              )}
            </Card>

            {/* Document Summary */}
            {coverageData.document_summary && (
              <div style={{ marginBottom: 24 }}>
                <Title level={5}>Bao Phủ Theo Document</Title>
                <Row gutter={16}>
                  {Object.entries(coverageData.document_summary).map(([docTitle, stats]) => (
                    <Col span={8} key={docTitle}>
                      <Card size="small">
                        <div style={{ marginBottom: 8 }}>
                          <Text strong ellipsis title={docTitle}>
                            {docTitle.length > 20 ? `${docTitle.substring(0, 20)}...` : docTitle}
                          </Text>
                        </div>
                        <Progress
                          percent={Number(stats.coverage_percentage.toFixed(1))}
                          size="small"
                          format={() => `${stats.covered_units}/${stats.total_units}`}
                        />
                        <div style={{ fontSize: '12px', color: '#666', marginTop: 4 }}>
                          Coverage: {stats.coverage_percentage.toFixed(1)}%
                        </div>
                      </Card>
                    </Col>
                  ))}
                </Row>
              </div>
            )}

            {/* Detailed Analysis */}
            <Tabs defaultActiveKey="uncovered">
              <TabPane tab={
                <span>
                  <ExclamationCircleOutlined />
                  Units Chưa Bao Phủ ({coverageData.uncovered_units})
                </span>
              } key="uncovered">
                <List
                  size="small"
                  dataSource={coverageData.units_analysis?.filter(unit => !unit.is_covered).slice(0, 10) || []}
                  renderItem={item => (
                    <List.Item>
                      <div style={{ width: '100%' }}>
                        <div style={{ marginBottom: 4 }}>
                          <Text strong>{item.document_title}</Text>
                          <Tag color="red" style={{ float: 'right' }}>
                            Similarity: {item.max_similarity?.toFixed(3) || '0.000'}
                          </Tag>
                        </div>
                        <div style={{ color: '#666' }}>
                          {item.content_preview}
                        </div>
                      </div>
                    </List.Item>
                  )}
                />
              </TabPane>
              
              <TabPane tab={
                <span>
                  <CheckCircleOutlined />
                  Units Được Bao Phủ Tốt
                </span>
              } key="covered">
                <List
                  size="small"
                  dataSource={coverageData.units_analysis?.filter(unit => unit.is_covered)
                    .sort((a, b) => b.max_similarity - a.max_similarity)
                    .slice(0, 10) || []}
                  renderItem={item => (
                    <List.Item>
                      <div style={{ width: '100%' }}>
                        <div style={{ marginBottom: 4 }}>
                          <Text strong>{item.document_title}</Text>
                          <Tag color="green" style={{ float: 'right' }}>
                            Similarity: {item.max_similarity?.toFixed(3) || '0.000'}
                          </Tag>
                        </div>
                        <div style={{ color: '#666', marginBottom: 4 }}>
                          {item.content_preview}
                        </div>
                        {item.best_question && (
                          <div style={{ fontSize: '12px', color: '#1890ff' }}>
                            📝 Câu hỏi phù hợp nhất: {item.best_question.question.substring(0, 80)}...
                          </div>
                        )}
                      </div>
                    </List.Item>
                  )}
                />
              </TabPane>
            </Tabs>
          </div>
        )}
      </Modal>
    </div>
  );
};

const getDataTypeColor = (type) => {
  switch (type) {
    case 'word_matching': return '#1890ff';
    case 'concept_understanding': return '#52c41a';
    case 'multi_paragraph_reading': return '#722ed1';
    case 'multi_hop_reasoning': return '#fa8c16';
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
