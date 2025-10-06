import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Layout, Menu } from 'antd';
import {
  HomeOutlined,
  FolderOutlined,
  FileTextOutlined,
  DatabaseOutlined,
  TagsOutlined,
  BarChartOutlined,
  DownloadOutlined
} from '@ant-design/icons';
import { useNavigate, useLocation } from 'react-router-dom';

import HomePage from './components/HomePage';
import TopicManagement from './components/TopicManagement';
import DocumentManagement from './components/DocumentManagement';
import DataGeneration from './components/DataGeneration';
import DataLabeling from './components/DataLabeling';
import Statistics from './components/Statistics';
import DataExport from './components/DataExport';

import './App.css';

const { Header, Sider, Content } = Layout;

function App() {
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    {
      key: '/',
      icon: <HomeOutlined />,
      label: 'Trang chủ',
    },
    {
      key: '/topics',
      icon: <FolderOutlined />,
      label: 'Quản lý chủ đề',
    },
    {
      key: '/documents',
      icon: <FileTextOutlined />,
      label: 'Quản lý tài liệu',
    },
    {
      key: '/generate',
      icon: <DatabaseOutlined />,
      label: 'Sinh dữ liệu',
    },
    {
      key: '/label',
      icon: <TagsOutlined />,
      label: 'Duyệt dữ liệu',
    },
    {
      key: '/stats',
      icon: <BarChartOutlined />,
      label: 'Thống kê',
    },
    {
      key: '/export',
      icon: <DownloadOutlined />,
      label: 'Xuất dữ liệu',
    },
  ];

  const handleMenuClick = ({ key }) => {
    navigate(key);
  };

  return (
    <Layout className="main-layout">
      <Header>
        <div className="logo">LegalSLM</div>
        <h1 className="header-title" style={{ display: 'inline-block', marginLeft: 20 }}>
          Tool Tạo Dữ Liệu Finetune
        </h1>
      </Header>
      
      <Layout>
        <Sider width={250} style={{ background: '#fff' }}>
          <Menu
            mode="inline"
            selectedKeys={[location.pathname]}
            items={menuItems}
            onClick={handleMenuClick}
            style={{ height: '100%', borderRight: 0 }}
          />
        </Sider>
        
        <Layout style={{ padding: '0 24px 24px' }}>
          <Content className="content-wrapper">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/topics" element={<TopicManagement />} />
              <Route path="/documents" element={<DocumentManagement />} />
              <Route path="/generate" element={<DataGeneration />} />
              <Route path="/label" element={<DataLabeling />} />
              <Route path="/stats" element={<Statistics />} />
              <Route path="/export" element={<DataExport />} />
            </Routes>
          </Content>
        </Layout>
      </Layout>
    </Layout>
  );
}

export default App;
