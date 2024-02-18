import React from 'react';
import { Breadcrumb, Layout, Menu, theme } from 'antd';
import UploadApp from './upload';
const { Header, Content, Footer } = Layout;
const items = [
  {
    key: '1',
    label: 'Home'
  },
  {
    key: '2',
    label: 'Processing'
  },
  {
    key: '3',
    label:'Contact'
  }

]
const LayoutApp = () => {
  const {
    token: { colorBgContainer, borderRadiusLG },
  } = theme.useToken();
  return (
    <Layout style={{ minWidth: '100vh'}}>
      <Header
        style={{
          display: 'flex',
          alignItems: 'center',
        }}
      >
        <div className="demo-logo" />
        <h1 style={{color: 'white', paddingRight: '15px', fontFamily: 'Arial'}}>ECGPro</h1>
        <Menu
          theme="dark"
          mode="horizontal"
          defaultSelectedKeys={['1']}
          items={items}
          style={{
            flex: 1,
            minWidth: 0,
          }}
        />
      </Header>
      <Content
        style={{
          padding: '0 48px',
        }}
      >
        <Breadcrumb
          style={{
            margin: '16px 0',
          }}
        >
          <Breadcrumb.Item>Home</Breadcrumb.Item>
          <Breadcrumb.Item>Start</Breadcrumb.Item>
          <Breadcrumb.Item>App</Breadcrumb.Item>
        </Breadcrumb>
        <div
          style={{
            background: colorBgContainer,
            minHeight: 500,
            padding: 24,
            borderRadius: borderRadiusLG,
          }}
        >
          <UploadApp/>
        </div>
      </Content>
      <Footer
        style={{
          textAlign: 'center',
        }}
      >
        ECGPro ©{new Date().getFullYear()} Created by ECG Babes
      </Footer>
    </Layout>
  );
};
export default LayoutApp;