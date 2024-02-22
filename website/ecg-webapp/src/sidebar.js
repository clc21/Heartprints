import React from 'react';
import { UploadOutlined, UserOutlined, VideoCameraOutlined , DownloadOutlined} from '@ant-design/icons';
import { Layout, Menu, Row, theme, Button } from 'antd';
import UploadApp from './upload';
import './App.css'
import './sidebar.css'
import ImageApp from './image';
const { Header, Content, Footer, Sider } = Layout;
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
      label:'Export'
    }
  
  ]
const SidebarApp = () => {
  const {
    token: { colorBgContainer, borderRadiusLG },
  } = theme.useToken();
  return (
    <Layout>
      <Sider
        breakpoint="lg"
        collapsedWidth="0"
        onBreakpoint={(broken) => {
          console.log(broken);
        }}
        onCollapse={(collapsed, type) => {
          console.log(collapsed, type);
        }}
      >
        <div>
        <h1>
         ECGPro   
        </h1>
        </div>
        <Menu theme="dark" mode="inline" defaultSelectedKeys={['1']} items={items} />
      </Sider>
      <Layout>
        <Content
          style={{
            margin: '24px 16px 0',
          }}
        >
          <div
            style={{
              padding: 24,
              minHeight: '100%',
              background: colorBgContainer,
              borderRadius: borderRadiusLG,
              display:'flex',
              flexDirection: 'row'
            }}
          >
            <div style={{width: '50%', display: 'flex', flexDirection:'column', justifyContent: 'center', alignItems: 'center' }}>
            <UploadApp/>
            <Button style={{borderRadius: '10px', backgroundColor:'#058d82',color: 'white', width: '97%'}}>Upload</Button>
            </div>
            <div style={{width: '50%', display: 'flex', flexDirection:'column', justifyContent: 'center', alignItems: 'center'}}>
            <ImageApp/>
            <Button style={{ borderRadius: '10px', backgroundColor: '#058d82', color: 'white', margin: '10px' }}
            icon={<DownloadOutlined style={{ color: 'white' }} />}
/>

            </div>
            
          </div>
        </Content>
      </Layout>
    </Layout>
  );
};
export default SidebarApp;