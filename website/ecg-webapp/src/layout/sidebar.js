import React, { useState, useEffect } from 'react';
import { UploadOutlined, UserOutlined, VideoCameraOutlined , DownloadOutlined} from '@ant-design/icons';
import { Layout, Menu, Row, theme, Button, message } from 'antd';
import UploadApp from './content/upload';
import '../App/App.css'
import './sidebar.css'
import ImageApp from './content/image';
const { Content, Sider } = Layout;
const imageUrl = process.env.PUBLIC_URL + "../../assets/heartprint.png"
const items = [
    {
      key: '1',
      label: 'Home'
    },
    {
      key: '2',
      label: 'About'
    },
    {
      key: '3',
      label:'Help'
    }
  
  ]

const SidebarApp = () => {
  const [inputFile, setInputFile] = useState(null);
  const [outputFile, setOutputFile] = useState(null);
  
  useEffect(() => {
    console.log("FILE CHANGED:", inputFile);
  }, [inputFile]);

  
  const {
    token: { colorBgContainer, borderRadiusLG },
  } = theme.useToken();

  const handleDownload = async () => {
    try {
      const response = await fetch(imageUrl);
      if (!response.ok) throw new Error('Network response was not ok');
      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.setAttribute('download', 'downloadedImage.png'); // Set the desired file name here
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  const handleFileUpload = () => {
    if (!inputFile) {
      message.error("Please select a file to upload.");
      return;
    }

    message.success("Uploading file to server for processing.");

    // TODO: for now, we set the output file to the input file.
    // Later, this should await the result of our remote processing.
    setOutputFile(inputFile);
  }


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
            {/* <h1 style={{color: 'black'}}>Welcome to ECG Pro</h1> */}
            <UploadApp updateFile={f => setInputFile(f)} removeFile={() => setInputFile(null)}/>
            <Button
              style={{
                top: '5%',
                borderRadius: '10px',
                backgroundColor:'#058d82',
                color: 'white',
                width: '97%'
              }}
              onClick={handleFileUpload}
            >
              Upload
            </Button>
            </div>
            <div style={{width: '50%', display: 'flex', flexDirection:'column', justifyContent: 'center', alignItems: 'center'}}>
            <ImageApp imageUrl={outputFile ? URL.createObjectURL(outputFile) : null}/>
            <Button style={{ borderRadius: '10px', backgroundColor: '#058d82', color: 'white', margin: '10px' }}
            onClick={handleDownload} icon={<DownloadOutlined style={{ color: 'white' }}/>}
/>

            </div>
            
          </div>
        </Content>
      </Layout>
    </Layout>
  );
};
export default SidebarApp;