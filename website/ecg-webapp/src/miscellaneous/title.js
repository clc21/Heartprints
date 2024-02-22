import React from 'react';
import { Typography } from 'antd';
const { Title } = Typography;
const TitleApp = () => (
  <>
    <Title style={{ background : '#001529', 
                    color: 'white', 
                    textAlign: 'left', 
                    padding: '0px',
                    margin: '0',
                    display: 'flex',
                    justifyContent: 'flex-start'}}>ECGPro</Title>
  </>
);
export default TitleApp;