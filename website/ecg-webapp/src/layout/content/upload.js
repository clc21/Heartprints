import React, {useState} from 'react';
import { InboxOutlined } from '@ant-design/icons';
import { message, Upload } from 'antd';
import '../sidebar.css';

const { Dragger } = Upload;

const UploadApp = (props) => {
  const handleImageUpload = ({ file, onSuccess }) => {
    setTimeout(() => {
      props.updateFile(file);
      onSuccess("ok");
    }, 0);
  };
  
  const uploadProps = {
    accept: 'image/png,.pdf,.dat',
    name: 'file',
    multiple: false,
    customRequest: handleImageUpload,
    showUploadList: true,
    maxCount: 1,
    beforeUpload: file => {
      // this.setState(state => ({
      //     fileList: [file]
      // }))
      // return false;
    },
    onChange(info) {
      const { status } = info.file;
      if (status !== 'uploading') {
        console.log(info.file, info.fileList);
      }
      if (status === 'done') {
        message.success(`${info.file.name} file uploaded successfully.`);
      } else if (status === 'error') {
        message.error(`${info.file.name} file upload failed.`);
      }
    },
    onDrop(e) {
      console.log('Dropped files', e.dataTransfer.files);
    },
    onRemove: file => {
      props.removeFile();
      // this.setState(state => {
      //     return {
      //         fileList: []
      //     };
      // });
  },
  };
  
  return (
    <div style={{height:'50%', padding: '10px'}}>
      <Dragger {...uploadProps} >
        <p className="ant-upload-drag-icon">
          <InboxOutlined style={{ color: '#06a598' }}/>
        </p>
        <p className="ant-upload-text">Click or drag file to this area to upload</p>
        <p className="ant-upload-hint">
          Support for a single upload. Strictly prohibited from uploading company data or other
          banned files.
        </p>
      </Dragger>
    </div>
  );
};
export default UploadApp;