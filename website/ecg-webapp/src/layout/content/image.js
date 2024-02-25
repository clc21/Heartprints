import React from 'react';
import { Image } from 'antd';

const ImageApp = ({ imageUrl }) => (
  <Image
    width={'100%'}
    src={imageUrl}
  />
);
export default ImageApp;