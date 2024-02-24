import React from 'react';
import { Image } from 'antd';

const ImageApp = ({ imageUrl }) => (
  <Image
    width={310}
    src={imageUrl}
  />
);
export default ImageApp;