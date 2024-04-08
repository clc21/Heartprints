import React, { useEffect, useState } from 'react';
import { Image, Spin } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';

const ImageApp = ({ imageUrl }) => {
  const [loading, setLoading] = useState(false);

  // For now, load for 2 seconds after an image is uploaded.
  useEffect(() => {
    if (imageUrl != null) {
      setLoading(true);
      setTimeout(
        () => setLoading(false),
        3000
      );
    }
  }, [imageUrl])

  return (
    <div style={{width: '100%', height: '60%', justifyContent: 'center', alignItems: 'center', display: 'flex'}}>
    {
      loading
        ? <Spin
          indicator={
            <LoadingOutlined
              style={{
                fontSize: 24,
              }}
              spin
            />
          }
        />
        : <Image
          width={'100%'}
          src={imageUrl}
        />
    }
    </div>
  );
};
export default ImageApp;