import React from "react";
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './Dropzone.css';

const Dropzone = () => {
  const [prediction, setPrediction] = useState(null);
  const { getRootProps, getInputProps } = useDropzone({
    accept: 'image/*',
    onDrop: async acceptedFiles => {
      // Send the image file to the server
      const response = await axios.post('http://localhost:5000/predict', {
        image: acceptedFiles[0]
      });

      // Display the prediction
      setPrediction(response.data.class);
    }
  });

  return (
    <div>
      <div {...getRootProps({ className: 'dropzone' })}>
        <input {...getInputProps()} />
        <p>Drag 'n' drop some files here, or click to select files</p>
      </div>
      {prediction && <p>Prediction: {prediction}</p>}
    </div>
  );
};
export default Dropzone;
