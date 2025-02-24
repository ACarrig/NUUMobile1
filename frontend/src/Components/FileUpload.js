import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function FileUpload() {
  const [file, setFile] = useState(null);
  const navigate = useNavigate();

  // Handle file selection (from input)
  const handleFileChange = async (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      await handleFileUpload(selectedFile); // Upload immediately after selecting
    }
  };

  // Handle drag over event (prevents default behavior)
  const handleDragOver = (event) => {
    event.preventDefault();
  };

  // Handle drag leave event (reset background color)
  const handleDragLeave = () => {
    document.getElementById('file-drop-area').style.backgroundColor = '#f0f0f0';
  };

  // Handle file drop (set the dropped file and update background color)
  const handleDrop = async (event) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
      await handleFileUpload(droppedFile); // Upload immediately after dropping
    }
    document.getElementById('file-drop-area').style.backgroundColor = '#f0f0f0';
  };

  // Change background color on drag over
  const handleDragEnter = () => {
    document.getElementById('file-drop-area').style.backgroundColor = '#e0e0e0';
  };

  // Handle file upload
  const handleFileUpload = async (fileToUpload) => {
    const formData = new FormData();
    formData.append('file', fileToUpload);

    try {
      const response = await fetch('http://localhost:5001/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        alert('File uploaded successfully');
        console.log(data);
        navigate('/analysis');
      } else {
        alert('File upload failed');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file');
    }
  };

  return (
    <section className="file-upload">
      <div
        className="file-drop-area"
        id="file-drop-area"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        style={{ backgroundColor: '#f0f0f0', padding: '20px', borderRadius: '8px', textAlign: 'center', border: '2px dashed #ccc' }}
      >
        <img src="/assets/upload-icon.png" alt="Upload Icon" />
        <p>Drag & Drop your file here</p>
        <p>or</p>
        
        {/* Hidden file input */}
        <input
          type="file"
          className="upload-btn"
          onChange={handleFileChange}
          id="upload-btn"
          style={{ display: 'none' }}
        />
        
        {/* Label for the file input button */}
        <label htmlFor="upload-btn" className="upload-btn-label">
          Upload File
        </label>

        <p>Supported file types: XLS, CSV</p>
        
        {/* Display the selected file */}
        {file && <p>Selected file: {file.name}</p>}
      </div>
    </section>
  );
}

export default FileUpload;
