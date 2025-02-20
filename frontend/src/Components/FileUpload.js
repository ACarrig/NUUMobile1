import React, { useState } from 'react';

function FileUpload() {
  const [file, setFile] = useState(null);

  // Handle file selection (from input)
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
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
  const handleDrop = (event) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
    }
    document.getElementById('file-drop-area').style.backgroundColor = '#f0f0f0';
  };

  // Change background color on drag over
  const handleDragEnter = () => {
    document.getElementById('file-drop-area').style.backgroundColor = '#e0e0e0';
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
