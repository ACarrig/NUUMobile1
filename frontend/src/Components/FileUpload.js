import React, { useState } from 'react';
import './FileUpload.css'; // Import the styles for the upload area

const FileUpload = () => {
  const [files, setFiles] = useState([]);

  // Handle drag over event (prevents default behavior)
  const handleDragOver = (e) => {
    e.preventDefault();
  };

  // Handle drag enter event (change background color)
  const handleDragEnter = () => {
    document.getElementById('file-drop-area').style.backgroundColor = '#e0e0e0';
  };

  // Handle drag leave event (reset background color)
  const handleDragLeave = () => {
    document.getElementById('file-drop-area').style.backgroundColor = '#f0f0f0';
  };

  // Handle file drop event (process dropped files)
  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files);
    setFiles((prevFiles) => [...prevFiles, ...droppedFiles]); // Add dropped files
    handleDragLeave(); // Reset background color
  };

  // Handle file selection via the input button
  const handleFileInput = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles((prevFiles) => [...prevFiles, ...selectedFiles]); // Add selected files
  };

  // Remove file from the list
  const removeFile = (index) => {
    const updatedFiles = files.filter((_, i) => i !== index);
    setFiles(updatedFiles);
  };

  // Handle file upload confirmation (you can modify this to do actual uploading)
  const handleUpload = () => {
    alert('Files uploaded successfully!');
    // Here you can send the files to your server
    setFiles([]); // Reset the files after upload
  };

  return (
    <div className="upload-container">
      <div
        id="file-drop-area"
        className="drag-drop-area"
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <p>Drag & Drop Files Here</p>
        <p>or</p>
        <input
          type="file"
          multiple
          onChange={handleFileInput}
          id="file-input"
          className="file-input"
        />
        <button
          onClick={() => document.getElementById('file-input').click()}
          className="upload-button"
        >
          Upload Files
        </button>
        <p>Supported formats: .xls, .csv</p>
      </div>

      <div className="file-previews">
        {files.map((file, index) => (
          <div key={index} className="file-preview">
            <span>{file.name}</span>
            <button
              onClick={() => removeFile(index)}
              className="remove-file"
            >
              x
            </button>
          </div>
        ))}
      </div>

      {files.length > 0 && (
        <button onClick={handleUpload} className="confirm-button">
          Confirm Upload
        </button>
      )}

       {/* Footer */}
       <footer className="footer">
        <p>2025 Nuu Mobile. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default FileUpload;
