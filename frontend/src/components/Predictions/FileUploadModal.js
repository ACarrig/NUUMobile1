import React, { useState } from 'react';
import './FileUploadModal.css';

const FileUploadModal = ({ onClose, onUploadSuccess }) => {
  const [files, setFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = (event) => {
    const selectedFiles = Array.from(event.target.files);
    setFiles(selectedFiles);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const droppedFiles = Array.from(event.dataTransfer.files);
    setFiles(droppedFiles);
  };

  const removeFile = (index) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      alert('Please select files to upload');
      return;
    }

    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('http://localhost:5001/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        alert('Files uploaded successfully!');
        onUploadSuccess();
        onClose();
      } else {
        alert('File upload failed');
      }
    } catch (error) {
      console.error('Error uploading files:', error);
      console.log('Error uploading files');
    }
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <div className="modal-header">
          <h3>Upload New Files</h3>
          <button onClick={onClose} className="close-button">
            x
          </button>
        </div>
        
        <div 
          className={`drop-area ${isDragging ? 'dragging' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <p>Drag & drop files here</p>
          <p>or</p>
          <input
            type="file"
            id="file-input"
            multiple
            onChange={handleFileChange}
            accept=".xls,.xlsx,.csv"
          />
          <label htmlFor="file-input" className="browse-button">
            Browse Files
          </label>
          <p>Supported formats: .xls, .xlsx, .csv</p>
        </div>

        <div className="file-list">
          {files.map((file, index) => (
            <div key={index} className="file-item">
              <span>{file.name}</span>
              <button onClick={() => removeFile(index)} className="remove-button">
                ×
              </button>
            </div>
          ))}
        </div>

        <div className="modal-actions">
          <button onClick={onClose} className="cancel-button">
            Cancel
          </button>
          <button 
            onClick={handleUpload} 
            className="upload-button"
            disabled={files.length === 0}
          >
            Upload {files.length} File{files.length !== 1 ? 's' : ''}
          </button>
        </div>
      </div>
    </div>
  );
};

export default FileUploadModal;