import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './FileUpload.css';

function FileUpload() {
  const [files, setFiles] = useState([]); // Use array to store multiple files
  const navigate = useNavigate();

  // Handle file selection (from input)
  const handleFileChange = async (event) => {
    const selectedFiles = Array.from(event.target.files); // Convert FileList to an array
    setFiles((prevFiles) => [...prevFiles, ...selectedFiles]); // Add selected files to the state
  };

  // Handle drag over event (prevent default behavior)
  const handleDragOver = (event) => {
    event.preventDefault();
  };

  // Handle drag enter event (change background color)
  const handleDragEnter = () => {
    document.getElementById('file-drop-area').style.backgroundColor = '#e0e0e0';
  };

  // Handle drag leave event (reset background color)
  const handleDragLeave = () => {
    document.getElementById('file-drop-area').style.backgroundColor = '#f0f0f0';
  };

  // Handle file drop (set the dropped files and update background color)
  const handleDrop = async (event) => {
    event.preventDefault();
    const droppedFiles = Array.from(event.dataTransfer.files); // Get dropped files as an array
    setFiles((prevFiles) => [...prevFiles, ...droppedFiles]); // Add dropped files to the state
    document.getElementById('file-drop-area').style.backgroundColor = '#f0f0f0';
  };

  // Remove file from the list
  const removeFile = (index) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  // Handle file upload to backend (multiple files)
  const handleFileUpload = async () => {
    if (files.length === 0) {
      alert('No files selected!');
      return;
    }

    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file); // Append each file to the FormData object
    });

    try {
      const response = await fetch('http://localhost:5001/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        alert('Files uploaded successfully');
        console.log(data);
        navigate('/analysis'); // Redirect to analysis page
        setFiles([]); // Reset files after upload
      } else {
        alert('File upload failed');
      }
    } catch (error) {
      console.error('Error uploading files:', error);
      alert('Error uploading files');
    }
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
          onChange={handleFileChange}
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
            <button onClick={() => removeFile(index)} className="remove-file">
              x
            </button>
          </div>
        ))}
      </div>

      {files.length > 0 && (
        <button onClick={handleFileUpload} className="confirm-button">
          Confirm Upload {files.length} {files.length === 1 ? 'File' : 'Files'}
        </button>
      )}

      {/* Footer */}
      <footer className="footer">
        <p>2025 Nuu Mobile. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default FileUpload;
