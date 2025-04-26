import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './FileUpload.css';

function FileUpload() {
  useEffect(() => {
    document.title = "Upload - Churn Predictor";
  }, []);
  
  const [files, setFiles] = useState([]); // Files selected for upload
  const [uploadedFiles, setUploadedFiles] = useState([]); // Files fetched from server
  const [showModal, setShowModal] = useState(false); // For showing the confirmation modal
  const [fileToRemove, setFileToRemove] = useState(null); // The file that needs to be removed
  const navigate = useNavigate();

  // Fetch files from the backend when the component mounts
  const fetchUploadedFiles = async () => {
    try {
      const response = await fetch('http://localhost:5001/get_files');
      if (response.ok) {
        const data = await response.json();
        setUploadedFiles(data.files); // Set the fetched files into state
      } else {
        console.error('Failed to fetch files');
      }
    } catch (error) {
      console.error('Error fetching files:', error);
    }
  };

  useEffect(() => {
    fetchUploadedFiles(); // Call fetchUploadedFiles when component mounts
  }, []); // Empty dependency array to run once when the component mounts
  
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

  // Remove file from the list of selected files
  const removeFile = (index) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  const confirmRemoveFile = (index) => {
    setFileToRemove(uploadedFiles[index]); // Set the file to be removed
    setShowModal(true); // Show the confirmation modal
  };

  const removeUploadedFile = async () => {
    const fileToRemoveName = fileToRemove.name;
  
    try {
      const response = await fetch('http://localhost:5001/delete_file/' + fileToRemoveName, {
        method: 'DELETE',
      });
  
      if (response.ok) {
        const data = await response.json();
        alert(data.message);  // Show success message
        setUploadedFiles(uploadedFiles.filter((file) => file.name !== fileToRemoveName)); // Remove file from state
        setShowModal(false); // Close the modal
      } else {
        const errorData = await response.json(); // Attempt to parse the error message
        console.log('Error: ' + errorData.message); // Show error message from backend
      }
    } catch (error) {
      console.log('Error deleting file: ' + error.message);  // Handle any unexpected errors
    }
  };  

  const cancelRemoveFile = () => {
    setShowModal(false); // Close the modal without deleting the file
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
        navigate('/dashboard'); // Redirect to analysis page
        setFiles([]); // Reset files after upload
        fetchUploadedFiles(); // Re-fetch the list of uploaded files
      } else {
        console.log('File upload failed');
      }
    } catch (error) {
      console.error('Error uploading files:', error);
      console.log('Error uploading files');
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
          <div className="icon-container">
            <span className="upload-icon iconify" data-icon="cuida:upload-outline" data-inline="false"></span>
            Upload Files
          </div>
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

      <div className="uploaded-files">
        <div className="icon-container">
          <span className="file-icon iconify" data-icon="mdi:file" data-inline="false"></span>
          <h3>Uploaded Files</h3>
        </div>
        {uploadedFiles.length > 0 ? (
          <ul>
            {uploadedFiles.map((file, index) => (
              <li key={index} className="file-preview">
                <span>{file.name}</span>
                <button onClick={() => confirmRemoveFile(index)} className="remove-file">
                  <span className="remove-icon iconify" data-icon="mdi:cancel-bold" data-inline="false"></span>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p>No files uploaded yet.</p>
        )}
      </div>

      {/* Modal for confirmation */}
      {showModal && (
        <div className="confirm-delete-modal-overlay">
          <div className="confirm-delete-modal">
            <h3>Are you sure you want to delete the file: {fileToRemove?.name}?</h3>
            <div className="confirm-delete-modal-actions">
              <button onClick={removeUploadedFile} className="confirm-button">
                <div className="icon-container">
                  <span className="delete-icon iconify" data-icon="material-symbols:delete-outline-rounded" data-inline="false"></span>
                  Delete
                </div>
              </button>

              <button onClick={cancelRemoveFile} className="cancel-button">
                <div className="icon-container">
                  <span className="delete-icon iconify" data-icon="material-symbols:cancel-outline-rounded" data-inline="false"></span>
                  Cancel
                </div>
              </button>

            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default FileUpload;
