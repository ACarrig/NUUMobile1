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

  // Handle file drop (set the dropped file)
  const handleDrop = (event) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
    }
  };

  return (
    <section className="file-upload">
      <div
        className="file-drop-area"
        id="file-drop-area"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <img src="/assets/upload-icon.png" alt="Upload Icon" />
        <p>Drag & Drop your file here</p>
        <p>or</p>
        
        {/* Hidden file input */}
        <input
          type="file"
          className="upload-btn"  // The input will be hidden
          onChange={handleFileChange}
          id="upload-btn"         // Add an id to label it
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
