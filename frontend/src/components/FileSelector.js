import React from 'react';
import './Selector.css';

const FileSelector = ({ files, selectedFile, onFileChange }) => {
  return (
    <div className="file-dropdown-container">
      <label htmlFor="file-dropdown">Selected File: </label>
      <select id="file-dropdown" value={selectedFile} onChange={onFileChange}>
        <option value="">-- Choose a file --</option>
        {files.map((file, index) => (
          <option key={index} value={file.name}>{file.name}</option>
        ))}
      </select>
    </div>
  );
};

export default FileSelector;
