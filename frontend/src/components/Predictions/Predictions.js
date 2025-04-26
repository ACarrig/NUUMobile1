import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom'; 
import FileUploadModal from './FileUploadModal';
import FileSelector from '../FileSelector';
import SheetSelector from '../SheetSelector';
import SummaryPanel from './SummaryPanel';
import ModelInfo from './ModelInfo';
import PredictionTable from './PredictionTable';
import "./Predictions.css";

const Predictions = () => {
  useEffect(() => {
    document.title = "Predictions - Churn Predictor";
  }, []);
  
  // Router hooks to access current location and navigation
  const location = useLocation();
  const navigate = useNavigate();
  
  // Get query parameters from URL
  const queryParams = new URLSearchParams(location.search);
  const initialSelectedFile = queryParams.get('file') || '';
  const initialSelectedSheet = queryParams.get('sheet') || '';
  const initialSelectedModel = queryParams.get('model') || 'ensemble';

  // State management
  const [showUploadModal, setShowUploadModal] = useState(false); // Controls upload modal visibility
  const [files, setFiles] = useState([]); // List of available files
  const [selectedFile, setSelectedFile] = useState(initialSelectedFile); // Currently selected file
  const [sheets, setSheets] = useState([]); // List of sheets in selected file
  const [selectedSheet, setSelectedSheet] = useState(initialSelectedSheet); // Currently selected sheet
  const [selectedModel, setSelectedModel] = useState(initialSelectedModel);

  // Model options for dropdown
  const modelOptions = [
    { value: 'xgb', label: 'XGBoost Model' },
    { value: 'ensemble', label: 'Ensemble Model' },
    { value: 'nn', label: 'Neural Network (MLPC)' },
  ];

  // Fetches list of available files from backend
  const fetchFiles = async () => {
    const response = await fetch('http://localhost:5001/get_files');
    const data = await response.json();
    if (data.files) {
      setFiles(data.files);
    }
  };

  // Fetches list of sheets for a given file from backend
  const fetchSheets = async (file) => {
    const response = await fetch(`http://localhost:5001/get_sheets/${file}`);
    const data = await response.json();
    if (data.sheets) {
      setSheets(data.sheets);
    }
  };

  // Effect hook to load initial data
  useEffect(() => {
    const loadData = async () => {
      try {
        await fetchFiles(); // Get all available files

        // If there's a selected file in URL, fetch its sheets
        if (selectedFile) {
          await fetchSheets(selectedFile);
        }
      } catch (error) {
        console.error('Error loading data:', error);
      }
    };

    loadData();
  }, [selectedFile]); // Runs when component mounts or selectedFile changes

  // Callback for successful file upload
  const handleUploadSuccess = fetchFiles;

  // Handles file selection change
  const handleFileSelectChange = (event) => {
    const file = event.target.value;
    setSelectedFile(file);
    setSelectedSheet(''); // Reset sheet selection when file changes
    navigate(`?file=${file}&sheet=`); // Update URL with new selection
  };

  // Handles sheet selection change
  const handleSheetSelectChange = (event) => {
    const sheet = event.target.value;
    setSelectedSheet(sheet);
    navigate(`?file=${selectedFile}&sheet=${sheet}`); // Update URL with new selection
  };

  // Handles model selection change
  const handleModelSelectChange = (event) => {
    const model = event.target.value;
    setSelectedModel(model);
    navigate(`?file=${selectedFile}&sheet=${selectedSheet}&model=${model}`); // Update URL with new model selection
  };

  return (
    <div className="predictions-container">
      {/* Header section with title and upload button */}
      <div className='header'>
        <h1>Predictions for {selectedFile} - {selectedSheet}</h1>
        <button 
          onClick={() => setShowUploadModal(true)}
          className="upload-new-button">
          Upload New File
        </button>
      </div>

      {/* Dropdown selection area */}
      <div className="dropdown-container">
        {/* File selection dropdown */}
        <FileSelector 
          files={files} 
          selectedFile={selectedFile} 
          onFileChange={handleFileSelectChange} 
        />
        
        {/* Sheet selection dropdown (only shown when file is selected) */}
        {selectedFile && (
          <SheetSelector 
            sheets={sheets} 
            selectedSheet={selectedSheet} 
            onSheetChange={handleSheetSelectChange} 
          />
        )}
        
        {/* Model selection dropdown (only shown when file and sheet are selected) */}
        {selectedFile && selectedSheet && (
          <div className="model-dropdown-container">
            <label htmlFor="model-select">Model:</label>
            <select 
              id="model-select"
              value={selectedModel}
              onChange={handleModelSelectChange}
            >
              {modelOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Main content area (only shown when file and sheet are selected) */}
      {selectedFile && selectedSheet && (
        <div className="churn-container">
          {/* Summary panel showing aggregated predictions */}
          <SummaryPanel 
            selectedFile={selectedFile}
            selectedSheet={selectedSheet}
            selectedModel={selectedModel}
          />
          
          {/* Table showing detailed predictions */}
          <PredictionTable 
            selectedFile={selectedFile}
            selectedSheet={selectedSheet}
            selectedModel={selectedModel}
          />
        </div>
      )}

      {/* Model information section (only shown when file and sheet are selected) */}
      {selectedFile && selectedSheet && (
        <ModelInfo 
          selectedFile={selectedFile} 
          selectedSheet={selectedSheet} 
          selectedModel={selectedModel} 
        />
      )}

      {/* File upload modal (shown conditionally) */}
      {showUploadModal && (
        <FileUploadModal 
          onClose={() => setShowUploadModal(false)}
          onUploadSuccess={handleUploadSuccess}
        />
      )}
    </div>
  );  
};

export default Predictions;