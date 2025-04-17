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
  const location = useLocation();
  const navigate = useNavigate();
  const queryParams = new URLSearchParams(location.search);
  const initialSelectedFile = queryParams.get('file') || '';
  const initialSelectedSheet = queryParams.get('sheet') || '';

  const [showUploadModal, setShowUploadModal] = useState(false);
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(initialSelectedFile);
  const [sheets, setSheets] = useState([]);
  const [selectedSheet, setSelectedSheet] = useState(initialSelectedSheet);
  const [predictionData, setPredictionData] = useState([]);
  const [hasDeviceNumber, setHasDeviceNumber] = useState(true);
  const [selectedModel, setSelectedModel] = useState('ensemble');

  const modelOptions = [
    { value: 'ensemble', label: 'Ensemble Model' },
    { value: 'nn', label: 'Neural Network' }
  ];

  const handleUploadSuccess = async () => {
    // Refresh the file list
    const response = await fetch('http://localhost:5001/get_files');
    const data = await response.json();
    if (data.files) {
      setFiles(data.files);
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const fileResponse = await fetch('http://localhost:5001/get_files');
        const fileData = await fileResponse.json();
        if (fileData.files) {
          setFiles(fileData.files);
        }

        if (selectedFile !== '') {
          const sheetResponse = await fetch(`http://localhost:5001/get_sheets/${selectedFile}`);
          const sheetData = await sheetResponse.json();
          if (sheetData.sheets) {
            setSheets(sheetData.sheets);
          }

          if (selectedSheet !== '') {
            const endpointPrefix = selectedModel === 'ensemble' ? 'em' : 'nn';
            const predictionResponse = await fetch(
              `http://localhost:5001/${endpointPrefix}_predict_data/${selectedFile}/${selectedSheet}`
            );
            const predictionData = await predictionResponse.json();
            if (predictionData.predictions) {
              setPredictionData(predictionData.predictions);
              setHasDeviceNumber("Device number" in predictionData.predictions[0]);
            }
          }
        }
      } catch (error) {
        console.log('Error fetching data: ' + error);
      }
    };

    fetchData();
  }, [selectedFile, selectedSheet, selectedModel]);

  const handleFileSelectChange = (event) => {
    const file = event.target.value;
    setSelectedFile(file);
    setSelectedSheet('');
    navigate(`?file=${file}&sheet=`);
  };

  const handleSheetSelectChange = (event) => {
    const sheet = event.target.value;
    setSelectedSheet(sheet);
    navigate(`?file=${selectedFile}&sheet=${sheet}`);
  };

  const handleModelSelectChange = (event) => {
    setSelectedModel(event.target.value);
  };

  return (
    <div className="predictions-container">
      <div className='header'>

      <h1>Predictions for {selectedFile} - {selectedSheet}</h1>
      <button 
        onClick={() => setShowUploadModal(true)}
        className="upload-new-button">
        Upload New File
      </button>
      </div>

      <div className="dropdown-container">
        <FileSelector files={files} selectedFile={selectedFile} onFileChange={handleFileSelectChange} />
        {selectedFile && (
          <SheetSelector sheets={sheets} selectedSheet={selectedSheet} onSheetChange={handleSheetSelectChange} />
        )}
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

      {selectedFile && selectedSheet && (
        <div className="churn-container">
          <div>
            <SummaryPanel 
              filteredPredictionData={predictionData} 
              selectedFile={selectedFile}
              selectedSheet={selectedSheet}
            />
          </div>

          <PredictionTable 
            predictionData={predictionData} 
            hasDeviceNumber={hasDeviceNumber} 
          />
        </div>
      )}
      
      {selectedFile && selectedSheet && (
        <ModelInfo 
          selectedFile={selectedFile} 
          selectedSheet={selectedSheet} 
          selectedModel={selectedModel} 
        />
      )}

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