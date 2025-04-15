import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom'; 
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

  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(initialSelectedFile);
  const [sheets, setSheets] = useState([]);
  const [selectedSheet, setSelectedSheet] = useState(initialSelectedSheet);
  const [predictionData, setPredictionData] = useState([]);
  const [hasDeviceNumber, setHasDeviceNumber] = useState(true);

  // Fetch files, sheets, and predictions
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
            const predictionResponse = await fetch(`http://localhost:5001/em_predict_data/${selectedFile}/${selectedSheet}`);
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
  }, [selectedFile, selectedSheet]);

  // Handle file selection change
  const handleFileSelectChange = (event) => {
    const file = event.target.value;
    setSelectedFile(file);
    setSelectedSheet('');
    navigate(`?file=${file}&sheet=`);
  };

  // Handle sheet selection change
  const handleSheetSelectChange = (event) => {
    const sheet = event.target.value;
    setSelectedSheet(sheet);
    navigate(`?file=${selectedFile}&sheet=${sheet}`);
  };

  return (
    <div className="predictions-container">
      <h1>Predictions for {selectedFile} - {selectedSheet}</h1>

      <div className="dropdown-container">
        <FileSelector files={files} selectedFile={selectedFile} onFileChange={handleFileSelectChange} />
        {selectedFile && (
          <SheetSelector sheets={sheets} selectedSheet={selectedSheet} onSheetChange={handleSheetSelectChange} />
        )}
      </div>

      {selectedFile && selectedSheet && (
        <div className="churn-container">
          <div>
            <SummaryPanel filteredPredictionData={predictionData} />
          </div>

          <PredictionTable 
            predictionData={predictionData} 
            hasDeviceNumber={hasDeviceNumber} 
          />
        </div>
      )}
      
      {selectedFile && selectedSheet && <ModelInfo selectedFile={selectedFile} selectedSheet={selectedSheet} />}
    </div>
  );  
};

export default Predictions;
