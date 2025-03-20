import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom'; 
import FileSelector from '../components/FileSelector';
import SheetSelector from '../components/SheetSelector';
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
        // Fetch files
        const fileResponse = await fetch('http://localhost:5001/get_files');
        const fileData = await fileResponse.json();
        if (fileData.files) {
          setFiles(fileData.files); // Store files in state
        }

        if (selectedFile !== '') {
          // Fetch sheets if a file is selected
          const sheetResponse = await fetch(`http://localhost:5001/get_sheets/${selectedFile}`);
          const sheetData = await sheetResponse.json();
          if (sheetData.sheets) {
            setSheets(sheetData.sheets); // Store sheet names in state
          }

          // Fetch predictions if a sheet is selected
          if (selectedSheet !== '') {
            const predictionResponse = await fetch(`http://localhost:5001/predict_data/${selectedFile}/${selectedSheet}`);
            const predictionData = await predictionResponse.json();
            if (predictionData.predictions) {
              setPredictionData(predictionData.predictions);

              // Check if "Device number" exists in the predictions
              setHasDeviceNumber("Device number" in predictionData.predictions[0]);
            }
          }
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, [selectedFile, selectedSheet]); // Re-fetch when selectedFile or selectedSheet changes

  // Handle file selection change
  const handleFileSelectChange = (event) => {
    const file = event.target.value;
    setSelectedFile(file);
    setSelectedSheet(''); // Reset sheet selection
    navigate(`?file=${file}&sheet=`); // Update URL with the new file and reset sheet query parameter
  };

  // Handle sheet selection change
  const handleSheetSelectChange = (event) => {
    const sheet = event.target.value;
    setSelectedSheet(sheet);
    navigate(`?file=${selectedFile}&sheet=${sheet}`); // Update URL with the new sheet
  };

  return (
    <div>
      <h1>Predictions for {selectedFile} - {selectedSheet}</h1>

      <div className="dropdown-container">
        {/* File and Sheet Selection */}
        <FileSelector files={files} selectedFile={selectedFile} onFileChange={handleFileSelectChange} />
        {selectedFile && (
          <SheetSelector
            sheets={sheets}
            selectedSheet={selectedSheet}
            onSheetChange={handleSheetSelectChange}
          />
        )}
      </div>
      
      {selectedFile && selectedSheet && (
         <div className="table-container">
          {predictionData.length > 0 ? (
            <table>
              <thead>
                <tr>
                  <th>Row Index</th>
                  {hasDeviceNumber && <th>Device Number</th>}
                  <th>Churn Prediction</th>
                </tr>
              </thead>
              <tbody>
                {predictionData.map((prediction, index) => (
                  <tr key={index}>
                    <td>{prediction["Row Index"]}</td>
                    {hasDeviceNumber && <td>{prediction["Device number"]}</td>}
                    <td>{prediction["Churn Prediction"]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p>Loading predictions...</p>
          )}
        </div>
      )}
     
    </div>
  );
};

export default Predictions;
