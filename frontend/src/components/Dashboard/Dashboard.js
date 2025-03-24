import React, { useState, useEffect } from 'react';
import './Dashboard.css';
import FileSelector from '../FileSelector';
import SheetSelector from '../SheetSelector';
import Top5Apps from './Top5Apps';
import AgeRangeChart from './AgeRangeChart';
import ModelFrequencyChart from './ModelTypeChart';
import CarrierChart from './PhoneCarrierChart';
import DefectsChart from './DefectsChart';
import ParamCorrChart from './ParamCorrChart';
import FeatureImportanceChart from './FeatureImportanceChart';

const Dashboard = () => {
  const [files, setFiles] = useState([]); // State to hold file list
  const [selectedFile, setSelectedFile] = useState(''); // Default to empty string for "Choose a file"
  const [sheets, setSheets] = useState([]); // State to hold sheet names
  const [selectedSheet, setSelectedSheet] = useState(''); // Default to empty string for "Choose a sheet"
  const [columns, setColumns] = useState([]); // State to store column names

  // Fetch files, sheets, and columns together
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch files
        const fileResponse = await fetch('http://localhost:5001/get_files');
        const fileData = await fileResponse.json();
        if (fileData.files) {
          setFiles(fileData.files); // Store files in state
        }

        // Fetch sheets if a file is selected
        if (selectedFile !== '') {
          const sheetResponse = await fetch(`http://localhost:5001/get_sheets/${selectedFile}`);
          const sheetData = await sheetResponse.json();
          if (sheetData.sheets) {
            setSheets(sheetData.sheets); // Store sheet names in state
          }

          // Fetch columns if a sheet is selected
          if (selectedSheet !== '') {
            const columnResponse = await fetch(`http://localhost:5001/get_all_columns/${selectedFile}/${selectedSheet}`);
            const columnData = await columnResponse.json();
            if (columnData.columns) {
              setColumns(columnData.columns); // Store columns in state
            }
          }
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, [selectedFile, selectedSheet]); // Runs when selectedFile or selectedSheet changes

  // Handle file selection from dropdown
  const handleFileSelectChange = (event) => {
    setSelectedFile(event.target.value); // Update the selected file
    setSelectedSheet(''); // Reset the sheet selection when the file changes
  };

  // Handle sheet selection from dropdown
  const handleSheetSelectChange = (event) => {
    setSelectedSheet(event.target.value); // Update the selected sheet
  };

  // Function to open any URL in a new window
  const openWindow = (url) => {
    window.open(url, '_blank'); // Opens the provided URL in a new tab
  };

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <h1>Welcome to Your Dashboard</h1>
        <p>Select the uploaded file & sheet to view your data</p>
      </header>

      <div className="dropdown-container">
        <FileSelector files={files} selectedFile={selectedFile} onFileChange={handleFileSelectChange} />
        {selectedFile && <SheetSelector sheets={sheets} selectedSheet={selectedSheet} onSheetChange={handleSheetSelectChange} />}
      </div>

      {selectedFile && selectedSheet && (
        <div>
          <h2>Predictions</h2>
          <div className="info-container">
            <div className='predictions-container'>
              <FeatureImportanceChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet}/>
            </div>
          </div>
        </div>
      )}
      
      {selectedFile && selectedSheet && (
        <div>
          <h2>Summary</h2>
          <div className="info-container">
            {columns.includes("App Usage") && (
              <Top5Apps openWindow={openWindow} />
            )}

            {columns.includes("Age Range") && (
              <AgeRangeChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />
            )}

            {columns.includes("Model") && (
              <ModelFrequencyChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />
            )}

            {columns.includes("sim_info") && (
              <CarrierChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />
            )}

            {columns.includes("Type") && (
              <DefectsChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />
            )}

            {columns.includes("Sale Channel") && (
              <ParamCorrChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />
            )}

          </div>
        </div>
      )}

    </div>
  );
};

export default Dashboard;
