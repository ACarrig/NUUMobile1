import React, { useState, useEffect } from 'react';
import './Dashboard.css';
import FileSelector from '../FileSelector';
import SheetSelector from '../SheetSelector';
import Top5Apps from './Top5Apps';
import AgeRangeChart from './AgeRangeChart';
import ModelFrequencyChart from './ModelTypeChart';
import SimInfo from './SimInfoChart';
import SlotsChart from './SlotsChart';
import DefectsChart from './DefectsChart';
import CorrMapChart from './CorrMapChart';
import ParamCorrChart from './ParamCorrChart';
import FeatureImportanceChart from './FeatureImportanceChart';
import MonthlySalesChart from './MonthlySalesChart';

const Dashboard = () => {
  const [files, setFiles] = useState([]); // State to hold file list
  const [selectedFile, setSelectedFile] = useState(''); // Default to empty string for "Choose a file"
  const [sheets, setSheets] = useState([]); // State to hold sheet names
  const [selectedSheet, setSelectedSheet] = useState(''); // Default to empty string for "Choose a sheet"
  const [columns, setColumns] = useState([]); // State to store column names

  // Fetch files
  useEffect(() => {
    const fetchFiles = async () => {
      try {
        const fileResponse = await fetch('http://localhost:5001/get_files');
        const fileData = await fileResponse.json();
        if (fileData.files) {
          setFiles(fileData.files);
        }
      } catch (error) {
        console.error('Error fetching files:', error);
      }
    };
  
    fetchFiles();
  }, []); // Runs only once when the component mounts
  
  // Fetch Sheets of the selected file
  useEffect(() => {
    const fetchSheets = async () => {
      if (!selectedFile) return; // Only fetch sheets if a file is selected
  
      try {
        const sheetResponse = await fetch(`http://localhost:5001/get_sheets/${selectedFile}`);
        const sheetData = await sheetResponse.json();
        if (sheetData.sheets) {
          setSheets(sheetData.sheets);
        }
      } catch (error) {
        console.error('Error fetching sheets:', error);
      }
    };
  
    fetchSheets();
  }, [selectedFile]); // Runs when selectedFile changes
  
  // Fetch columns of the selected sheet of the selected file
  useEffect(() => {
    const fetchColumns = async () => {
      if (!selectedFile || !selectedSheet) return; // Only fetch columns if both file and sheet are selected
  
      try {
        const columnResponse = await fetch(`http://localhost:5001/get_all_columns/${selectedFile}/${selectedSheet}`);
        const columnData = await columnResponse.json();
        if (columnData.columns) {
          setColumns(columnData.columns);
        }
      } catch (error) {
        console.error('Error fetching columns:', error);
      }
    };
  
    fetchColumns();
  }, [selectedFile, selectedSheet]); // Runs when selectedFile or selectedSheet changes
  
  // Handle file selection from dropdown
  const handleFileSelectChange = (event) => {
    setSelectedFile(event.target.value); // Update the selected file
    setSelectedSheet(''); // Reset the sheet selection when the file changes
    setColumns([]); // Clear previous columns
  };

  // Handle sheet selection from dropdown
  const handleSheetSelectChange = (event) => {
    setSelectedSheet(event.target.value); // Update the selected sheet
    setColumns([]); // Clear previous columns
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

      {selectedFile && selectedSheet && columns.length> 0 && (
        <div>
          <h2>Predictions</h2>
          <div className="info-container">
            <div className='predictions-container'>
              <FeatureImportanceChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet}/>
            </div>
          </div>
        </div>
      )}
      
      {selectedFile && selectedSheet && columns.length> 0 && (
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
              <SimInfo openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />
            )}

            {columns.includes("Slot 1") && columns.includes("Slot 2") && (
              <SlotsChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />
            )}

            {(columns.includes("Churn") || columns.includes("Type")) && (
              <CorrMapChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />
            )}

            {columns.includes("Type") && (
              <ParamCorrChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />
            )}

            {columns.includes("Sale Channel") && columns.includes("Month") && (
              <MonthlySalesChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />
            )}

          </div>

          <h2>Summary of Returns</h2>
          <div className="info-container">
            {columns.includes("Type") && columns.includes("Defect / Damage type") && (
                <DefectsChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />
              )}
          </div>

        </div>
      )}

    </div>
  );
};

export default Dashboard;
