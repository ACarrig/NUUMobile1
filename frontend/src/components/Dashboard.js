import React, { useState, useEffect } from 'react';
import './Dashboard.css';
import FileSelector from '../components/FileSelector';
import SheetSelector from '../components/SheetSelector';
import Top5Apps from '../components/Top5Apps';
import AgeRangeChart from '../components/AgeRangeChart';
import ModelFrequencyChart from '../components/ModelTypeChart';
import CarrierChart from '../components/PhoneCarrierChart';

const Dashboard = () => {
  const [files, setFiles] = useState([]); // State to hold file list
  const [selectedFile, setSelectedFile] = useState(''); // Default to empty string for "Choose a file"
  const [sheets, setSheets] = useState([]); // State to hold sheet names
  const [selectedSheet, setSelectedSheet] = useState(''); // Default to empty string for "Choose a sheet"
  const [columns, setColumns] = useState([]); // State to store column names

  // Fetch files from backend
  useEffect(() => {
    const fetchFiles = async () => {
      try {
        const response = await fetch('http://localhost:5001/get_files');
        const data = await response.json();
        if (data.files) {
          setFiles(data.files); // Store the fetched files in state
        }
      } catch (error) {
        console.error('Error fetching files:', error);
      }
    };

    fetchFiles();
  }, []); // Empty dependency array ensures this runs once when the component mounts

  // Fetch sheets when a file is selected
  useEffect(() => {
    if (selectedFile !== '') {
      const fetchSheets = async () => {
        try {
          const response = await fetch(`http://localhost:5001/get_sheets/${selectedFile}`);
          const data = await response.json();
          if (data.sheets) {
            setSheets(data.sheets);  // Update the sheets state with the fetched sheet names
          } else {
            console.warn("No sheets found in the response");
          }
        } catch (error) {
          console.error('Error fetching sheets:', error);
        }
      };
  
      fetchSheets();
    } else {
      setSheets([]); // Clear sheets if no file is selected
    }
  }, [selectedFile]); // This runs every time the selectedFile changes
  
  // Fetch all the columns from the selected file and sheet
  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const fetchColumns = async () => {
        try {
          const response = await fetch(`http://localhost:5001/get_all_columns/${selectedFile}/${selectedSheet}`);
          const data = await response.json();  // Parse the response JSON
          if (data.columns) {
            setColumns(data.columns); // Store columns in state
          }
        } catch (error) {
          console.error(`Error fetching columns: ${error}`);
        }
      };

      fetchColumns();
    }
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

      <h2>Summary</h2>
      {selectedFile && selectedSheet && (
        <div className="info-container">
          {columns.includes("App Usage") && (
          // <Top5Apps top5Apps={top5Apps} openWindow={openWindow} />
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

        </div>
      )}

    </div>
  );
};

export default Dashboard;
