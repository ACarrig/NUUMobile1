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
  const [top5Apps, setTop5Apps] = useState({});  // State for top 5 apps
  const [ageRange, setAgeRange] = useState([]); // State to store age range frequency data
  const [modelFrequency, setModelFrequency] = useState([]); // State for model frequency data
  const [top5Carriers, setTop5Carriers] = useState({});

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

  // Fetch Age Range from the selected file and sheet
  useEffect(() => {
    if (selectedFile && selectedSheet) {
      // Only attempt to fetch age range if "Age Range" column exists
      if (columns.includes("Age Range")) {
        const fetchAgeRange = async () => {
          try {
            console.log(`Fetching age range for file: ${selectedFile}, sheet: ${selectedSheet}`);
            const response = await fetch(`http://localhost:5001/get_age_range/${selectedFile}/${selectedSheet}`);
            const data = await response.json();  // Parse the response JSON    
            if (data.age_range_frequency) {
              setAgeRange(data.age_range_frequency); // Store frequency data in state
            }
          } catch (error) {
            console.error(`Error fetching age range: ${error}`);
          }
        };
    
        fetchAgeRange();
      } else {
        console.log("Age Range column not found. Skipping data fetch.");
      }
    }
  }, [selectedFile, selectedSheet, columns]);  // Runs when selectedFile, selectedSheet, or columns changes

  // Fetch Model Frequency from the selected file and sheet
  useEffect(() => {
    if (selectedFile && selectedSheet) {
      // Only attempt to fetch model frequency if "Model" column exists
      if (columns.includes("Model")) {
        const fetchModelFrequency = async () => {
          try {
            console.log(`Fetching model frequency for file: ${selectedFile}, sheet: ${selectedSheet}`);
            const response = await fetch(`http://localhost:5001/get_top5_model_type/${selectedFile}/${selectedSheet}`);
            const data = await response.json();  // Parse the response JSON
            if (data.model) {
              setModelFrequency(data.model); // Store model frequency data in state
            }
          } catch (error) {
            console.error(`Error fetching model frequency: ${error}`);
          }
        };
    
        fetchModelFrequency();
      } else {
        console.log("Model column not found. Skipping data fetch.");
      }
    }
  }, [selectedFile, selectedSheet, columns]); // Runs when selectedFile, selectedSheet, or columns changes

  // Fetch top 5 most used carrier name
  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const fetchTop5Carriers = async () => {
        try {
          const response = await fetch(`http://localhost:5001/get_top5_carrier_name/${selectedFile}/${selectedSheet}`);
          const data = await response.json();  // Parse the response JSON
          if (data.carrier) {  // Assuming the correct key is top_5_apps
            setTop5Carriers(data.carrier);  // Store the top 5 apps in state
          }
        } catch (error) {
          console.error(`Error fetching top 5 apps: ${error}`);
        }
      };

      fetchTop5Carriers();
    }
  }, [selectedFile, selectedSheet]); // Runs when selectedFile or selectedSheet changes

  // Fetch top 5 most used apps
  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const fetchTop5Apps = async () => {
        try {
          const response = await fetch('http://localhost:5001/top5apps');
          const data = await response.json();  // Parse the response JSON
          if (data && data.top_5_apps) {  // Assuming the correct key is top_5_apps
            setTop5Apps(data.top_5_apps);  // Store the top 5 apps in state
          }
        } catch (error) {
          console.error(`Error fetching top 5 apps: ${error}`);
        }
      };

      fetchTop5Apps();
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

  // Helper function to get the highest age range
  const getHighestAgeRange = (ageRange) => {
    const maxCount = Math.max(...Object.values(ageRange));
    const maxAge = Object.keys(ageRange).find(age => ageRange[age] === maxCount);
    return `${maxAge}`;
  };

  // Helper function to get the lowest age range
  const getLowestAgeRange = (ageRange) => {
    const minCount = Math.min(...Object.values(ageRange));
    const minAge = Object.keys(ageRange).find(age => ageRange[age] === minCount);
    return `${minAge}`;
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
          <Top5Apps top5Apps={top5Apps} openWindow={openWindow} />
          )}

          {columns.includes("Age Range") && (
            <AgeRangeChart ageRange={ageRange} getHighestAgeRange={getHighestAgeRange} getLowestAgeRange={getLowestAgeRange} openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />
          )}

          {columns.includes("Model") && (
          <ModelFrequencyChart modelFrequency={modelFrequency} openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />
          )}

          {columns.includes("sim_info") && (
          <CarrierChart top5Carriers={top5Carriers} openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />
          )}

        </div>
      )}

    </div>
  );
};

export default Dashboard;
