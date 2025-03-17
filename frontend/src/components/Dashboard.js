import React, { useState, useEffect } from 'react';
import './Dashboard.css';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';  // Import Recharts components

const Dashboard = () => {
  const [files, setFiles] = useState([]); // State to hold file list
  const [selectedFile, setSelectedFile] = useState(''); // Default to empty string for "Choose a file"
  const [sheets, setSheets] = useState([]); // State to hold sheet names
  const [selectedSheet, setSelectedSheet] = useState(''); // Default to empty string for "Choose a sheet"
  const [columns, setColumns] = useState([]); // State to store column names
  const [top5Apps, setTop5Apps] = useState({});  // State for top 5 apps
  const [ageRange, setAgeRange] = useState([]); // State to store age range frequency data
  const [modelFrequency, setModelFrequency] = useState([]); // State for model frequency data

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
            console.log('Data received for age range:', data);
    
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
            console.log('Data received for model frequency:', data);
    
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

  // Fetch top 5 most used apps only when file and sheet are selected
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
        <p>Manage your settings and view your data</p>
      </header>

      <div className="dropdown-container">
        <div className="file-dropdown-container">
          <label htmlFor="file-dropdown">Select File: </label>
          <select
            id="file-dropdown"
            value={selectedFile}
            onChange={handleFileSelectChange}
          >
            <option value="">Choose a file</option> {/* Empty line with a placeholder */}
            {files.map((file, index) => (
              <option key={index} value={file.name}>
                {file.name}
              </option>
            ))}
          </select>
        </div>

        {selectedFile !== '' && (
          <div className="sheet-dropdown-container">
            <label htmlFor="sheet-dropdown">Select Sheet: </label>
            <select
              id="sheet-dropdown"
              value={selectedSheet}
              onChange={handleSheetSelectChange}
            >
              <option value="">Choose a sheet</option> {/* Empty line with a placeholder */}
              {sheets.map((sheet, index) => (
                <option key={index} value={sheet}>
                  {sheet}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>
      
      {/* Only show the summary if both file and sheet are selected */}
      {selectedFile !== '' && selectedSheet !== '' && (
        <>
          <h2>Summary</h2>
          <div className="info-container">

            {/* Only show the Top 5 Most Used Apps section if "App Usage (s)" is in the columns list */}
            {columns.includes("App Usage") && (
              <div className="summary-box">
                <h3>Top 5 Most Used Apps</h3>
                <ol>
                  {Object.entries(top5Apps).sort((a, b) => b[1] - a[1]).map(([app, usage], index) => (
                    <li key={index}>
                      {app}: {usage} hrs
                    </li>
                  ))}
                </ol>
                <button onClick={() => openWindow('/appdata')}>View App Data</button>
              </div>
            )}

            {/* Age Range Frequency Chart */}
            {columns.includes("Age Range") && (
              <div className="summary-box">
                <h3>Age Range Frequency</h3>
                <div className="summary-graph">
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={Object.entries(ageRange).map(([age, count]) => ({ age, count }))}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="age" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="count" fill="#C4D600" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Display highest and lowest age range */}
                {ageRange && Object.keys(ageRange).length > 0 && (
                  <div className="age-range-info">
                    <p><strong>Highest Age Range: </strong>{getHighestAgeRange(ageRange)}</p>
                    <p><strong>Lowest Age Range: </strong>{getLowestAgeRange(ageRange)}</p>
                  </div>
                )}

                <button onClick={() => openWindow(`/agerange?file=${selectedFile}&sheet=${selectedSheet}`)}>View Age Range</button>

              </div>
            )}

            {/* Model Frequency Chart */}
            {columns.includes("Model") && (
              <div className="summary-box">
                <h3>Top 5 Most Used Models</h3>
                <div className="summary-graph">
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={Object.entries(modelFrequency)
                      .map(([model, count]) => ({ model, count })) // Convert model frequency to an array of objects
                      .sort((a, b) => b.count - a.count) // Sort by frequency in descending order
                      .slice(0, 5) // Slice to get top 5 models
                    }>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="model" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="count" fill="#C4D600" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <button onClick={() => openWindow(`/modeltype?file=${selectedFile}&sheet=${selectedSheet}`)}>View More Model Frequency</button>
              </div>
            )}

            {/* Phone Carrier Chart */}
            {columns.includes("sim_info") && (
              <div className="summary-box">
                <h3>Top 5 Most Used Phone Carriers</h3>
                <div className="summary-graph">
                  
                </div>

                <button onClick={() => openWindow(`/sim_info?file=${selectedFile}&sheet=${selectedSheet}`)}>View More Sim Info</button>
              </div>
            )}

          </div>
        </>
      )}
    </div>
  );
};

export default Dashboard;
