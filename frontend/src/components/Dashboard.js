import React, { useState, useEffect } from 'react';
import './Dashboard.css';

const Dashboard = () => {
  const [files, setFiles] = useState([]); // State to hold file list
  const [selectedFile, setSelectedFile] = useState(''); // Default to empty string for "Choose a file"
  const [sheets, setSheets] = useState([]); // State to hold sheet names
  const [selectedSheet, setSelectedSheet] = useState(''); // Default to empty string for "Choose a sheet"
  const [columns, setColumns] = useState([]); // State to store column names
  const [top5Apps, setTop5Apps] = useState({});  // State for top 5 apps
  const [aiSummary, setAiSummary] = useState("") // State for ai summary of data
  
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
        alert('Error fetching files:', error);  // Use alert instead of console
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
            alert("No sheets found in the response");
          }
        } catch (error) {
          alert('Error fetching sheets:', error);
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
          } else {
            alert('No columns found');
          }
        } catch (error) {
          alert(`Error fetching columns: ${error}`);
        }
      };

      fetchColumns();
    }
  }, [selectedFile, selectedSheet]); // Runs when selectedFile or selectedSheet changes

  // Fetch top 5 most used apps only when file and sheet are selected
  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const fetchTop5Apps = async () => {
        try {
          const response = await fetch('http://localhost:5001/top5apps');
          const data = await response.json();  // Parse the response JSON
          if (data && data.top_5_apps) {  // Assuming the correct key is `top_5_apps`
            setTop5Apps(data.top_5_apps);  // Store the top 5 apps in state
          } else {
            alert('No data received for top 5 apps');
          }
        } catch (error) {
          alert(`Error fetching top 5 apps: ${error}`);
        }
      };

      fetchTop5Apps();
    }
  }, [selectedFile, selectedSheet]); // Runs when selectedFile or selectedSheet changes

  // Fetch the AI response
  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const aisummary = async () => {
        try {
          const response = await fetch('http://localhost:5001/app_usage_summary');
          const data = await response.json();
          if (data && data.aiSummary) {
            setAiSummary(data.aiSummary);
          } else {
            alert('No ai summary received');
          }
        } catch (error) {
          alert(`Error fetching summary: ${error}`);
        }
      };

      aisummary();
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

  // Function to format aiSummary into HTML (convert lists)
  const formatAiSummary = (summary) => {
    // Replace bullet points or numbered lists with <ul> or <ol>
    const listRegex = /(\*|\-|\d+\.)\s+/g; // Detect bullet points or numbered lists
    let formattedSummary = summary.split('\n').map(line => {
      if (line.match(listRegex)) {
        return `<ul><li>${line.replace(listRegex, '')}</li></ul>`;
      } else {
        return `<p>${line}</p>`;
      }
    }).join('');
    
    return formattedSummary;
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

            <div className="ai-summary-box">
              <h3>AI Summary of Data</h3>
              <div>
                {aiSummary ? (
                  <p>{aiSummary}</p>  // Display the summary if it's available
                ) : (
                  <p>Loading summary...</p>  // Show loading message if summary is still being fetched
                )}
              </div>
            </div>

            {/* Only show the Top 5 Most Used Apps section if "App Usage (s)" is in the columns list */}
            {columns.includes("App Usage (s)") && (
              <div className="summary-box">
                <h3>Top 5 Most Used Apps</h3>
                <ol>
                  {Object.entries(top5Apps).map(([app, usage], index) => (
                    <li key={index}>
                      {app}: {usage} hrs
                    </li>
                  ))}
                </ol>
                {/* Button to open AppData in a new window */}
                <button onClick={() => openWindow('/appdata')}>View App Data</button>
              </div>
            )}

            <div className="summary-box"></div>

          </div>
        </>
      )}
    </div>
  );
};

export default Dashboard;
