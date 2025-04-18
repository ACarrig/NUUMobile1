import React, { useState, useEffect, useRef } from 'react';
import './Dashboard.css';
import FileSelector from '../FileSelector';
import SheetSelector from '../SheetSelector';

import ColumnsGraphChart from './ColumnsGraphChart';

import FeatureImportanceChart from './FeatureImportanceChart';

import AppUsageChart from './SummaryCharts/AppUsageChart';
import AgeRangeChart from './SummaryCharts/AgeRangeChart';
import ModelFrequencyChart from './SummaryCharts/ModelTypeChart';
import SimInfo from './SummaryCharts/SimInfoChart';
import SlotsChart from './SummaryCharts/SlotsChart';
import CorrMapChart from './SummaryCharts/CorrMapChart';
import MonthlySalesChart from './SummaryCharts/MonthlySalesChart';

import DefectsChart from './ReturnsChart/DefectsChart';
import FeedbackChart from './ReturnsChart/FeedbackChart';
import ResPartyChart from './ReturnsChart/ResPartyChart';
import VerificationChart from './ReturnsChart/VerificationChart';


const Dashboard = () => {
  const [files, setFiles] = useState([]); // State to hold file list
  const [selectedFile, setSelectedFile] = useState(''); // Default to empty string for "Choose a file"
  const [sheets, setSheets] = useState([]); // State to hold sheet names
  const [selectedSheet, setSelectedSheet] = useState(''); // Default to empty string for "Choose a sheet"
  const [columns, setColumns] = useState([]); // State to store column names
  const [activeTab, setActiveTab] = useState('predictions');
  
  const [aiComparisonSummary, setAiComparisonSummary] = useState("");
  const [isRefreshingSummary, setIsRefreshingSummary] = useState(false);
  const lastRequestRef = useRef(""); // Store last request signature

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

  const fetchAiComparisonSummary = React.useCallback(async (isManualRefresh = false) => {
    if (!selectedFile || !selectedSheet) return;
    
    if (isManualRefresh) {
      setIsRefreshingSummary(true);
    }
    setAiComparisonSummary("");

    try {
      const response = await fetch(`http://localhost:5001/returns_comparison_summary?file=${selectedFile}&sheet=${selectedSheet}`);
      const data = await response.json();
      if (data.summary) {
        setAiComparisonSummary(data.summary);
      } else {
        alert('No summary found');
      }
    } catch (error) {
      alert('Error fetching summary:', error);
    } finally {
      setIsRefreshingSummary(false);
    }
  }, [selectedFile, selectedSheet]); // Add dependencies here

  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const requestSignature = `${selectedFile}__${selectedSheet}`;
      
      if (requestSignature === lastRequestRef.current) return;
      lastRequestRef.current = requestSignature;

      fetchAiComparisonSummary(false);
    }
  }, [selectedFile, selectedSheet, fetchAiComparisonSummary]); // Add fetchAiComparisonSummary to dependencies

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <h1>Welcome to the Churn Predictor Tool!</h1>
        <p>Please select a uploaded file & sheet to view its data</p>
      </header>

      <div className="dropdown-container">
        <FileSelector files={files} selectedFile={selectedFile} onFileChange={handleFileSelectChange} />
        {selectedFile && <SheetSelector sheets={sheets} selectedSheet={selectedSheet} onSheetChange={handleSheetSelectChange} />}
      </div>

      {selectedFile && selectedSheet && columns.length > 0 && (
        <div className="tabbed-interface">
          <div className="tabs-container">
            <button
              className={`tab-button ${activeTab === 'predictions' ? 'active' : ''}`}
              onClick={() => setActiveTab('predictions')}
            >
              Predictions
            </button>
            <button
              className={`tab-button ${activeTab === 'summary' ? 'active' : ''}`}
              onClick={() => setActiveTab('summary')}
            >
              Summary
            </button>
            {columns.includes("Type") && (
              <button
                className={`tab-button ${activeTab === 'returns' ? 'active' : ''}`}
                onClick={() => setActiveTab('returns')}
              >
                Returns
              </button>
            )}
            <button
              className={`tab-button ${activeTab === 'columnPlotter' ? 'active' : ''}`}
              onClick={() => setActiveTab('columnPlotter')}
            >
              Column Plotter
            </button>
          </div>

          <div className="tab-content">
            {activeTab === 'predictions' && (
              <div className="info-container">
                <div className='predictions-container'>
                  <FeatureImportanceChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet}/>
                </div>
              </div>
            )}

            {activeTab === 'summary' && (
              <div className="info-container">
                {columns.includes("App Usage") && (
                  <AppUsageChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />
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

                {columns.includes("Sale Channel") && (
                  <MonthlySalesChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />
                )}

                {(columns.includes("Churn") || columns.includes("Type")) && (
                  <CorrMapChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />
                )}
              </div>
            )}

            {activeTab === 'returns' && columns.includes("Type") && (
              <div>
                <div className="info-container">
                  {columns.includes("Type") && columns.includes("Defect / Damage type") && (
                    <DefectsChart selectedFile={selectedFile} selectedSheet={selectedSheet} />
                  )}

                  {columns.includes("Type") && columns.includes("Feedback") && (
                    <FeedbackChart selectedFile={selectedFile} selectedSheet={selectedSheet} />
                  )}

                  {columns.includes("Type") && columns.includes("Verification") && (
                    <VerificationChart selectedFile={selectedFile} selectedSheet={selectedSheet} />
                  )}

                  {columns.includes("Type") && columns.includes("Responsible Party") && (
                    <ResPartyChart selectedFile={selectedFile} selectedSheet={selectedSheet} />
                  )}
                </div>

                <div className='aiSummary-container'>
                  <div className="aiSummary-header">
                    <h2>AI Comparison Summary about the Returns</h2>
                    <button 
                      className="refresh-button" 
                      onClick={() => fetchAiComparisonSummary(true)}
                      disabled={isRefreshingSummary}
                    >
                      <span 
                        className={`aiSummary-icon iconify`} 
                        data-icon="material-symbols:refresh-rounded" 
                        data-inline="false"
                      ></span>
                    </button>
                  </div>
                  <div>
                    {aiComparisonSummary ? (
                      <p>{aiComparisonSummary}</p>
                    ) : (
                      <p>{isRefreshingSummary ? 'Loading summary...' : 'No summary available'}</p>
                    )}
                  </div>
                </div>
              
              <button className = "openWindowButton" onClick={() => openWindow(`/returnsinfo?file=${selectedFile}&sheet=${selectedSheet}`)}>View More Feedback</button>


              </div>
            )}

            {activeTab === 'columnPlotter' && (
              <div className="info-container">
                <ColumnsGraphChart selectedFile={selectedFile} selectedSheet={selectedSheet}/>
              </div>
            )}

          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;