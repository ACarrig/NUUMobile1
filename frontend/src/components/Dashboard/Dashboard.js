import React, { useState, useEffect, useCallback } from 'react';
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
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [sheets, setSheets] = useState([]);
  const [selectedSheet, setSelectedSheet] = useState('');
  const [columns, setColumns] = useState([]);
  const [activeTab, setActiveTab] = useState('predictions');

  const [aiComparisonSummary, setAiComparisonSummary] = useState('');
  const [isRefreshingSummary, setIsRefreshingSummary] = useState(false);

  useEffect(() => {
    document.title = 'Dashboard - Churn Predictor';
  }, []);

  const openWindow = (url) => window.open(url, '_blank');

  const fetchData = useCallback(async (endpoint, setter, key) => {
    try {
      const response = await fetch(`http://localhost:5001/${endpoint}`);
      const data = await response.json();
      if (data[key]) setter(data[key]);
    } catch (error) {
      console.error(`Error fetching ${key}:`, error);
    }
  }, []);

  useEffect(() => {
    fetchData('get_files', setFiles, 'files');
  }, [fetchData]);

  useEffect(() => {
    if (selectedFile) fetchData(`get_sheets/${selectedFile}`, setSheets, 'sheets');
  }, [selectedFile, fetchData]);

  useEffect(() => {
    if (selectedFile && selectedSheet)
      fetchData(`get_all_columns/${selectedFile}/${selectedSheet}`, setColumns, 'columns');
  }, [selectedFile, selectedSheet, fetchData]);

  const handleFileSelectChange = (e) => {
    setSelectedFile(e.target.value);
    setSelectedSheet('');
    setColumns([]);
  };

  const handleSheetSelectChange = (e) => {
    setSelectedSheet(e.target.value);
    setColumns([]);
  };

  const fetchAiComparisonSummary = useCallback(async () => {
    if (!selectedFile || !selectedSheet) return;
  
    setIsRefreshingSummary(true);
    setAiComparisonSummary('');
  
    try {
      const response = await fetch(`http://localhost:5001/returns_comparison_summary?file=${selectedFile}&sheet=${selectedSheet}`);
      const data = await response.json();
      setAiComparisonSummary(data.summary || 'No summary available');
    } catch (error) {
      console.error('Error fetching AI summary:', error);
      console.log('Error fetching summary');
    } finally {
      setIsRefreshingSummary(false);
    }
  }, [selectedFile, selectedSheet]);  

  useEffect(() => {
    if (activeTab === 'returns') fetchAiComparisonSummary();
  }, [activeTab, fetchAiComparisonSummary]);  

  const renderTabContent = () => {
    switch (activeTab) {
      case 'predictions':
        return (
          <div className="info-container">
            <FeatureImportanceChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />
          </div>
        );
      case 'summary':
        return (
          <div className="info-container">
            {columns.includes('App Usage') && <AppUsageChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />}
            {columns.includes('Age Range') && <AgeRangeChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />}
            {columns.includes('Model') && <ModelFrequencyChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />}
            {columns.includes('sim_info') && <SimInfo openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />}
            {columns.includes('Slot 1') && columns.includes('Slot 2') && <SlotsChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />}
            {columns.includes('Sale Channel') && <MonthlySalesChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />}
            {(columns.includes('Churn') || columns.includes('Type')) && <CorrMapChart openWindow={openWindow} selectedFile={selectedFile} selectedSheet={selectedSheet} />}
          </div>
        );
      case 'returns':
        return (
          <div>
            <div className="info-container">
              {columns.includes('Type') && columns.includes('Defect / Damage type') && <DefectsChart selectedFile={selectedFile} selectedSheet={selectedSheet} />}
              {columns.includes('Type') && columns.includes('Feedback') && <FeedbackChart selectedFile={selectedFile} selectedSheet={selectedSheet} />}
              {columns.includes('Type') && columns.includes('Verification') && <VerificationChart selectedFile={selectedFile} selectedSheet={selectedSheet} />}
              {columns.includes('Type') && columns.includes('Responsible Party') && <ResPartyChart selectedFile={selectedFile} selectedSheet={selectedSheet} />}
            </div>

            <div className="aiSummary-container">
              <div className="aiSummary-header">
                <h2>AI Comparison Summary about the Returns</h2>
                <button className="refresh-button" onClick={fetchAiComparisonSummary} disabled={isRefreshingSummary}>
                  <span className="aiSummary-icon iconify" data-icon="material-symbols:refresh-rounded" />
                </button>
              </div>
              <div>
                <p>{aiComparisonSummary || (isRefreshingSummary ? 'Loading summary...' : 'No summary available')}</p>
              </div>
              <button className="openWindowButton" onClick={() => openWindow(`/returnsinfo?file=${selectedFile}&sheet=${selectedSheet}`)}>View More Feedback</button>
            </div>
          </div>
        );
      case 'columnPlotter':
        return (
          <div className="info-container">
            <ColumnsGraphChart selectedFile={selectedFile} selectedSheet={selectedSheet} />
          </div>
        );
      default:
        return null;
    }
  };

  const tabs = [
    { id: 'predictions', label: 'Predictions' },
    { id: 'summary', label: 'Summary' },
    { id: 'returns', label: 'Returns', condition: columns.includes('Type') },
    { id: 'columnPlotter', label: 'Column Plotter' },
  ];

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <h1>Welcome to the Churn Predictor Tool!</h1>
        <p>Please select an uploaded file & sheet to view its data</p>
      </header>

      <div className="dropdown-container">
        <FileSelector files={files} selectedFile={selectedFile} onFileChange={handleFileSelectChange} />
        {selectedFile && (
          <SheetSelector sheets={sheets} selectedSheet={selectedSheet} onSheetChange={handleSheetSelectChange} />
        )}
      </div>

      {selectedFile && selectedSheet && columns.length > 0 && (
        <div className="tabbed-interface">
          <div className="tabs-container">
            {tabs
              .filter(tab => tab.condition === undefined || tab.condition)
              .map(tab => (
                <button
                  key={tab.id}
                  className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
                  onClick={() => setActiveTab(tab.id)}
                >
                  {tab.label}
                </button>
              ))}
          </div>
          <div className="tab-content">{renderTabContent()}</div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
