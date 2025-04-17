import React, { useEffect, useState, useRef } from 'react';
import './Summary.css';

const Summary = ({ selectedFile, selectedSheet, selectedColumn }) => {
  const [aiSummary, setAiSummary] = useState("");
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [showAnimation, setShowAnimation] = useState(false);
  const lastRequestRef = useRef("");

  const fetchSummary = async (isManualRefresh = false) => {
    if (!selectedFile || !selectedSheet) return;
    
    if (isManualRefresh) {
      setShowAnimation(true);
    }
    setIsRefreshing(true);
    setAiSummary("");
    
    try {
      let url = "";

      if (Array.isArray(selectedColumn) && selectedColumn.length === 2) {
        const [column1, column2] = selectedColumn;
        url = `http://localhost:5001/ai_summary2?file=${selectedFile}&sheet=${selectedSheet}&column1=${column1}&column2=${column2}`;
      } else {
        url = `http://localhost:5001/ai_summary?file=${selectedFile}&sheet=${selectedSheet}&column=${selectedColumn}`;
      }

      const response = await fetch(url);
      const data = await response.json();

      if (data && data.summary) {
        setAiSummary(data.summary);
      } else {
        alert('No AI summary received');
      }

    } catch (error) {
      alert(`Error fetching summary: ${error}`);
    } finally {
      setIsRefreshing(false);
      setShowAnimation(false);
    }
  };

  useEffect(() => {
    const columnSignature = Array.isArray(selectedColumn) 
      ? selectedColumn.join('|') 
      : selectedColumn;

    const requestSignature = `${selectedFile}__${selectedSheet}__${columnSignature}`;

    if (requestSignature === lastRequestRef.current) return;
    lastRequestRef.current = requestSignature;

    fetchSummary(false); // Pass false to indicate this is not a manual refresh
  }, [selectedFile, selectedSheet, selectedColumn]);

  return (
    <div className="summary-container">
      <div className="summary-header">
        <h2>AI Summary</h2>
        <button 
          className="refresh-button" 
          onClick={() => fetchSummary(true)}
          disabled={isRefreshing}>
          <span 
            className={`summary-icon iconify ${showAnimation ? 'refreshing' : ''}`} 
            data-icon="material-symbols:refresh-rounded" 
            data-inline="false"></span>
        </button>
      </div>

      <div>
        {aiSummary ? (
          <p>{aiSummary}</p>
        ) : (
          <p>{isRefreshing ? 'Loading summary...' : 'No summary available'}</p>
        )}
      </div>
    </div>
  );
};

export default Summary;