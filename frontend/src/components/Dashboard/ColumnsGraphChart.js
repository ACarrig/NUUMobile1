import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer} from 'recharts';
import './ColumnsGraphChart.css';
import AiSummary from '../../DataBreakdown/Summary'

const ColumnsGraphChart = ({ selectedFile, selectedSheet }) => {
  const [columns, setColumns] = useState([]);
  const [selectedColumn, setSelectedColumn] = useState('');
  const [chartData, setChartData] = useState([]);

  // Fetch column names
  useEffect(() => {
    const fetchColumns = async () => {
      if (!selectedFile || !selectedSheet) return;
      try {
        const response = await fetch(`http://localhost:5001/get_all_columns/${selectedFile}/${selectedSheet}`);
        const data = await response.json();
        if (data.columns) setColumns(data.columns);
      } catch (error) {
        console.error('Error fetching columns:', error);
      }
    };
    fetchColumns();
  }, [selectedFile, selectedSheet]);

  // Fetch value counts for selected column
  useEffect(() => {
    if (selectedFile && selectedSheet && selectedColumn) {
      const fetchColumnData = async () => {
          try {
            const response = await fetch(`http://localhost:5001/get_column_data?file=${selectedFile}&sheet=${selectedSheet}&column=${selectedColumn}`);
            const data = await response.json();
            if (data.frequency) {
                setChartData(data.frequency); // Store frequency data in state
            }
          } catch (error) {
            alert(`Error fetching column data: ${error}`);
          }
      };

      fetchColumnData();
      }
  }, [selectedFile, selectedSheet, selectedColumn]);

  // Function to get the highest
  const getHighest = (chartData) => {
    const maxCount = Math.max(...Object.values(chartData));
    const maxColumn = Object.keys(chartData).find(data => chartData[data] === maxCount);
    return `${maxColumn}`;
  };

  // Function to get the lowest
  const getLowest = (chartData) => {
      const minCount = Math.min(...Object.values(chartData));
      const minColumn = Object.keys(chartData).find(data => chartData[data] === minCount);
      return `${minColumn}`;
  };

  return (
    <div className="columns-corr-chart">
      <p>Select a Column to Visualize its Value Counts</p>
      
      <select
        value={selectedColumn}
        onChange={(e) => setSelectedColumn(e.target.value)}
        className="dropdown"
      >
        <option value="">-- Select a Column --</option>
        {columns.map((col) => (
          <option key={col} value={col}>
            {col}
          </option>
        ))}
      </select>

      {chartData && Object.keys(chartData).length ? (
        <div>
          <div className="summary-graph">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={Object.entries(chartData)
                .map(([data, count]) => ({ data, count }))
                .sort((a, b) => b.count - a.count)
                .slice(0, 5)}>
                <XAxis dataKey="data" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#C4D600" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="column-stats">
            <p><strong>Highest {selectedColumn}: </strong>{getHighest(chartData)}</p>
            <p><strong>Lowest {selectedColumn}: </strong>{getLowest(chartData)}</p>
          </div>

          <AiSummary 
            selectedFile={selectedFile} 
            selectedSheet={selectedSheet} 
            selectedColumn={[selectedColumn]}
          />

        </div>
      ) : (
        <p>Loading top 5 from {selectedColumn}...</p>
      )}
    </div>
  );
};

export default ColumnsGraphChart;
