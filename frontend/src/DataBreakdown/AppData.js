import React, { useEffect, useState } from "react";
import { useLocation } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import "./Analysis.css";
import AiSummary from './Summary';

const AppData = () => {
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const selectedFile = queryParams.get('file');
  const selectedSheet = queryParams.get('sheet');

  const [appUsage, setAppUsage] = useState({}); // Local state for app usage
  const [mostUsedCounts, setMostUsedCounts] = useState({});

  // Fetch app usage when the component mounts
  useEffect(() => {
    const fetchAppUsage = async () => {
      try {
        const response = await fetch(`http://localhost:5001/get_app_usage/${selectedFile}/${selectedSheet}`);
        const data = await response.json();  
        if (data && data.app_frequency) {  
          setAppUsage(data.app_frequency);  
        }
      } catch (error) {
        console.error(`Error fetching app usage: ${error}`);
      }
    };

    fetchAppUsage();
  }, [selectedFile, selectedSheet]); // Runs only once when the component mounts

  useEffect(() => {
    const fetchMostUsedCounts = async () => {
      try {
        const response = await fetch(`http://localhost:5001/get_most_used_app_counts/${selectedFile}/${selectedSheet}`);
        const data = await response.json();
        if (data && data.most_used_counts) {
          setMostUsedCounts(data.most_used_counts);
        }
      } catch (error) {
        console.error(`Error fetching most used counts: ${error}`);
      }
    };
  
    fetchMostUsedCounts();
  }, [selectedFile, selectedSheet]);

  return (
    <div className="content">
      <h1>App Usage Data for {selectedFile} - {selectedSheet}</h1>

      <div className="graph-container">
        <div className="chart">
          <h2>Total App Usage by Hours</h2>
          <p>Sums total time spent on each app across all records</p>
          {appUsage && Object.keys(appUsage).length ? (
            <div className="summary-graph">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={Object.entries(appUsage)
                  .map(([app, count]) => ({ app, count: (count / 3600).toFixed(2) })) 
                  .sort((a, b) => b.count - a.count)}>
                  <XAxis dataKey="app" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#C4D600" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p>Loading app usage...</p>
          )}
        </div>
      </div>

      <div className="graph-container">
        <div className="chart">
          <h2>Most Used App Count per Phone</h2>
          <p>Counts how often an app was the most used app in a record</p>
          {mostUsedCounts && Object.keys(mostUsedCounts).length ? (
            <div className="summary-graph">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={Object.entries(mostUsedCounts)
                  .map(([app, count]) => ({ app, count }))
                  .sort((a, b) => b.count - a.count)}>
                  <XAxis dataKey="app" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#C4D600" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p>Loading most used app counts...</p>
          )}
        </div>
      </div>

      <AiSummary 
        selectedFile={selectedFile} 
        selectedSheet={selectedSheet} 
        selectedColumn={["App Usage"]}
      />

    </div>
  );
};

export default AppData;