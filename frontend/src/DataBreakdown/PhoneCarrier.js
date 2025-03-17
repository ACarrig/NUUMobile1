import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip as BarTooltip, ResponsiveContainer, 
  PieChart, Pie, Cell, Tooltip, Legend 
} from 'recharts'; 
import "./Analysis.css";

const ModelType = () => {
  const [carrierData, setCarrierData] = useState([]);
  const [aiSummary, setAiSummary] = useState(""); 

  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const selectedFile = queryParams.get('file');
  const selectedSheet = queryParams.get('sheet');

  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const fetchCarrierName = async () => {
        try {
          const response = await fetch(`http://localhost:5001/get_carrier_name/${selectedFile}/${selectedSheet}`);
          const data = await response.json();
          if (data.carrier) {
            setCarrierData(data.carrier);
          } else {
            alert('No carrier name found');
          }
        } catch (error) {
          alert('Error fetching carrier name:', error);
        }
      };

      fetchCarrierName();
    }
  }, [selectedFile, selectedSheet]);

  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const aisummary = async () => {
        try {
          const response = await fetch(`http://localhost:5001/ai_summary/${selectedFile}/${selectedSheet}/sim_info`);
          const data = await response.json();
          if (data && data.aiSummary) {
            setAiSummary(data.aiSummary);
          } else {
            alert('No AI summary received');
          }
        } catch (error) {
          alert(`Error fetching summary: ${error}`);
        }
      };

      aisummary();
    }
  }, [selectedFile, selectedSheet]);

  // Bar Chart Data (count < 20)
  const barChartData = carrierData
    ? Object.entries(carrierData)
        .reduce((acc, [carrier, count]) => {
          if (count < 20) {
            const existing = acc.find(item => item.carrier === "Others");
            if (existing) {
              existing.count += count;
            } else {
              acc.push({ carrier: "Others", count });
            }
          } else {
            acc.push({ carrier, count });
          }
          return acc;
        }, [])
        .sort((a, b) => b.count - a.count)
    : [];

  // Pie Chart Data (count < 30)
  const pieChartData = carrierData
    ? Object.entries(carrierData)
        .reduce((acc, [carrier, count]) => {
          if (count < 30) {
            const existing = acc.find(item => item.name === "Others");
            if (existing) {
              existing.value += count;
            } else {
              acc.push({ name: "Others", value: count });
            }
          } else {
            acc.push({ name: carrier, value: count });
          }
          return acc;
        }, [])
        .sort((a, b) => b.value - a.value)
    : [];

  const colors = [
    "#C4D600", "#00C4D6", "#FF6347", "#8A2BE2", "#FF8C00", "#20B2AA", 
    "#FFD700", "#FF4500", "#FF1493", "#32CD32", "#BA55D3", "#00BFFF", 
    "#DC143C", "#FF00FF", "#FFD700", "#4B0082", "#FF6347", "#800080"
  ];

  return (
    <div>
      <div className="content">
        <h1>Sim Info for {selectedFile} - {selectedSheet}</h1>

        <h2>Phone Carriers</h2>
        <div className="graph-container">
          {/* Pie Chart */}
          <div className="chart">
            <h3>Pie Chart</h3>
            <ResponsiveContainer width="100%" height={350}>
              <PieChart>
                <Pie 
                  data={pieChartData} 
                  dataKey="value" 
                  nameKey="name" 
                  fill="#C4D600"
                >
                  {pieChartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                  ))}
                </Pie>

                <Tooltip 
                  formatter={(value, name) => [`${name}: ${value}`]} 
                  contentStyle={{ backgroundColor: '#fff', borderRadius: '5px', border: '1px solid #ccc' }} 
                />

                <Legend 
                  layout="horizontal" 
                  verticalAlign="bottom" 
                  align="center" 
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Bar Chart */}
          <div className="chart">
            <h3>Bar Chart</h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={barChartData}>
                <XAxis dataKey="carrier" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#C4D600" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* AI Summary */}
        <div className="summary-container">
          <h2>Summary</h2>
          <div>
            {aiSummary ? (
              <p>{aiSummary}</p>
            ) : (
              <p>Loading summary...</p>
            )}
          </div>
        </div>

      </div>
    </div>
  );
};

export default ModelType;
