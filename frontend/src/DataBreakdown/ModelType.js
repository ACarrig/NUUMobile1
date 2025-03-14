import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as BarTooltip, ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } from 'recharts'; 
import { PieChart as PieChartIcon, BarChart as BarChartIcon } from 'lucide-react'; 
import "./Analysis.css";

const ModelType = () => {
  const [modelData, setModelData] = useState([]);
  const [aiSummary, setAiSummary] = useState(""); 

  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const selectedFile = queryParams.get('file');
  const selectedSheet = queryParams.get('sheet');

  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const fetchModelType = async () => {
        try {
          const response = await fetch(`http://localhost:5001/get_model_type/${selectedFile}/${selectedSheet}`);
          const data = await response.json();
          if (data.model) {
            setModelData(data.model);
          } else {
            alert('No model data found');
          }
        } catch (error) {
          alert('Error fetching model data:', error);
        }
      };

      fetchModelType();
    }
  }, [selectedFile, selectedSheet]);
  
  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const aisummary = async () => {
        try {
          const response = await fetch(`http://localhost:5001/ai_summary/${selectedFile}/${selectedSheet}/Model`);
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

  const chartData = modelData
    ? Object.entries(modelData).map(([model, count]) => ({
        model, 
        count, 
      }))
    .sort((a, b) => b.count - a.count)
    : [];

  const filteredData = chartData.reduce((acc, { model, count }) => {
    if (count < 30) {
      const existing = acc.find(item => item.name === "Others");
      if (existing) {
        existing.value += count;
      } else {
        acc.push({ name: "Others", value: count });
      }
    } else {
      acc.push({ name: model, value: count });
    }
    return acc;
  }, []);

  const colors = [
    "#C4D600", "#00C4D6", "#FF6347", "#8A2BE2", "#FF8C00", "#20B2AA", 
    "#FFD700", "#FF4500", "#FF1493", "#32CD32", "#BA55D3", "#00BFFF", 
    "#DC143C", "#FF00FF", "#FFD700", "#4B0082", "#FF6347", "#800080"
  ];

  return (
    <div>
      <div className="content">
        <h1>Model Type for {selectedFile} - {selectedSheet}</h1>

        <div className="graph-container">
          <div className="chart">
            <h2>Pie Chart</h2>
            <ResponsiveContainer width="100%" height={500}>
              <PieChart>
                <Pie 
                  data={filteredData} 
                  dataKey="value" 
                  nameKey="name" 
                  fill="#C4D600"
                >
                  {filteredData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                  ))}
                </Pie>

                <Tooltip 
                  formatter={(value, name) => [`${name}: ${value}`]} 
                  contentStyle={{ backgroundColor: '#fff', borderRadius: '5px', border: '1px solid #ccc' }} 
                />

                {/* Add the Legend below the Pie Chart */}
                <Legend 
                  layout="horizontal" 
                  verticalAlign="bottom" 
                  align="center" 
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="chart">
            <h2>Bar Chart</h2>
            <ResponsiveContainer width="100%" height={500}>
              <BarChart data={chartData} margin={{ left: 30, right: 30, top: 20, bottom: 70 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="model" 
                  tick={{ angle: -45, textAnchor: 'end' }} 
                  interval={0} 
                />
                <YAxis />
                <BarTooltip />
                <Bar dataKey="count" fill="#C4D600" barSize={40} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

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
