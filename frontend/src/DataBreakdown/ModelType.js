import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as BarTooltip, ResponsiveContainer, PieChart, Pie, Cell, Tooltip as PieTooltip } from 'recharts';
import "./Analysis.css";

const ModelType = () => {
  const [modelData, setModelData] = useState([]);
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const selectedFile = queryParams.get('file');
  const selectedSheet = queryParams.get('sheet');
  const [aiSummary, setAiSummary] = useState(""); // State for AI summary of data
  
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
  
  // Fetch the AI response
  useEffect(() => {
    if (selectedFile && selectedSheet) {
        const aisummary = async () => {
            try {
            const response = await fetch(`http://localhost:5001/ai_summary/${selectedFile}/${selectedSheet}/Model`);
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

  // Convert the model data into a format suitable for Recharts (array of objects)
  const chartData = modelData 
    ? Object.entries(modelData).map(([model, count]) => ({
        model,    // Use 'model' as the key (model name)
        count,    // The count associated with the model
      }))
    .sort((a, b) => b.count - a.count)  // Sort by count in descending order
    : [];

  // Prepare data for Pie chart, combining models with count less than 30 as "Others"
  const pieData = chartData.reduce((acc, { model, count }) => {
    if (count < 30) {
      // Add to 'Others' if count is less than 30
      const others = acc.find(item => item.model === "Others");
      if (others) {
        others.count += count;
      } else {
        acc.push({ model: "Others", count });
      }
    } else {
      acc.push({ model, count });
    }
    return acc;
  }, []);

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#C4D600']; // Color palette for pie chart

  return (
    <div>
        <div className="content">
            <h1>Model Type for {selectedFile} - {selectedSheet}</h1>

            {/* Bar Chart */}
            <div className="graph">
                {chartData.length > 0 && (
                    <div>
                    <h2>Model Frequency Chart</h2>
                    <ResponsiveContainer width="100%" height={500}> {/* Increased height */}
                        <BarChart data={chartData} margin={{ left: 30, right: 30, top: 20, bottom: 70 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                          dataKey="model" 
                          tick={{ angle: -45, textAnchor: 'end' }}  // Rotate X-Axis labels to prevent overlap
                          interval={0}  // Optional: Adjust if necessary
                        />
                        <YAxis />
                        <BarTooltip />
                        <Bar dataKey="count" fill="#C4D600" barSize={40} /> {/* Increased bar size */}
                        </BarChart>
                    </ResponsiveContainer>
                    </div>
                )}
            </div>

            {/* Pie Chart */}
            {pieData.length > 0 && (
                <div>
                    <h2>Model Type Distribution</h2>
                    <ResponsiveContainer width="100%" height={600}> {/* Increased height */}
                        <PieChart>
                            <Pie
                                data={pieData}
                                dataKey="count"
                                nameKey="model"
                                cx="50%"
                                cy="50%"
                                outerRadius={230}  // Increase the outerRadius to make the pie chart larger
                                fill="#8884d8"
                                labelLine={true}  // Disable the label line
                                label={({ name, cx, cy, midAngle, innerRadius, outerRadius, value, index }) => {
                                  const RADIAN = Math.PI / 180;
                                  const radius = outerRadius + 60;
                                  const x = cx + radius * Math.cos(-midAngle * RADIAN);
                                  const y = cy + radius * Math.sin(-midAngle * RADIAN);
                                  return (
                                    <text
                                      x={x}
                                      y={y}
                                      textAnchor="middle"
                                      fill={COLORS[index % COLORS.length]}
                                      fontSize="16"
                                    >
                                      {name}
                                    </text>
                                  );
                                }}
                            >
                                {pieData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                            <PieTooltip />  {/* Only show count on hover */}
                        </PieChart>
                    </ResponsiveContainer>
                </div>
            )}

            <div>
                <h2>Summary</h2>
                <div>
                    {aiSummary ? (
                    <p>{aiSummary}</p>  // Display the summary if it's available
                    ) : (
                    <p>Loading summary...</p>  // Show loading message if summary is still being fetched
                    )}
                </div>
            </div>

        </div>

    </div>
  );
};

export default ModelType;
