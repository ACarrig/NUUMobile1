import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, PieChart, Pie, Cell, Label } from 'recharts';
import "./Analysis.css";
import AiSummary from './Summary';

const ModelType = () => {
  const [modelType, setModelType] = useState([]);
  const [modelChannelPerformance, setModelChannelPerformance] = useState({});

  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const selectedFile = queryParams.get('file');
  const selectedSheet = queryParams.get('sheet');

  // Fetch model type data
  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const fetchModelType = async () => {
        try {
          const response = await fetch(`http://localhost:5001/get_model_type/${selectedFile}/${selectedSheet}`);
          const data = await response.json();
          if (data.model) {
            setModelType(data.model);
          } else {
            console.log('No model data found');
          }
        } catch (error) {
          console.log('Error fetching model data:', error);
        }
      };
      fetchModelType();
    }
  }, [selectedFile, selectedSheet]);

  // Fetch model performance by channel data
  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const fetchModelPerformanceByChannel = async () => {
        try {
          const response = await fetch(`http://localhost:5001/get_model_performance_by_channel/${selectedFile}/${selectedSheet}`);
          const data = await response.json();
          if (data.model_channel_performance) {
            setModelChannelPerformance(data.model_channel_performance);
          } else {
            console.log('No performance data found');
          }
        } catch (error) {
          console.log('Error fetching performance data:', error);
        }
      };
      fetchModelPerformanceByChannel();
    }
  }, [selectedFile, selectedSheet]);

  // Prepare sales channels
  const salesChannelTypes = Object.keys(modelChannelPerformance).reduce((channels, model) => {
    const modelChannels = modelChannelPerformance[model];
    Object.keys(modelChannels).forEach(channel => {
      if (!channels.includes(channel)) channels.push(channel);
    });
    return channels;
  }, []);

  // Prepare data for each sales channel's donut chart
  const getDonutChartData = (channel) => {
    const modelData = Object.entries(modelChannelPerformance).map(([model, channels]) => {
      return {
        model,
        count: channels[channel] || 0,
      };
    });

    const others = modelData.filter(item => item.count < 20);
    const othersCount = others.reduce((sum, item) => sum + item.count, 0);

    const filteredData = modelData.filter(item => item.count >= 20);
    if (othersCount > 0) {
      filteredData.push({ model: "Others", count: othersCount });
    }

    // Ensure that "Others" is always at the last
    return filteredData.sort((a, b) => a.model === "Others" ? 1 : (b.model === "Others" ? -1 : b.count - a.count));
  };

  // Prepare colors for each segment of the donut chart
  const COLORS = [
    '#C4D600', '#FF8042', '#0088FE', '#FFBB28', '#00C49F', '#FF00FF', 
    '#FFD700', '#4B0082', '#FF6347', '#8A2BE2', '#A52A2A', '#DC143C', 
    '#00008B', '#A9A9A9', '#7FFF00'
  ];  

  return (
    <div>
      <div className="content">
        <h1>Model Type for {selectedFile} - {selectedSheet}</h1>

        {/* Bar Chart for Model Types */}
        <div className="graph-container">
          <div className="chart">
            <h2>Types of Models</h2>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={Object.entries(modelType)
                .map(([model, count]) => ({ model, count }))
                .sort((a, b) => b.count - a.count)} >
                <XAxis dataKey="model" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#C4D600" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        {/* Donut Charts for Model Performance by Sales Channel */}
          <div className="donut-graph-container">
            <h2>Model Performance by Sales Channel</h2>
            {Object.keys(modelChannelPerformance).length > 0 && (
            <div className="donut-charts-container">
              {salesChannelTypes.map((channel, index) => (
                <div key={index} className="donut-chart">
                  <ResponsiveContainer width="100%" height={400}>
                    <PieChart>
                      <Pie
                        data={getDonutChartData(channel)}
                        dataKey="count"
                        nameKey="model"
                        innerRadius={85}  // Donut inner radius
                        outerRadius={120}  // Outer radius of the donut
                        fill="#C4D600"
                        label={false}  // Remove the label with count
                      >
                        {/* Display the channel name at the center of the donut chart */}
                        <Label value={channel} position="center" fontSize={18} fontWeight="bold" fill="#000" />
                        {getDonutChartData(channel).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              ))}
            </div>
            )}
          </div>

        <AiSummary 
          selectedFile={selectedFile} 
          selectedSheet={selectedSheet} 
          selectedColumn={["Model"]}
        />

      </div>
    </div>
  );
};

export default ModelType;
