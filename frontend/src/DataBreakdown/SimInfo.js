import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';   
import "./Analysis.css";
import AiSummary from "./Summary"

const SimInfo = () => {
  const [carrierData, setCarrierData] = useState([]);
  const [insertedVsUninserted, setInsertedVsUninserted] = useState({ inserted: 0, uninserted: 0 });

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
            console.log('No carrier name found');
          }
        } catch (error) {
          console.log('Error fetching carrier name:', error);
        }
      };

      fetchCarrierName();
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
        }, [] )
        .sort((a, b) => b.count - a.count)
    : [];

  // Data for Inserted vs Uninserted/Emergency Calls
  useEffect(() => {
    if (carrierData) {
      const inserted = Object.entries(carrierData).reduce((acc, [carrier, count]) => {
        // Exclude 'PERMISSION_DENIED' from both inserted and uninserted
        if (carrier.toLowerCase().includes("permission_denied")) {
          return acc;  // Skip this entry
        }
        
        if (carrier.toLowerCase().includes("uninserted") || carrier.toLowerCase().includes("emergency calls only")) {
          acc.uninserted += count;
        } else {
          acc.inserted += count;
        }
        return acc;
      }, { inserted: 0, uninserted: 0 });
      
      setInsertedVsUninserted(inserted);
    }
  }, [carrierData]);  

  const insertedVsUninsertedData = [
    { name: "Inserted", value: insertedVsUninserted.inserted },
    { name: "Uninserted", value: insertedVsUninserted.uninserted }
  ];

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
          {/* Bar Chart for Carriers */}
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
        
        <div className="graph-container">
            {/* Pie Chart for Inserted vs Uninserted*/}
            <div className="chart">
                <h3>Inserted vs Uninserted</h3>
                <ResponsiveContainer width="100%" height={350}>
                <PieChart>
                    <Pie 
                    data={insertedVsUninsertedData} 
                    dataKey="value" 
                    nameKey="name" 
                    fill="#FF6347"
                    >
                    {insertedVsUninsertedData.map((entry, index) => (
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
        </div>

        {/* AI Summary */}
        <AiSummary 
          selectedFile={selectedFile} 
          selectedSheet={selectedSheet} 
          selectedColumn={["sim_info"]}
        />

      </div>
    </div>
  );
};

export default SimInfo;
