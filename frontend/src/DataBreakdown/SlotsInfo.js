import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';   
import "./Analysis.css";

const SlotsInfo = () => {
    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const selectedFile = queryParams.get('file');
    const selectedSheet = queryParams.get('sheet');

    const [carrierData, setCarrierData] = useState([]);
    const [slot1, setSlot1] = useState('Slot 1');
    const [slot2, setSlot2] = useState('Slot 2');
    const [aiSummary, setAiSummary] = useState(""); 
    const [insertedVsUninserted, setInsertedVsUninserted] = useState({ inserted: 0, uninserted: 0 });

    // Fetch data for Slot 1
    useEffect(() => {
    if (selectedFile && selectedSheet) {
        const fetchSlot1CarrierName = async () => {
            try {
                const response = await fetch(`http://localhost:5001/get_carrier_name_from_1slot/${selectedFile}/${selectedSheet}/${slot1}`);
                const data = await response.json();

                if (data.carrier) {
                // Update carrierData with the filtered data
                setSlot1(data.carrier);
                }
            } catch (error) {
                console.log('Error fetching carrier name:', error);
            }
        };

        fetchSlot1CarrierName();
    }
    }, [selectedFile, selectedSheet, slot1]); // Runs when selectedFile, selectedSheet, or currentSlot changes

    // Fetch data for Slot 2
    useEffect(() => {
        if (selectedFile && selectedSheet) {
            const fetchSlot2CarrierName = async () => {
                try {
                    const response = await fetch(`http://localhost:5001/get_carrier_name_from_1slot/${selectedFile}/${selectedSheet}/${slot2}`);
                    const data = await response.json();

                    if (data.carrier) {
                    // Update carrierData with the filtered data
                    setSlot2(data.carrier);
                    }
                } catch (error) {
                    console.log('Error fetching carrier name:', error);
                }
            };

            fetchSlot2CarrierName();
        }
    }, [selectedFile, selectedSheet, slot2]); // Runs when selectedFile, selectedSheet, or currentSlot changes

    // Fetch data for combined slot
    useEffect(() => {
        if (selectedFile && selectedSheet) {
            const fetchCarrierName = async () => {
                try {
                    const response = await fetch(`http://localhost:5001/get_carrier_name_from_slot/${selectedFile}/${selectedSheet}`);
                    const data = await response.json();

                    if (data.carrier) {
                    // Update carrierData with the filtered data
                    setCarrierData(data.carrier);
                    }
                } catch (error) {
                    console.log('Error fetching carrier name:', error);
                }
            };

            fetchCarrierName();
        }
    }, [selectedFile, selectedSheet, carrierData]); // Runs when selectedFile, selectedSheet, or currentSlot changes

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

    // Fetch AI summary
    useEffect(() => {
    if (selectedFile && selectedSheet) {
        const aisummary = async () => {
        try {
            const response = await fetch(`http://localhost:5001/ai_summary2/${selectedFile}/${selectedSheet}/Slot 1/Slot 2`);
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

    return (
    <div>
        <div className="content">
        <h1>Slot Info for {selectedFile} - {selectedSheet}</h1>
        </div>
        
        {/* All Graphs Container */}
        <div className="all-slot-graphs-container">
            <h2>Phone Carriers</h2>
            {/* Slot 1 and Slot 2 Side by Side */}
            <div className="slot-graphs">

                <div className="slot-graph">
                    <h3>Phone Carriers (Slot 1)</h3>
                    {slot1 && Object.keys(slot1).length ? (
                    <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={Object.entries(slot1)
                        .map(([carrier, count]) => ({ carrier, count }))
                        .sort((a, b) => b.count - a.count)}>
                        <XAxis dataKey="carrier" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="count" fill="#C4D600" />
                        </BarChart>
                    </ResponsiveContainer>
                    ) : (
                    <p>Loading phone carriers...</p>
                    )}
                </div>

                <div className="slot-graph">
                    <h3>Phone Carriers (Slot 2)</h3>
                    {slot2 && Object.keys(slot2).length ? (
                    <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={Object.entries(slot2)
                        .map(([carrier, count]) => ({ carrier, count }))
                        .sort((a, b) => b.count - a.count)}>
                        <XAxis dataKey="carrier" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="count" fill="#C4D600" />
                        </BarChart>
                    </ResponsiveContainer>
                    ) : (
                    <p>Loading phone carriers...</p>
                    )}
                </div>

            </div>

            {/* Combined Slot Graph */}
            <div className="combined-graph">
                <h3>Phone Carriers (Combined)</h3>
                {carrierData && Object.keys(carrierData).length ? (
                    <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={Object.entries(carrierData)
                        .map(([carrier, count]) => ({ carrier, count }))
                        .sort((a, b) => b.count - a.count)}>
                        <XAxis dataKey="carrier" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="count" fill="#C4D600" />
                        </BarChart>
                    </ResponsiveContainer>
                ) : (
                    <p>Loading top phone carriers...</p>
                )}
            </div>
            
            <div>
                <h3>Summary about Phone Carriers</h3>
            </div>

        </div>

        <div className="graph-container">
            {/* Pie Chart for Inserted vs Uninserted*/}
            <div className="chart">
                <h2>Inserted vs Uninserted</h2>
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


    );
};

export default SlotsInfo;
