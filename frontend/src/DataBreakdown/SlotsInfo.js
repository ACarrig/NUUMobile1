import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';   
import "./SlotsInfo.css";
import AiSummary from './Summary';

const SlotsInfo = () => {
    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const selectedFile = queryParams.get('file');
    const selectedSheet = queryParams.get('sheet');

    const [carrierData, setCarrierData] = useState([]);
    const [slot1, setSlot1] = useState([]);
    const [slot2, setSlot2] = useState([]);
    const [insertedVsUninserted, setInsertedVsUninserted] = useState({ inserted: 0, uninserted: 0 });
    const [carrierCountry, setCarrierCountry] = useState([]);

    // Fetch data for Slot 1
    useEffect(() => {
    if (selectedFile && selectedSheet) {
        const fetchSlot1CarrierName = async () => {
            try {
                const response = await fetch(`http://localhost:5001/get_carrier_name_from_1slot/${selectedFile}/${selectedSheet}/Slot 1`);
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
    }, [selectedFile, selectedSheet]); // Runs when selectedFile, selectedSheet, or currentSlot changes

    // Fetch data for Slot 2
    useEffect(() => {
        if (selectedFile && selectedSheet) {
            const fetchSlot2CarrierName = async () => {
                try {
                    const response = await fetch(`http://localhost:5001/get_carrier_name_from_1slot/${selectedFile}/${selectedSheet}/Slot 2`);
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
    }, [selectedFile, selectedSheet]); // Runs when selectedFile, selectedSheet, or currentSlot changes

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
    }, [selectedFile, selectedSheet]); // Runs when selectedFile, selectedSheet, or currentSlot changes 
    
    // Fetch data for combined slot
    useEffect(() => {
        if (selectedFile && selectedSheet) {
            const fetchCarrierCountry = async () => {
                try {
                    const response = await fetch(`http://localhost:5001/get_carrier_country/${selectedFile}/${selectedSheet}`);
                    const data = await response.json();

                    if (data.country) {
                    // Update carrierData with the filtered data
                    setCarrierCountry(data.country);
                    }
                } catch (error) {
                    console.log('Error fetching carrier name:', error);
                }
            };

            fetchCarrierCountry();
        }
    }, [selectedFile, selectedSheet]); // Runs when selectedFile, selectedSheet, or currentSlot changes

    // Get top 10 ccountry
    const getTop10Country = () => {
        if (!carrierCountry || Object.keys(carrierCountry).length === 0) return [];
        
        // Filter out uninserted carriers
        const filteredCarrierCountry = Object.entries(carrierCountry)
            .filter(([carrier]) => 
                !carrier.toLowerCase().includes("uninserted") && 
                !carrier.toLowerCase().includes("emergency calls only")
            )
            .map(([country, count]) => ({ country, count }))
            .sort((a, b) => b.count - a.count);
        
        // Return the top 10 country
        return filteredCarrierCountry.slice(0, 10);
    };   


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

    // Calculate percentage difference between inserted and uninserted
    const getPercentageDifference = (inserted, uninserted) => {
        const total = inserted + uninserted;
        if (total === 0) return 0;
        return ((Math.abs(inserted - uninserted) / total) * 100).toFixed(2);
    };

    const percentageDifference = getPercentageDifference(insertedVsUninserted.inserted, insertedVsUninserted.uninserted);

    const colors = [
        "#C4D600", "#00C4D6", "#FF6347", "#8A2BE2", "#FF8C00", "#20B2AA", 
        "#FFD700", "#FF4500", "#FF1493", "#32CD32", "#BA55D3", "#00BFFF", 
        "#DC143C", "#FF00FF", "#FFD700", "#4B0082", "#FF6347", "#800080"
    ];

    return (
    <div>
        <div className="slots-content">
        <h1>Slot Info for {selectedFile} - {selectedSheet}</h1>
        </div>
        
        {/* All Graphs Container */}
        <div className="all-slot-graphs-container">
            <h2>Phone Carriers</h2>
            
            {/* Slot 1 and Slot 2 */}
                <div className="slot-graphs">
                {slot1 && Object.keys(slot1).length ? (
                    <div className="slot-graph">
                        <h3>Phone Carriers (Slot 1)</h3>
                        <ResponsiveContainer width="100%" height={250}>
                            <BarChart data={Object.entries(slot1)
                            .map(([carrier, count]) => ({ carrier, count }))
                            .sort((a, b) => b.count - a.count)}>
                            <XAxis dataKey="carrier" />
                            <YAxis />
                            <Tooltip />
                            <Bar dataKey="count" fill="#C4D600" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                ) : (
                    <p>Loading phone carriers...</p>
                )}

                {slot2 && Object.keys(slot2).length ? (
                    <div className="slot-graph">
                        <h3>Phone Carriers (Slot 2)</h3>
                        <ResponsiveContainer width="100%" height={250}>
                            <BarChart data={Object.entries(slot2)
                            .map(([carrier, count]) => ({ carrier, count }))
                            .sort((a, b) => b.count - a.count)}>
                            <XAxis dataKey="carrier" />
                            <YAxis />
                            <Tooltip />
                            <Bar dataKey="count" fill="#C4D600" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                ) : (
                    <p>Loading phone carriers...</p>
                )}
            </div>
                
            {carrierData && Object.keys(carrierData).length ? (
                <div className="combined-graph">
                    <h3>Phone Carriers (Combined)</h3>
                    <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={Object.entries(carrierData)
                        .map(([carrier, count]) => ({ carrier, count }))
                        .sort((a, b) => b.count - a.count)}>
                        <XAxis dataKey="carrier" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="count" fill="#C4D600" />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            ) : (
                <p>Loading phone carriers...</p>
            )}
            
        </div>
        
        <div className="graph-container">
        {carrierData && Object.keys(carrierData).length ? (
            <>
            {/* Pie Chart for Inserted vs Uninserted */}
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

            <div className="slot-graph-summary-container">
                <h3>Summary - Inserted vs Uninserted</h3>
                <ul>
                <li><strong>Inserted:</strong> {insertedVsUninserted.inserted}</li>
                <li><strong>Uninserted:</strong> {insertedVsUninserted.uninserted}</li>
                <li><strong>Percentage Difference:</strong> {percentageDifference}%</li>
                </ul>
            </div>
            </>
        ) : (
            <div>
            <p>Loading inserted vs uninserted...</p>
            </div>
        )}
        </div>

        <div className="graph-container">
            <div className="chart">
            <h2>SIM Card Counts by Country</h2>
            <ResponsiveContainer width="100%" height={400}>
                <BarChart data={Object.entries(carrierCountry)
                .map(([country, count]) => ({ country, count }))
                .sort((a, b) => b.count - a.count)} >
                <XAxis dataKey="country" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#C4D600" />
                </BarChart>
            </ResponsiveContainer>
            </div>
            <div className='slot-graph-summary-container'>
                <h3>Top 10 Countries by SIM Card Count</h3>
                <ul>
                    {getTop10Country().map((country, index) => (
                        <li key={index}>{country.country}: {country.count}</li>
                    ))}
                </ul>
            </div>
        </div>


        {/* AI Summary */}
        <AiSummary 
          selectedFile={selectedFile} 
          selectedSheet={selectedSheet} 
          selectedColumn={["Slot 1", "Slot 2"]}
        />

    </div>


    );
};

export default SlotsInfo;
