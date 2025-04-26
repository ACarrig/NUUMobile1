import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import "./Analysis.css";

const ParamCorr = () => {
    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const selectedFile = queryParams.get('file');
    const selectedSheet = queryParams.get('sheet');

    const [correlation, setCorrelation] = useState([]);
    const [corrMap, setCorrMap] = useState([]);
    const [aiSummary, setAiSummary] = useState("");

    useEffect(() => {
        if (selectedFile) {
            const fetchCorrelation = async () => {
                try {
                    const response = await fetch(`http://localhost:5001/param_churn_correlation/${selectedFile}/${selectedSheet}`);
                    const data = await response.json();

                    if (data.corr) {
                        setCorrelation(data.corr);
                    } else {
                        console.log('No data found');
                    }
                }
                catch (error) {
                    console.log('Error fetching correlation data:', error);
                }
            }
            fetchCorrelation();
        }
    }, [selectedFile, selectedSheet]);

    // Format the correlation data for plotting
    const formattedData = correlation ? Object.entries(correlation).map(([param, value]) => ({ param, value })) : [];

    useEffect(() => {
        if (selectedFile) {
            const fetchCorrMap = async () => {
                try {
                    const response = await fetch(`http://localhost:5001/churn_corr_heatmap/${selectedFile}/${selectedSheet}`);
                    const data = await response.json();

                    if (data.image) {
                        setCorrMap(data.image);
                    } else {
                        console.log('No data found');
                    }
                }
                catch (error) {
                    console.log('Error fetching correlation heatmap:', error);
                }
            }
            fetchCorrMap();
        }
    }, [selectedFile, selectedSheet]);


    useEffect(() => {
        const fetchAiSummary = async () => {
            if (!selectedFile) return;

            try {
                const response = await fetch(`http://localhost:5001/churn_corr_summary/${selectedFile}/${selectedSheet}`);
                const data = await response.json();

                if (data && data.aiSummary) {
                    setAiSummary(data.aiSummary);
                } else {
                    console.warn('No AI summary received');
                }
            } catch (error) {
                console.error('Error fetching AI summary:', error);
            }
        };

        fetchAiSummary();
    }, [selectedFile, selectedSheet]);

    return (
        <div className="content">
            <h1>Churn Correlations for {selectedFile} - {selectedSheet}</h1>
            {formattedData.length ? (
                <div className="graph-container">
                    <div className='chart'>
                        <h2>Params Correlated with Churn</h2>
                        <ResponsiveContainer width="100%" height={400}>
                            <BarChart
                                data={formattedData.sort((a, b) => a.value - b.value)}  // Sort by correlation value
                                layout="vertical"
                            >
                                <XAxis type="number" domain={[-1, 1]} />  {/* Force -1 to 1 range */}
                                <YAxis type="category" dataKey="param" width={150} />
                                <Tooltip />
                                <ReferenceLine x={0} stroke="#000" strokeWidth={1} />
                                <Bar dataKey="value" fill="#C4D600" />
                            </BarChart>
                            </ResponsiveContainer>

                    </div>
                </div>
            ) : (
                <p>Loading correlation data...</p>
            )}

            {corrMap.length? (
                <div className='graph-container'>
                    <div className='chart'>
                        <h2>Correlation Heatmap</h2>
                        <img
                            src={`data:image/png;base64,${corrMap}`}
                            alt="Correlation Map"
                            style={{ width: '100%', maxWidth: '800px', height: 'auto' }}
                        />
                    </div>
                </div>
            ): (
                <p>Loading correlation map...</p>
            )}

            {/* {correlation && Object.keys(correlation).length ? (
            <div className="graph-container">
                <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={Object.entries(correlation)
                        .map(([corr, count]) => ({ corr, count }))
                        .sort((a, b) => b.count - a.count)
                        .slice(0, 5)}>
                        <XAxis dataKey="corr" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="count" fill="#C4D600" />
                    </BarChart>
                </ResponsiveContainer>
            </div>
            ):(
                <p>Loading Parameter Correlation Bar Chart...</p>
            )} */}

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

export default ParamCorr;
