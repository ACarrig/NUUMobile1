import React, { useState, useEffect } from 'react';
import '../Dashboard.css';

const CorrMapChart = ({ openWindow, selectedFile, selectedSheet }) => {
    const [corrMap, setCorrMap] = useState([]);

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

    return (
        <div className="summary-box">
            <h3>Correlation Heatmap</h3>
                {corrMap && Object.keys(corrMap).length ? (
                    <div className="summary-graph">
                        <img
                            src={`data:image/png;base64,${corrMap}`}
                            alt="Correlation Map"
                            style={{ width: '100%', maxWidth: '900px', height: 'auto' }}
                        />
                    </div>
                    ): (
                    <p>Loading correlation map...</p>
                )}
            <button onClick={() => openWindow(`/paramcorr?file=${selectedFile}&sheet=${selectedSheet}`)}>View Correlations</button>
        </div>
    );
};

export default CorrMapChart;
