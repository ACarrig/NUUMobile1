import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import "./ReturnsInfo.css";
import AiSummary from "./Summary"

const ReturnsInfo = () => {
    const [returnsData, setReturnsData] = useState([]);
    const [numReturns, setNumReturns] = useState(0);
    const [feedback, setFeedback] = useState([]);
    const [verification, setVerification] = useState([]);    
    const [resParty, setResParty] = useState([]);    

    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const selectedFile = queryParams.get('file');
    const selectedSheet = queryParams.get('sheet');

    const [aiComparisonSummary, setAiComparisonSummary] = useState("");
    const lastRequestRef = useRef(""); // Store last request signature
    
    useEffect(() => {
        if (selectedFile && selectedSheet) {
            const fetchReturnsData = async () => {
              try {
                const response = await fetch(`http://localhost:5001/device_return_info/${selectedFile}/${selectedSheet}`);
                const data = await response.json();
                if (data.defects) {
                  setReturnsData(data.defects);
                } else {
                    console.log('No defects data found');
                }
              } catch (error) {
                console.log('Error fetching returns data:', error);
              }
            };

            fetchReturnsData();
        }
    }, [selectedFile, selectedSheet]);


      useEffect(() => {
        if (selectedFile && selectedSheet) {
            const fetchNumReturns = async () => {
                try {
                    const response = await fetch(`http://localhost:5001/num_returns/${selectedFile}/${selectedSheet}`)
                    const data = await response.json();
                    if (data && data.num_returns) {
                        setNumReturns(data.num_returns);
                    } else {
                        console.log('No defect count received');
                    }
                } catch (error) {
                    console.log('Error fetching returns data:', error);
                }
            };

            fetchNumReturns();
        }
    }, [selectedFile, selectedSheet]);

    useEffect(() => {
        if (selectedFile && selectedSheet) {
            const fetchFeedback = async () => {
                try {
                    const response = await fetch(`http://localhost:5001/feedback_info/${selectedFile}/${selectedSheet}`);
                    const data = await response.json();
                        if (data.feedback) {
                            setFeedback(data.feedback);
                        } else {
                            console.log('No feedback data found');
                        }
                    } catch (error) {
                        console.log('Error fetching feedback:', error);
                }
            };

        fetchFeedback();
        }
    }, [selectedFile, selectedSheet]);

    useEffect(() => {
        if (selectedFile && selectedSheet) {
            const fetchVerification = async () => {
                try {
                    const response = await fetch(`http://localhost:5001/verification_info/${selectedFile}/${selectedSheet}`);
                    const data = await response.json();
                    if (data.verification) {
                        setVerification(data.verification);
                    } else {
                        console.log('No verification data found');
                    }
                } catch (error) {
                    console.log('Error fetching verification:', error);
                }
            };

            fetchVerification();
        }
    }, [selectedFile, selectedSheet]);

    useEffect(() => {
        if (selectedFile && selectedSheet) {
            const fetchResParty = async () => {
                try {
                    const response = await fetch(`http://localhost:5001/resparty_info/${selectedFile}/${selectedSheet}`);
                    const data = await response.json();
                    if (data.responsible_party) {
                        setResParty(data.responsible_party);
                    } else {
                        console.log('No responsible party found');
                    }
                } catch (error) {
                    console.log('Error fetching responsible party:', error);
                }
            };

            fetchResParty();
        }
    }, [selectedFile, selectedSheet]);

    const mostCommonDefect = returnsData && Object.entries(returnsData)
    .reduce((a, b) => (a[1] > b[1] ? a : b), ["", 0]);

    const mostCommonFeedback = feedback && Object.entries(feedback)
    .reduce((a, b) => (a[1] > b[1] ? a : b), ["", 0]);

    const mostCommonVerification = verification && Object.entries(verification)
        .reduce((a, b) => (a[1] > b[1] ? a : b), ["", 0]);

    const mostCommonParty = resParty && Object.entries(resParty)
        .reduce((a, b) => (a[1] > b[1] ? a : b), ["", 0]);

    useEffect(() => {
        if (selectedFile && selectedSheet) {
            if (!selectedFile || !selectedSheet) return;
        
            const requestSignature = `${selectedFile}__${selectedSheet}`;
        
            if (requestSignature === lastRequestRef.current) return; // Avoid unnecessary repeated requests
            lastRequestRef.current = requestSignature; // Save this as the last request
    
            const fetchAiComparisonSummary = async () => {
                try {
                    const response = await fetch(`http://localhost:5001/returns_comparison_summary?file=${selectedFile}&sheet=${selectedSheet}`);
                    const data = await response.json();
                    if (data.summary) {
                        setAiComparisonSummary(data.summary);
                    } else {
                        console.log('No summary found');
                    }
                } catch (error) {
                    console.log('Error fetching summary:', error);
                }
            };

            fetchAiComparisonSummary();
        }
    }, [selectedFile, selectedSheet]);

    return (
        <div className="content">
            <h1>Return Info for {selectedFile} - {selectedSheet}</h1>

            <div className="analysis-container">
                {numReturns ? (
                <p><strong>Total Returns in this Data Set:</strong> {numReturns}</p>
                ) : (
                <p>Loading total...</p>
                )}
            </div>

            {/* Bar Chart for Defects Types */}
            <div className="graphs-container">
                <div className="graph-container">
                    <div className='chart'>
                        <h2>Feedback from Return</h2>
                        <p>Customer feedback</p>
                        <ResponsiveContainer width="100%" height={400}>
                            <BarChart data={Object.entries(feedback)
                            .map(([feedback, count]) => ({ feedback, count }))
                            .sort((a, b) => b.count - a.count)}>
                            <XAxis dataKey="feedback" />
                            <YAxis />
                            <Tooltip />
                            <Bar dataKey="count" fill="#C4D600" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
                <AiSummary 
                    selectedFile={selectedFile} 
                    selectedSheet={selectedSheet} 
                    selectedColumn={["Feedback"]}
                />
            </div>

            <div className="graphs-container">
                <div className="graph-container">
                    <div className="chart">
                        <h2>Type of Defects in Return</h2>
                        <p>Verified issues from the testing team</p>
                        <ResponsiveContainer width="100%" height={400}>
                        <BarChart data={Object.entries(returnsData)
                            .map(([defects, count]) => ({ defects, count }))
                            .sort((a, b) => b.count - a.count)} >
                            <XAxis dataKey="defects" />
                            <YAxis />
                            <Tooltip />
                            <Bar dataKey="count" fill="#C4D600" />
                        </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
                <AiSummary 
                    selectedFile={selectedFile} 
                    selectedSheet={selectedSheet} 
                    selectedColumn={["Defect / Damage type"]}
                />
            </div>

            <div className="graphs-container">
                <div className="graph-container">
                    <div className='chart'>
                        <h2>Verification of Defects in Return</h2>
                        <p>Damage type</p>
                        <ResponsiveContainer width="100%" height={400}>
                            <BarChart data={Object.entries(verification)
                            .map(([verification, count]) => ({ verification, count }))
                            .sort((a, b) => b.count - a.count)}>
                            <XAxis dataKey="verification" />
                            <YAxis />
                            <Tooltip />
                            <Bar dataKey="count" fill="#C4D600" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
                <AiSummary 
                    selectedFile={selectedFile} 
                    selectedSheet={selectedSheet} 
                    selectedColumn={["Verification"]}
                />
            </div>

            <div className="graphs-container">
                <div className="graph-container">
                    <div className='chart'>
                        <h2>Responsible Party of Defects in Return</h2>
                        <p>The reason why the device is returned</p>
                        <ResponsiveContainer width="100%" height={400}>
                            <BarChart data={Object.entries(resParty)
                            .map(([resParty, count]) => ({ resParty, count }))
                            .sort((a, b) => b.count - a.count)}>
                            <XAxis dataKey="resParty" />
                            <YAxis />
                            <Tooltip />
                            <Bar dataKey="count" fill="#C4D600" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
                <AiSummary 
                    selectedFile={selectedFile} 
                    selectedSheet={selectedSheet} 
                    selectedColumn={["Responsible Party"]}
                />
            </div>

            <div className='analysis-container'>
                <h2>Returns Summary</h2>
                    <div className='analysis-summary'>
                        {numReturns ? (
                            <p><strong>Total Returns in this Data Set:</strong> {numReturns}</p>
                        ) : (
                            <p>Loading total...</p>
                        )} <br />


                        <p>
                            Most reported feedback: <strong>{mostCommonFeedback[0]}</strong> ({mostCommonFeedback[1]} times).<br />
                            Most reported defect / damage type: <strong>{mostCommonDefect[0]}</strong> ({mostCommonDefect[1]} times).<br />
                            Most common verification result: <strong>{mostCommonVerification[0]}</strong> ({mostCommonVerification[1]} times). <br />
                            Most common responsible party: <strong>{mostCommonParty[0]}</strong> ({mostCommonParty[1]} times).
                        </p>

                    </div>

                    <div className="summary-container">
                        <h2>AI Summary</h2>
                        <div>
                            {aiComparisonSummary ? <p>{aiComparisonSummary}</p> : <p>Loading Comparison Summary...</p>}
                        </div>
                    </div>
            </div>

        </div>
    );
};

export default ReturnsInfo;
