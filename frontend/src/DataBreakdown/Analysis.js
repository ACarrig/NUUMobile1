import React, { useEffect, useState } from "react";
import { useNavigate } from 'react-router-dom';
import "./Analysis.css";

// This page should hold a bunch of buttons giving the user options to select 
//    what kinds of analysis / pages they want to navigate to
// e.g. 'Apps' button could take a user to /appdata to see app usage visuals & metrics
const Analysis = () => {
  const navigate = useNavigate();

  return (
    <div className="analysis-container">
      <h1>Analysis Page</h1>
      <div className="button-group">
        <button className="btn" onClick={() => navigate("/upload")}>
          Go to file upload page
        </button>
        <button className="btn" onClick={() => navigate("/appdata")}>
          Go to app usage page
        </button>
      </div>
    </div>
  );
};

export default Analysis;
