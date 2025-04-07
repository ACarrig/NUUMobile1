import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";

import "./App.css";
import Navbar from "./components/Navbar";
import FileUpload from "./components/FileUpload";
import Dashboard from "./components/Dashboard/Dashboard";
import Predictions from "./components/Predictions/Predictions";
import AppData from "./DataBreakdown/AppData";
import AgeRange from "./DataBreakdown/AgeRange"
import ModelType from "./DataBreakdown/ModelType"
import SimInfo from "./DataBreakdown/SimInfo"
import SlotsInfo from "./DataBreakdown/SlotsInfo";
import ReturnsInfo from "./DataBreakdown/ReturnsInfo";
import ParamCorr from "./DataBreakdown/ParamCorr";
import MonthlySales from "./DataBreakdown/MonthlySales";

const App = () => {
  return (
    <Router>
      <Navbar /> {/* Place Navbar here to make it appear on all pages */}

      <Routes>
        {/* Redirect default route (/) to /upload */}
        <Route path="/" element={<Navigate to="/upload" />} />

        {/* Upload Page Route */}
        <Route path="/upload" element={<FileUpload />} />

        {/* Dashboard Page Route */}
        <Route path="/dashboard" element={<Dashboard />} />

        {/* Predictions Page Route */}
        <Route path="/predictions" element={<Predictions />} />

        {/* App Usage Page Route */}
        <Route path="/appdata" element={<AppData />} />
        
        {/* Age Rane Page Route */}
        <Route path="/agerange" element={<AgeRange />} />

        {/* Model Type Page Route */}
        <Route path="/modeltype" element={<ModelType />} />

        {/* Returns Info Page Route */}
        <Route path="/returnsinfo" element={<ReturnsInfo />} />

        {/* Sim Info Page Route */}
        <Route path="/sim_info" element={<SimInfo />} />

        {/* Slots Info Page Route */}
        <Route path="/slots_info" element={<SlotsInfo />} />

        {/* Parameter Churn Correlation Route */}
        <Route path="/paramcorr" element={<ParamCorr />} />

        {/* Monthly Sales Page Route */}
        <Route path="/monthlysales" element={<MonthlySales />} />
        
      </Routes>
    </Router>
  );
}

export default App;
