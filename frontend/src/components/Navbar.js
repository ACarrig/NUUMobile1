import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-logo">
        <img src={process.env.PUBLIC_URL + '/assets/NuuMobileLogo.png'} alt="Logo" className="logo" />
      </div>
      <div className="navbar-items">
        <Link to="/upload" className="nav-item">Upload</Link>
        <Link to="/dashboard" className="nav-item">Dashboard</Link>
        <Link to="/predictions" className="nav-item">Predictions</Link>
      </div>
      <div className="navbar-right"></div> {/* Empty div for right-side space */}
    </nav>
  );
};

export default Navbar;
