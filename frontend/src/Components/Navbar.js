import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './Navbar.css'; // Import the CSS file for styles

const Navbar = () => {
  const navigate = useNavigate();

  return (
    <nav className="navbar">
      <div className="navbar-logo">
      <img src={process.env.PUBLIC_URL + '/assets/NuuMobileLogo.png'} alt="Logo" className="logo" />
      </div>
      <div className="navbar-items">
        <Link to="/upload" className="nav-item">Upload</Link>
      </div>
      <button className="nav-button" onClick={() => navigate(-1)}>Back</button>
    </nav>
  );
};

export default Navbar;