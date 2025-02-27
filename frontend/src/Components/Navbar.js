import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css'; // Import the CSS file for styles

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-logo">
      <img src={process.env.PUBLIC_URL + '/assets/NuuMobileLogo.png'} alt="Logo" className="logo" />
      </div>
      <div className="navbar-items">
        <Link to="/upload" className="nav-item">Upload</Link>
      </div>
    </nav>
  );
};

export default Navbar;