import React from 'react';
import './NavBar.css';

// Simple version without react-router to avoid dependencies
const NavBar: React.FC = () => {
    return (
        <nav className="navbar">
            {/* Logo and Site Name */}
            <div className="navbar-brand">
                {/* Temporarily comment out the logo if it's causing issues */}
                {/* <img 
                  src="/logo.svg" 
                  alt="Logo"
                  className="navbar-logo"
                /> */}
                <span className="navbar-brand-text">
                    Stamps Collection
                </span>
            </div>

            {/* Navigation Links */}
            <div className="navbar-links">
                <a href="/" className="navbar-link">Home</a>
                <a href="/catalogue" className="navbar-link">Catalogue</a>
                <a href="/collection" className="navbar-link">Collection</a>
                <a href="/profile" className="navbar-link">Profile</a>
            </div>
        </nav>
    );
};

export default NavBar;
