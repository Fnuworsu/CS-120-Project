import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import ImageUpload from './components/ImageUpload';
import './App.css';

const App: React.FC = () => {
  return (
    <Router>
      <div className="app">
        <nav className="nav-bar">
          <Link to="/" className="nav-link">Home</Link>
          <Link to="/analyze" className="nav-link">Analyze Image</Link>
        </nav>

        <Routes>
          <Route path="/" element={
            <div className="welcome-container">
              <h1>Eye Disease Detection System</h1>
              <p>Upload retinal images to detect various eye conditions using our AI-powered system.</p>
              <Link to="/analyze" className="cta-button">Start Analysis</Link>
            </div>
          } />
          <Route path="/analyze" element={<ImageUpload />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
