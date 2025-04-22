import React from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import ExaminationForm from './components/ExaminationForm';
import './App.css';

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="app">
        <nav className="navbar">
          <div className="nav-brand">Eye Examination AI System</div>
          <div className="nav-links">
            <Link to="/">Home</Link>
            <Link to="/examination/1">New Examination</Link>
          </div>
        </nav>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/examination/:patientId" element={<ExaminationForm />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
};

const Home: React.FC = () => {
  return (
    <div className="home">
      <h1>Welcome to Eye Examination AI System</h1>
      <p>This system helps doctors analyze eye images for various conditions using AI.</p>
      <div className="features">
        <div className="feature">
          <h3>AI-Powered Analysis</h3>
          <p>Detect glaucoma, cataract, scarring, and other conditions</p>
        </div>
        <div className="feature">
          <h3>Comprehensive Reports</h3>
          <p>Generate detailed examination reports with AI insights</p>
        </div>
        <div className="feature">
          <h3>Patient Management</h3>
          <p>Track patient history and examination results</p>
        </div>
      </div>
      <Link to="/examination/1" className="cta-button">
        Start New Examination
      </Link>
    </div>
  );
};

export default App;
