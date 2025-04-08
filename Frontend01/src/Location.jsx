import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Location.css';


const GoaLocationPage = () => {
  const navigate = useNavigate();
  
  const handleStartPlanning = () => {
    navigate('/profile'); // Navigate to planning page
  };

  return (
    <div className="location-page">
      {/* Header */}
      <header className="location-header">
        <div className="logo-container">
          <button className="logo-circle" ></button>
          <h1 className="logo-text">PlanIT</h1>
        </div>
        <div className="profile-icon">
          <div className="profile-circle" onClick={() => navigate('/profile')}></div>
        </div>
      </header>

      {/* Main content */}
      <main className="location-content">
      <div className="location-subtitle-container">
  <a href="/locations" className="location-subtitle-link">
    <h2 className="location-subtitle">Select your Location</h2>
    <div className="location-subtitle-icon">
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M9 18l6-6-6-6"/>
      </svg>
    </div>
  </a>
</div>

        <div className="destination-container">
          <div className="destination-info">
            <h1 className="destination-name">GOA</h1>
            <p className="destination-description">
              Goa, India's smallest state, is famous for its sun-kissed 
              beaches, vibrant nightlife, rich Portuguese heritage, and 
              delicious seafood. It offers a perfect mix of relaxation and 
              adventure, with activities like water sports, island tours, and 
              wildlife exploration. The state's charming towns, historic 
              churches, and bustling markets reflect its unique culture. 
              Whether you seek party vibes in North Goa or serene getaways 
              in South Goa.
            </p>
            <button 
              className="planning-button"
              onClick={handleStartPlanning}
            >
              Start Planning
            </button>
          </div>
          
          <div className="destination-image-container">
            <img 
              src="/Images/img6.jpg" 
              alt="Goa beach with umbrella chairs" 
              className="destination-image"
            />
          </div>
        </div>
      </main>
    </div>
  );
};

export default GoaLocationPage;