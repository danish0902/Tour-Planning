import React from "react";
import "./Home.css";
import { useNavigate } from 'react-router-dom';

const Home = () => {
  const navigate = useNavigate();
  return (
    <div className="violet-page">
      <div className="container">
        {/* Header */}
        <header className="header">
          <h1 className="logo">PlanIT</h1>
          <div className="auth-buttons">
            <button className="btn"  onClick={() => navigate('/Login')}>Login</button>
            <button className="btn"  onClick={() => navigate('/Signup')}>Signup</button>
          </div>
        </header>

        {/* Hero */}
        <section className="hero">
          <h2 className="hero-title">
            Discover Awesome Trip <br />
            You Have Never Seen
          </h2>
          <p className="subtitle">
            An arrangement you make to have a hotel room, tickets etc.
            in a particular time in future.
          </p>
          <button className="cta">Create Your Trip 🛫</button>
        </section>

        {/* Feature 1 */}
        <section className="feature">
          <img
            src="/Images/img2.jpeg"
            alt="Tour Guide"
            className="feature-img"
          />
          <div className="feature-text">
            <h3>Complete Tour Guidance</h3>
            <div className="feature-description">
            <p>
              Our website offers a complete tour guide experience, helping users
              explore destinations effortlessly. It features detailed itineraries,
              travel tips, interactive maps, and local recommendations for a seamless journey.
            </p>
            </div>
          </div>
        </section>

        {/* Feature 2 */}
        <section className="feature reverse">
          <img
            src="./Images/img1.webp"
            alt="Emergency Support"
            className="feature-img"
          />
          <div className="feature-text">
            <h3>Travel Emergency Support</h3>
            <div className="feature-description">
            <p>
              Travel emergency support provides immediate assistance in unexpected
              situations like medical emergencies, lost passports, or flight cancellations.
              Quick response ensures travelers receive the necessary help.
            </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default Home;
