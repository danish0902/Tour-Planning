import React, { useState } from 'react';
import './signup.css';
import { useNavigate } from 'react-router-dom';

const Signup = () => {
     const navigate = useNavigate();
      
      
    
  const [showMore, setShowMore] = useState(false);

  return (
    <div className="signup-container">
      {/* Left Panel */}
      <div className="signup-left">
        <h1 className="logo"> PlanIT</h1>
        <h2 className="tagline">Travel Smart,<br />Explore Better!</h2>
        <p className="login-text">
          Already have an account? <a href="/login"><strong>Login</strong></a>
        </p>
      </div>

      {/* Right Panel */}
      <div className="signup-card">
        <h2 className="card-title">Create Account</h2>
        <form>
          <input type="text" placeholder="Name" className="input-field" />
          <input type="email" placeholder="Email" className="input-field" />
          <input type="text" placeholder="Username" className="input-field" />
          <input type="password" placeholder="Password" className="input-field" />

          {/* Hidden Inputs */}
          {showMore && (
            <>
              <input type="text" placeholder="Phone" className="input-field" />
              <input type="text" placeholder="City" className="input-field" />
              <input type="text" placeholder="Country" className="input-field" />
            </>
          )}

          <p className="more-btn" onClick={() => setShowMore(!showMore)}>
            More {showMore ? '▲' : '▼'}
          </p>

          <button type="submit"  onClick={() => navigate('/planning')} className="register-button">Register</button>
        </form>
      </div>
    </div>
  );
};

export default Signup;
