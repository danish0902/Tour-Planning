
import React, { useState } from 'react';
import './Login.css';

const LoginPage = () => {
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Login attempt with:', { userId, password });
    // Add your authentication logic here
  };

  return (
    <div 
      className="login-container" 
      style={{ backgroundImage: `url('/Images/img5 .jpg')` }}
    >
      <div className="login-card">
        <h1 className="welcome-title">Welcome User!!</h1>
        
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            className="input-field"
            placeholder="User id"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
          />
          
          <input
            type="password"
            className="input-field"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          
          <button type="submit" className="login-button">
            Login
          </button>
        </form>
        
        <div className="divider"></div>
        
        <p className="signup-text">
          Don't have an account?
          <a href="/signup" className="signup-link">Signup</a>
        </p>
      </div>
    </div>
  );
};

export default LoginPage;