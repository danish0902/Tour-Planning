import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './Home.jsx';
import LoginPage from './Login.jsx';
import Signup from './signup.jsx';
import GoaLocationPage from './Location.jsx';
import Profile from './profile.jsx';

import './App.css'

function App() {
  

  return (
    <>
    <div>
    
      <Router>
     <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/Login" element={<LoginPage/>} />
      <Route path="/Signup" element={<Signup/>} />
      <Route path="/planning" element={<GoaLocationPage />} />
      <Route path="/profile" element={<Profile/>} />
      
     </Routes>
     </Router>
      </div>
    </>
  )
}

export default App
