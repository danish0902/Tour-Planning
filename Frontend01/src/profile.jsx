import React, { useState } from 'react';
import './profile.css';

const Profile = () => {
  const [section, setSection] = useState('info');
  const [editable, setEditable] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const [userInfo, setUserInfo] = useState({
    name: 'John Doe',
    email: 'john.doe@example.com',
    phone: '+1-123-456-7890',
  });

  const toggleSidebar = () => setIsSidebarOpen(prev => !prev);

  const handleChange = (e) => {
    setUserInfo({ ...userInfo, [e.target.name]: e.target.value });
  };

  const handleEdit = () => {
    setEditable(!editable);
  };

  const renderSection = () => {
    switch (section) {
      case 'info':
        return (
          <div className="section-content">
            <h2>👤 Personal Information</h2>
            <div className="profile-field">
              <label>Name:</label>
              {editable ? (
                <input type="text" name="name" value={userInfo.name} onChange={handleChange} />
              ) : (
                <span>{userInfo.name}</span>
              )}
            </div>
            <div className="profile-field">
              <label>Email:</label>
              {editable ? (
                <input type="email" name="email" value={userInfo.email} onChange={handleChange} />
              ) : (
                <span>{userInfo.email}</span>
              )}
            </div>
            <div className="profile-field">
              <label>Phone:</label>
              {editable ? (
                <input type="text" name="phone" value={userInfo.phone} onChange={handleChange} />
              ) : (
                <span>{userInfo.phone}</span>
              )}
            </div>
            <button className="action-button" onClick={handleEdit}>
              {editable ? '💾 Save' : '✏️ Edit'}
            </button>
          </div>
        );
      case 'trips':
        return (
          <div className="section-content">
            <h2>🧳 Previous Trips</h2>
            <ul className="trip-list">
              <li>Rome – March 2023</li>
              <li>Paris – Dec 2022</li>
              <li>New York – Aug 2022</li>
            </ul>
          </div>
        );
      case 'password':
        return (
          <div className="section-content">
            <h2>🔐 Change Password</h2>
            <input type="password" placeholder="Old Password" className="input-field" />
            <input type="password" placeholder="New Password" className="input-field" />
            <input type="password" placeholder="Confirm New Password" className="input-field" />
            <button className="action-button">Update Password</button>
          </div>
        );
      case 'settings':
        return (
          <div className="section-content">
            <h2>⚙️ Settings</h2>
            <p>Language: English</p>
            <p>Notifications: Enabled</p>
            <button className="action-button">Edit Settings</button>
          </div>
        );
      case 'logout':
        return (
          <div className="section-content">
            <h2>🚪 Logout</h2>
            <p>Are you sure you want to logout?</p>
            <button className="action-button logout">Logout</button>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="profile-container">
      <button className="hamburger-button" onClick={toggleSidebar}>
        ☰
      </button>

      {isSidebarOpen && (
        <div className="profile-sidebar">
          <h1 className="logo">PlanIt</h1>
          <nav>
            <button onClick={() => setSection('info')}>Personal Info</button>
            <button onClick={() => setSection('trips')}>Previous Trips</button>
            <button onClick={() => setSection('password')}>Change Password</button>
            <button onClick={() => setSection('settings')}>Settings</button>
            <button onClick={() => setSection('logout')}>Logout</button>
          </nav>
        </div>
      )}

      <div className={`profile-main ${isSidebarOpen ? 'sidebar-open' : ''}`}>
        {renderSection()}
      </div>
    </div>
  );
};

export default Profile;
