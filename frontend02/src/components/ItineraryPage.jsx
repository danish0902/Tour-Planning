import React from 'react';
import { useLocation } from 'react-router-dom';

function ItineraryPage() {
  const location = useLocation();
  const itinerary = location.state?.itinerary;

  if (!itinerary) return <div className="p-6">No itinerary found. Please go back and enter your trip details.</div>;

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Your Goa Itinerary</h1>
      {itinerary.map((day, index) => (
        <div key={index} className="mb-6 border rounded p-4 bg-gray-50">
          <h2 className="text-lg font-semibold mb-2">Day {day.day}</h2>
          {day.itinerary.map((item, idx) => (
            <div key={idx} className="mb-3">
              <h3 className="font-medium">{item.Type}</h3>
              {item.Name && <p><strong>Name:</strong> {item.Name}</p>}
              {item.Cuisine && <p><strong>Cuisine:</strong> {item.Cuisine}</p>}
              {item.Category && <p><strong>Category:</strong> {item.Category}</p>}
              {item.Rating && <p><strong>Rating:</strong> {item.Rating}</p>}
              {item.Time && <p><strong>Time:</strong> {item.Time[0]} - {item.Time[1]}</p>}
              {day.distances[idx] !== null && <p><strong>Distance:</strong> {day.distances[idx].toFixed(2)} km</p>}
              {item.Note && <p><em>{item.Note}</em></p>}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

export default ItineraryPage;