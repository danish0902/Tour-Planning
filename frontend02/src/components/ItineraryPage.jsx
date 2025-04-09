import React from 'react';
import { useLocation } from 'react-router-dom';

function ItineraryPage() {
  const location = useLocation();
  const itinerary = location.state?.itinerary;

  if (!itinerary) {
    return <div className="p-6 text-center">No itinerary found. Please go back and try again.</div>;
  }

  return (
    <div className="p-6 max-w-6xl mx-auto bg-gray-50 min-h-screen">
      <h1 className="text-4xl font-bold mb-12 text-center text-indigo-700">Your Goa Itinerary</h1>

      {itinerary.map((dayPlan, index) => (
        <div key={index} className="mb-16">
          <h2 className="text-3xl font-semibold mb-6 text-indigo-600 text-center">Day {dayPlan.day}</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {dayPlan.itinerary.map((item, idx) => {
              let sectionTitle = "";
              if (item.Type.toLowerCase().includes("breakfast")) sectionTitle = "Breakfast Restaurant";
              else if (item.Type.toLowerCase().includes("lunch")) sectionTitle = "Lunch Restaurant";
              else if (item.Type.toLowerCase().includes("dinner")) sectionTitle = "Dinner Restaurant";
              else if (item.Type.toLowerCase().includes("tourist")) sectionTitle = item.Type;
              else if (item.Type.toLowerCase().includes("note")) sectionTitle = "Note";

              return (
                <div key={idx} className="bg-white rounded-2xl shadow-md p-6 border border-gray-200">
                  <h3 className="text-xl font-semibold text-indigo-700 mb-4">{sectionTitle}</h3>
                  {item.Name && <p className="text-gray-800"><strong>Name:</strong> {item.Name}</p>}
                  {item.Cuisine && <p className="text-gray-800"><strong>Cuisine:</strong> {item.Cuisine}</p>}
                  {item.Category && <p className="text-gray-800"><strong>Category:</strong> {item.Category}</p>}
                  {item.Rating && <p className="text-gray-800"><strong>Rating:</strong> {item.Rating}</p>}
                  {item.Time && <p className="text-gray-800"><strong>Time:</strong> {item.Time[0]} - {item.Time[1]}</p>}
                  {dayPlan.distances[idx] !== null && (
                    <p className="text-gray-800"><strong>Distance:</strong> {dayPlan.distances[idx].toFixed(2)} km</p>
                  )}
                  {item.Note && <p className="text-red-600 mt-2 font-medium">{item.Note}</p>}
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

export default ItineraryPage;
