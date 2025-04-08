import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

function InputForm() {
  const [numDays, setNumDays] = useState(3);
  const [locationsInput, setLocationsInput] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    const locations = locationsInput.split(',').map(loc => loc.trim());
    try {
      const response = await axios.post('http://localhost:5000/api/generate_itinerary', {
        numDays,
        locations,
      });
      navigate('/itinerary', { state: { itinerary: response.data } });
    } catch (error) {
      console.error('Error generating itinerary:', error);
    }
  };

  return (
    <div className="p-6 max-w-xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Plan Your Trip</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block mb-1">Number of Days</label>
          <input
            type="number"
            value={numDays}
            onChange={(e) => setNumDays(e.target.value)}
            min={1}
            className="w-full p-2 border rounded"
            required
          />
        </div>
        <div>
          <label className="block mb-1">Must-Visit Locations (comma-separated)</label>
          <input
            type="text"
            value={locationsInput}
            onChange={(e) => setLocationsInput(e.target.value)}
            placeholder="e.g. Baga, Calangute, Aguada"
            className="w-full p-2 border rounded"
            required
          />
        </div>
        <button type="submit" className="bg-blue-600 text-white px-4 py-2 rounded">
          Generate Itinerary
        </button>
      </form>
    </div>
  );
}

export default InputForm;
