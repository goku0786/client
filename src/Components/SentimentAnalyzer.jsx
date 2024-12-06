import React, { useState } from "react";
import axios from "axios";

const SentimentAnalyzer = () => {
  const [text, setText] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAnalyze = async () => {
    setLoading(true);
    setError("");
    setResults(null);

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", {
        text,
      });
      setResults(response.data.results);
    } catch (err) {
      setError("An error occurred while fetching sentiment analysis.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="bg-white shadow-lg rounded-lg p-8 w-full max-w-lg">
        <h1 className="text-2xl font-bold text-gray-800 mb-4 text-center">
          Aspect-Based Sentiment Analyzer
        </h1>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text for sentiment analysis..."
          className="w-full h-32 border border-gray-300 rounded-md p-4 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          onClick={handleAnalyze}
          className="w-full bg-blue-500 text-white py-2 px-4 rounded-md mt-4 hover:bg-blue-600 focus:ring-2 focus:ring-blue-400"
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>
        {error && <div className="text-red-500 text-center mt-4">{error}</div>}
        {results && (
          <div className="mt-6">
            <h2 className="text-lg font-bold text-gray-700">Results:</h2>
            <ul className="mt-2 space-y-2">
              {Object.entries(results).map(([aspect, sentiment]) => (
                <li
                  key={aspect}
                  className="flex justify-between p-3 bg-gray-100 rounded-md"
                >
                  <span className="font-medium">{aspect}:</span>
                  <span
                    className={`font-bold ${
                      sentiment === "positive"
                        ? "text-green-500"
                        : sentiment === "negative"
                        ? "text-red-500"
                        : "text-blue-500"
                    }`}
                  >
                    {sentiment}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default SentimentAnalyzer;
