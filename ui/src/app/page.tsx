"use client";

import React, { useState } from "react";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import { useGoogleMapsAutocomplete } from "../hooks/useGoogleMapsAutocomplete";
import { usePrediction } from "../hooks/usePrediction";
import { useLoadScript } from "@react-google-maps/api";
import { LoaderOptions } from "@googlemaps/js-api-loader";

const libraries: LoaderOptions["libraries"] = ["places"];

interface InputState {
  d: string;
  h: number;
  x: number;
  y: number;
}

export default function App() {
  // Always call hooks at the top level
  const { isLoaded, loadError } = useLoadScript({
    googleMapsApiKey: process.env.NEXT_PUBLIC_GOOGLE_API_KEY!,
    libraries,
  });

  const [input, setInput] = useState<InputState>({
    d: new Date().toISOString(),
    h: new Date().getHours(),
    x: 0,
    y: 0,
  });

  const { error, setError, inputRef } = useGoogleMapsAutocomplete({
    onPlaceSelected: (location, isDC) => {
      if (!isDC) {
        setError("Please select an address in the District of Columbia.");
        return;
      }
      setError(null);
      setInput((prev) => ({
        ...prev,
        x: location.lng(),
        y: location.lat(),
      }));
    },
  });

  const { isLoading, hasSubmitted, predictionResult, handleSubmit } = usePrediction({
    input,
    error,
  });

  const handleChange = (date: Date | null) => {
    if (date) {
      const dateString = date.toISOString().split("T")[0];
      const hour = date.getHours();
      setInput((prev) => ({
        ...prev,
        d: dateString,
        h: hour,
      }));
    }
  };

  const safeParseDate = (dateStr: string) => {
    const parsedDate = new Date(dateStr);
    return isNaN(parsedDate.getTime()) ? new Date() : parsedDate;
  };

  const getDateWithTime = () => {
    const date = safeParseDate(input.d);
    date.setHours(input.h);
    return date;
  };

  // Handle loading and error states after hooks are called
  if (!isLoaded) return <div>Loading...</div>;
  if (loadError) return <div>Error loading Google Maps</div>;

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-900">
      <div className="w-full max-w-xl flex flex-col gap-4 bg-gray-800 p-6 rounded-lg shadow-lg">
        <h1 className="text-center text-white font-semibold text-4xl">meter-made</h1>
        <h2 className="text-center text-white text-2xl">
          A machine learning model to predict parking tickets for expired meters in Washington, DC
        </h2>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4 w-full">
          <input
            ref={inputRef}
            type="text"
            placeholder="Enter an address"
            className="border p-2 w-full rounded"
          />
          <DatePicker
            selected={getDateWithTime()}
            onChange={handleChange}
            showTimeSelect
            dateFormat="Pp"
            className="border p-2 w-full rounded custom-datepicker"
            timeIntervals={15}
          />
          <button type="submit" className="bg-blue-500 text-white p-2 rounded" disabled={isLoading}>
            {isLoading ? "Loading..." : "Submit"}
          </button>
        </form>
        {error && <div className="mt-4 text-red-500">{error}</div>}
        {!hasSubmitted && <div className="mt-4 text-white">Please select a DC address, date, and time above</div>}
        {isLoading && <div className="mt-4 text-white">Loading...</div>}
        {predictionResult !== null && (
          <div
            className={`mt-4 p-4 border rounded ${
              predictionResult === 0 ? "bg-[#003B5C] text-white" : "bg-[#56A0D3] text-white"
            }`}
          >
            <strong>Prediction Result:</strong>{" "}
            {predictionResult === 0
              ? "You are unlikely to get an expired meter ticket"
              : "You are likely to get an expired meter ticket"}
          </div>
        )}
        <footer className="mt-8 text-center text-white">
          <a href="mailto:reedmarkham@gmail.com" className="flex items-center justify-center gap-2">
            <span>💌</span> reedmarkham@gmail.com
          </a>
        </footer>
      </div>
    </div>
  );
}
