# Refactoring Plan and Implementation for `meter-made-ui`

This document outlines the refactoring plan for the `meter-made-ui` project, focusing on breaking components early, using custom hooks, and replacing `useEffect` spaghetti with derived state or state machines.

---

## **Refactoring Goals**

1. **Break Components Early**  
   Extract reusable UI elements (e.g., form, prediction result, footer) into separate components for better modularity and readability.

2. **Use Custom Hooks**  
   Abstract complex logic (e.g., Google Maps Autocomplete, date handling) into custom hooks to simplify the main component.

3. **Replace `useEffect` Spaghetti**  
   Minimize the use of `useEffect` by using derived state or state machines where applicable.

---

## **Proposed File Structure**

The refactored project will have the following structure:

```
meter-made-ui/
├── src/
│   ├── app/
│   │   ├── page.tsx
│   ├── components/
│   │   ├── Form.tsx
│   │   ├── PredictionResult.tsx
│   │   ├── Footer.tsx
│   ├── hooks/
│   │   ├── useGoogleAutocomplete.ts
│   │   ├── useDateHandler.ts
│   ├── styles/
│   │   ├── custom-datepicker.css
├── public/
│   ├── screenshot.png
├── README.md
```

---

## **Custom Hooks**

### **`useGoogleAutocomplete`**
Handles Google Maps Autocomplete logic.

```tsx
// filepath: /Users/reedmarkham/Documents/github/meter-made/meter-made-ui/src/hooks/useGoogleAutocomplete.ts
import { useEffect, useRef, useState } from "react";

export function useGoogleAutocomplete(isLoaded: boolean, loadError: boolean, onPlaceChanged: (place: google.maps.places.PlaceResult) => void) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isLoaded || loadError || !inputRef.current) return;

    const options = {
      componentRestrictions: { country: "us" },
      fields: ["address_components", "geometry"],
    };

    if (typeof window !== "undefined" && window.google?.maps) {
      const autocomplete = new window.google.maps.places.Autocomplete(inputRef.current!, options);
      autocomplete.addListener("place_changed", () => {
        const place = autocomplete.getPlace();
        if (!place || !place.geometry || !place.geometry.location) {
          setError("Invalid place selected.");
          return;
        }
        setError(null);
        onPlaceChanged(place);
      });

      return () => window.google.maps.event.clearInstanceListeners(autocomplete);
    }
  }, [isLoaded, loadError, onPlaceChanged]);

  return { inputRef, error };
}
```

---

### **`useDateHandler`**
Manages date and time selection logic.

```tsx
// filepath: /Users/reedmarkham/Documents/github/meter-made/meter-made-ui/src/hooks/useDateHandler.ts
import { useState } from "react";

export function useDateHandler(initialDate: string, initialHour: number) {
  const [date, setDate] = useState(initialDate);
  const [hour, setHour] = useState(initialHour);

  const safeParseDate = (dateStr: string) => {
    const parsedDate = new Date(dateStr);
    return isNaN(parsedDate.getTime()) ? new Date() : parsedDate;
  };

  const getDateWithTime = () => {
    const parsedDate = safeParseDate(date);
    parsedDate.setHours(hour);
    return parsedDate;
  };

  const handleDateChange = (selectedDate: Date | null) => {
    if (selectedDate) {
      setDate(selectedDate.toISOString().split("T")[0]);
      setHour(selectedDate.getHours());
    }
  };

  return { getDateWithTime, handleDateChange };
}
```

---

## **Components**

### **Form Component**
Handles the input form for address and date selection.

```tsx
// filepath: /Users/reedmarkham/Documents/github/meter-made/meter-made-ui/src/components/Form.tsx
import React from "react";
import DatePicker from "react-datepicker";

interface FormProps {
  inputRef: React.RefObject<HTMLInputElement>;
  getDateWithTime: () => Date;
  handleDateChange: (date: Date | null) => void;
  handleSubmit: (e: React.FormEvent) => void;
  isLoading: boolean;
  error: string | null;
}

export function Form({ inputRef, getDateWithTime, handleDateChange, handleSubmit, isLoading, error }: FormProps) {
  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4 w-full">
      <input
        ref={inputRef}
        type="text"
        placeholder="Enter an address"
        className="border p-2 w-full rounded"
      />
      <DatePicker
        selected={getDateWithTime()}
        onChange={handleDateChange}
        showTimeSelect
        dateFormat="Pp"
        className="border p-2 w-full rounded custom-datepicker"
        timeIntervals={15}
      />
      <button type="submit" className="bg-blue-500 text-white p-2 rounded" disabled={isLoading}>
        {isLoading ? "Loading..." : "Submit"}
      </button>
      {error && <div className="mt-4 text-red-500">{error}</div>}
    </form>
  );
}
```

---

### **Prediction Result Component**
Displays the prediction result.

```tsx
// filepath: /Users/reedmarkham/Documents/github/meter-made/meter-made-ui/src/components/PredictionResult.tsx
import React from "react";

interface PredictionResultProps {
  predictionResult: number | null;
}

export function PredictionResult({ predictionResult }: PredictionResultProps) {
  if (predictionResult === null) return null;

  const message =
    predictionResult === 0
      ? "You are unlikely to get an expired meter ticket"
      : "You are likely to get an expired meter ticket";

  const bgColor = predictionResult === 0 ? "bg-[#003B5C]" : "bg-[#56A0D3]";

  return (
    <div className={`mt-4 p-4 border rounded ${bgColor} text-white`}>
      <strong>Prediction Result:</strong> {message}
    </div>
  );
}
```

---

### **Footer Component**
Displays the footer with contact information.

```tsx
// filepath: /Users/reedmarkham/Documents/github/meter-made/meter-made-ui/src/components/Footer.tsx
import React from "react";

export function Footer() {
  return (
    <footer className="mt-8 text-center text-white">
      <a href="mailto:reedmarkham@gmail.com" className="flex items-center justify-center gap-2">
        <span>💌</span> reedmarkham@gmail.com
      </a>
    </footer>
  );
}
```

---

## **Refactored Main Component**

```tsx
// filepath: /Users/reedmarkham/Documents/github/meter-made/meter-made-ui/src/app/page.tsx
import React, { useState } from "react";
import { useLoadScript } from "@react-google-maps/api";
import { useGoogleAutocomplete } from "../hooks/useGoogleAutocomplete";
import { useDateHandler } from "../hooks/useDateHandler";
import { Form } from "../components/Form";
import { PredictionResult } from "../components/PredictionResult";
import { Footer } from "../components/Footer";

const libraries = ["places"];

export default function App() {
  const apiKey = process.env.NEXT_PUBLIC_GOOGLE_API_KEY;
  if (!apiKey) {
    throw new Error("NEXT_PUBLIC_GOOGLE_API_KEY environment variable is not defined");
  }

  const { isLoaded, loadError } = useLoadScript({
    googleMapsApiKey: apiKey,
    libraries,
  });

  const [input, setInput] = useState({ x: 0, y: 0 });
  const [predictionResult, setPredictionResult] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSubmitted, setHasSubmitted] = useState(false);

  const { getDateWithTime, handleDateChange } = useDateHandler(new Date().toISOString(), new Date().getHours());
  const { inputRef, error } = useGoogleAutocomplete(isLoaded, !!loadError, (place) => {
    const location = place.geometry?.location;
    if (location) {
      setInput({ x: location.lng(), y: location.lat() });
    }
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (error) {
      alert(error);
      return;
    }
    setIsLoading(true);
    setHasSubmitted(true);
    try {
      const result = await makePrediction({ ...input, d: getDateWithTime().toISOString(), h: getDateWithTime().getHours() });
      setPredictionResult(result);
    } catch (err) {
      console.error(err);
      alert("Prediction failed.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-900">
      <div className="w-full max-w-xl flex flex-col gap-4 bg-gray-800 p-6 rounded-lg shadow-lg">
        <h1 className="text-center text-white font-semibold text-4xl">meter-made</h1>
        <h2 className="text-center text-white text-2xl">
          A machine learning model to predict parking tickets for expired meters in Washington, DC
        </h2>
        <Form
          inputRef={inputRef}
          getDateWithTime={getDateWithTime}
          handleDateChange={handleDateChange}
          handleSubmit={handleSubmit}
          isLoading={isLoading}
          error={error}
        />
        {!hasSubmitted && <div className="mt-4 text-white">Please select a DC address, date, and time above</div>}
        {isLoading && <div className="mt-4 text-white">Loading...</div>}
        <PredictionResult predictionResult={predictionResult} />
        <Footer />
      </div>
    </div>
  );
}

async function makePrediction(inputData: any) {
  const apiUrl = process.env.NEXT_PUBLIC_MODEL_API;
  if (!apiUrl) {
    throw new Error("NEXT_PUBLIC_MODEL_API environment variable is not defined");
  }

  const response = await fetch(apiUrl, {
    method: "POST",
    headers: {
      accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(inputData),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Prediction failed");
  }
  return data.ticketed;
}
```

---

## **Benefits of Refactoring**

1. **Separation of Concerns**: Each component and hook has a single responsibility.
2. **Reusability**: Hooks and components can be reused in other parts of the app.
3. **Readability**: The main component is now much cleaner and easier to understand.
4. **Scalability**: Adding new features or modifying existing ones becomes simpler.
