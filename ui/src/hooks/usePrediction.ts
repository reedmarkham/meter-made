import { useState } from "react";

interface UsePredictionProps {
  input: { d: string; h: number; x: number; y: number };
  error: string | null
}

export function usePrediction({ input, error }: UsePredictionProps) {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [hasSubmitted, setHasSubmitted] = useState<boolean>(false);
  const [predictionResult, setPredictionResult] = useState<number | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (error) {
      alert(error);
      return;
    }
    setIsLoading(true);
    setHasSubmitted(true);
    try {
      const result = await makePrediction(input);
      setPredictionResult(result);
    } catch (err) {
      console.error("Prediction error:", err);
      if (err instanceof Error) {
        alert(`Prediction failed: ${err.message}`);
      } else {
        alert("Prediction failed: An unknown error occurred");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return { isLoading, hasSubmitted, predictionResult, handleSubmit };
}

async function makePrediction(inputData: { d: string; h: number; x: number; y: number }) {
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

  if (response.ok) {
    return data.ticketed;
  } else {
    throw new Error(data.error || "Prediction failed");
  }
}