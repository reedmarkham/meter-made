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
  // POST to our own Next.js server route which will proxy the request to the
  // model API. This avoids exposing the model API URL to the browser and
  // allows the server-side code (running in the same project) to reach an
  // "internal" Cloud Run service.
  const response = await fetch(`/api/predict`, {
    method: "POST",
    headers: {
      accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(inputData),
  });

  const contentType = response.headers.get("content-type") || "";
  const bodyText = await response.text();

  type PredictionResponse = {
    ticketed?: number;
    error?: string;
  };

  let data: PredictionResponse | undefined;
  if (contentType.includes("application/json")) {
    try {
      data = JSON.parse(bodyText) as PredictionResponse;
    } catch {
      throw new Error(`Prediction failed: invalid JSON response from server`);
    }
  } else {
    throw new Error(`Prediction failed: expected JSON but got ${response.status} ${response.statusText}. Response body: ${bodyText.slice(0, 200)}`);
  }

  if (response.ok) {
    return data?.ticketed ?? 0;
  } else {
    throw new Error(data?.error || `Prediction failed: ${response.status} ${response.statusText}`);
  }
}