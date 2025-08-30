import { useEffect, useRef, useState } from "react";

interface UseGoogleMapsAutocompleteProps {
  onPlaceSelected: (location: google.maps.LatLng, isDC: boolean) => void;
}

export function useGoogleMapsAutocomplete({ onPlaceSelected }: UseGoogleMapsAutocompleteProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const apiKey = process.env.NEXT_PUBLIC_GOOGLE_API_KEY;
    if (!apiKey || !inputRef.current) return;

    const options = {
      componentRestrictions: { country: "us" },
      fields: ["address_components", "geometry"],
    };

    if (typeof window !== "undefined" && window.google?.maps) {
      const autocomplete = new window.google.maps.places.Autocomplete(inputRef.current!, options);
      autocomplete.addListener("place_changed", () => {
        const place = autocomplete.getPlace();
        if (!place || !place.geometry || !place.geometry.location) return;

        const location = place.geometry.location;
        const addressComponents = place.address_components;
        const isDC = addressComponents?.some((component) =>
          component.short_name === "DC" || component.long_name === "District of Columbia"
        );

        onPlaceSelected(location, isDC ?? false);
      });

      return () => window.google.maps.event.clearInstanceListeners(autocomplete);
    }
  }, [onPlaceSelected]);

  return { inputRef, error, setError };
}