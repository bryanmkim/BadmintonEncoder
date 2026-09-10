import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import BadmintonAnnotator from "./badminton-annotator.jsx";

// The app was written for Claude artifacts, which provide window.storage.
// Locally, back it with localStorage so annotations survive a reload.
if (!window.storage) {
  window.storage = {
    get: async key => {
      const value = localStorage.getItem(key);
      return value == null ? null : { key, value };
    },
    set: async (key, value) => {
      localStorage.setItem(key, value);
      return { key, value };
    },
  };
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <BadmintonAnnotator />
  </StrictMode>
);
