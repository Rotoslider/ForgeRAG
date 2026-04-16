/** @type {import('tailwindcss').Config} */
// Matches the Choom dark palette — violet primary, blue secondary, pink accent
// against a deep purple-black background. Picked to visually pair with the
// Choom web UI so agents and the knowledge-graph GUI feel like the same system.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        forge: {
          bg: "#0f0a1a",         // rgb(15 10 26)   page background
          panel: "#1a1025",      // rgb(26 16 37)   card / panel
          edge: "#2d1f42",       // rgb(45 31 66)   borders, dividers
          muted: "#c8b4dc",      // rgb(200 180 220) de-emphasized text
          fg: "#faf5ff",         // rgb(250 245 255) primary text
          primary: "#a78bfa",    // violet-400
          secondary: "#60a5fa",  // blue-400
          accent: "#f472b6",     // pink-400
          danger: "#f87171",     // red-400
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
