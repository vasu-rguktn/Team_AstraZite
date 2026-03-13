/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",      // for plain HTML
    "./src/**/*.{js,ts,jsx,tsx}", // for React/Vue/Next.js
  ],
  theme: {
    extend: { fontFamily: {
        inter: ["Inter", "sans-serif"],
        kanit: ['Kanit', 'sans-serif'],

      },},
  },
  plugins: [],
}
