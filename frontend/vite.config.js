import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: '/DeepFake-Detection-System-using-CNNs-and-Explainable-AI/',
  plugins: [react()],
  server: {
    port: 3000,
    host: '0.0.0.0'
  }
})