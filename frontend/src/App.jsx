import { useState } from 'react'
import './App.css'

function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleImageChange = (e) => {
    console.log('File input changed:', e.target.files)
    const file = e.target.files[0]
    if (file) {
      console.log('File selected:', file.name, file.type, file.size)
      setImage(file)
      const reader = new FileReader()
      reader.onload = (event) => {
        console.log('Preview loaded')
        setPreview(event.target.result)
      }
      reader.onerror = (error) => {
        console.error('FileReader error:', error)
        setError('Error loading image preview')
      }
      reader.readAsDataURL(file)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    console.log('Form submitted, image:', image)
    if (!image) {
      setError('Please select an image first')
      return
    }

    setLoading(true)
    setError(null)
    const formData = new FormData()
    formData.append('image', image)

    console.log('Sending request to backend...')
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
        headers: {
          'Accept': 'application/json'
        }
      })
      console.log('Response status:', response.status)
      
      const data = await response.json()
      console.log('Response data:', data)
      
      if (response.ok && (data.status === 'success' || data.prediction)) {
        setResult(data)
        setError(null)
      } else {
        setError(data.error || `Server error: ${response.status}`)
      }
    } catch (error) {
      console.error('Fetch error:', error)
      setError(`Failed to connect to backend: ${error.message}. Make sure the server is running on http://localhost:5000`)
    }
    setLoading(false)
  }

  return (
    <div className="container">
      <header>
        <h1>🔍 Deepfake Detection System</h1>
        <p>Hybrid AI model using CNN, FFT, and Transformers</p>
      </header>
      <main>
        <form onSubmit={handleSubmit} className="upload-form">
          <div className="file-input-wrapper">
            <input 
              id="image-input"
              type="file" 
              accept="image/*" 
              onChange={handleImageChange}
              disabled={loading}
            />
            <label htmlFor="image-input">
              {image ? `📁 ${image.name}` : 'Click or drag image to upload'}
            </label>
          </div>
          <button type="submit" disabled={loading || !image}>
            {loading ? 'Processing...' : 'Detect Deepfake'}
          </button>
        </form>

        {preview && (
          <div className="preview">
            <img src={preview} alt="Preview" />
            <p>{image.name}</p>
          </div>
        )}

        {loading && <div className="spinner">Analyzing image...</div>}

        {error && (
          <div className="error">
            ❌ Error: {error}
          </div>
        )}

        {result && (
          <div className={`result ${result.prediction.toLowerCase()}`}>
            <div className="result-header">
              <h2>Analysis Result</h2>
            </div>
            <div className="result-content">
              <div className="prediction">
                <span className="label">Prediction:</span>
                <span className={`value ${result.prediction.toLowerCase()}`}>
                  {result.prediction === 'Fake' ? '⚠️ FAKE' : '✓ REAL'}
                </span>
              </div>
              <div className="confidence">
                <span className="label">Confidence:</span>
                <span className="value">{(result.confidence * 100).toFixed(2)}%</span>
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill" 
                    style={{ width: `${result.confidence * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
      <footer>
        <p>Patent-level deepfake detection powered by hybrid neural networks</p>
      </footer>
    </div>
  )
}

export default App