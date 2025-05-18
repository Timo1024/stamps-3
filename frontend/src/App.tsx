import { useState, useEffect } from 'react'
// import reactLogo from './assets/react.svg'
// import viteLogo from '/vite.svg'
import './App.css'
import NavBar from './components/NavBar' // Import the NavBar component

function App() {
  // const [count, setCount] = useState(0)
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchWelcomeMessage = async () => {
      try {
        // Use window.location.hostname to get the current host dynamically
        // This will work in both development and Docker environments
        const backendUrl = `http://${window.location.hostname}:8000/`;
        console.log('Connecting to backend at:', backendUrl);

        const response = await fetch(backendUrl)
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`)
        }
        const data = await response.json()
        setMessage(data.message || 'Welcome!')
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An unknown error occurred')
        console.error('Error fetching welcome message:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchWelcomeMessage()
  }, [])

  return (
    <>
      <NavBar />
      {/* <div className="app-content">
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div> */}
      {/* <h1>Vite + React</h1> */}
      {/* <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.tsx</code> and save to test HMR
        </p>
      </div> */}

      <div className="card">
        {loading ? (
          <p>Loading message from backend...</p>
        ) : error ? (
          <p>Error connecting to backend: {error}</p>
        ) : (
          <p>Backend says: {message}</p>
        )}
      </div>

      {/* <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p> */}
    </>
  )
}

export default App
