import React from 'react';
import { useState, useEffect } from 'react'
import './Home.css'
import './Pages.css'

const Home: React.FC = () => {
    const [message, setMessage] = useState('')
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        const fetchWelcomeMessage = async () => {
            try {
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

        <div className="page">
            <div className='home'>
                <div className='title'>Welcome back!</div>



                {/* Backend status message */}
                <div className="card">
                    {loading ? (
                        <p>Loading message from backend...</p>
                    ) : error ? (
                        <p>Error connecting to backend: {error}</p>
                    ) : (
                        <p>Backend says: {message}</p>
                    )}
                </div>
            </div>
        </div>
    );
};

export default Home;
