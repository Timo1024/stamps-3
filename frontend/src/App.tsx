import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './App.css'
import NavBar from './components/NavBar'

// Import page components
import Home from './pages/Home'
import Catalogue from './pages/Catalogue'
import Collection from './pages/Collection'
import Profile from './pages/Profile'

function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <div className="page-container">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/catalogue" element={<Catalogue />} />
          <Route path="/collection" element={<Collection />} />
          <Route path="/profile" element={<Profile />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}

export default App
