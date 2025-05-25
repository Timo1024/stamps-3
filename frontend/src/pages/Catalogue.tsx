import React, { useState } from 'react';
import './Pages.css';
import './Catalogue.css';

const Catalogue: React.FC = () => {
    const [viewMode, setViewMode] = useState<'individual' | 'country' | 'year'>('individual');
    const [sidebarVisible, setSidebarVisible] = useState(true);
    const [filters, setFilters] = useState({
        country: '',
        yearFrom: '',
        yearTo: '',
        size: '',
        owned: 'all'
    });

    const handleFilterChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value } = e.target;
        setFilters(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const toggleSidebar = () => {
        setSidebarVisible(!sidebarVisible);
    };

    return (
        <div className="page catalogue-container">
            <div className='catalogue-heading'>Stamps Catalogue</div>
            <div className="catalogue-content">
                <div className={`catalogue-sidebar ${sidebarVisible ? 'visible' : 'hidden'}`}>
                    <div className="sidebar-section">
                        <h3 className="sidebar-heading">View Options</h3>
                        <ul className="sidebar-nav">
                            <li
                                className={`sidebar-nav-item ${viewMode === 'individual' ? 'active' : ''}`}
                                onClick={() => setViewMode('individual')}
                            >
                                Individual Stamps
                            </li>
                            <li
                                className={`sidebar-nav-item ${viewMode === 'country' ? 'active' : ''}`}
                                onClick={() => setViewMode('country')}
                            >
                                By Country
                            </li>
                            <li
                                className={`sidebar-nav-item ${viewMode === 'year' ? 'active' : ''}`}
                                onClick={() => setViewMode('year')}
                            >
                                By Year
                            </li>
                        </ul>
                    </div>

                    <div className="sidebar-section">
                        <h3 className="sidebar-heading">Filters</h3>
                        <div className="filter-group">
                            <label htmlFor="country">Country</label>
                            <select
                                id="country"
                                name="country"
                                value={filters.country}
                                onChange={handleFilterChange}
                            >
                                <option value="">All Countries</option>
                                <option value="usa">United States</option>
                                <option value="uk">United Kingdom</option>
                                <option value="canada">Canada</option>
                                <option value="australia">Australia</option>
                            </select>
                        </div>

                        <div className="filter-group year-range">
                            <label>Year Range</label>
                            <div className="year-inputs">
                                <input
                                    type="number"
                                    name="yearFrom"
                                    placeholder="From"
                                    value={filters.yearFrom}
                                    onChange={handleFilterChange}
                                    min="1840"
                                    max="2023"
                                />
                                <span>-</span>
                                <input
                                    type="number"
                                    name="yearTo"
                                    placeholder="To"
                                    value={filters.yearTo}
                                    onChange={handleFilterChange}
                                    min="1840"
                                    max="2023"
                                />
                            </div>
                        </div>

                        <div className="filter-group">
                            <label htmlFor="size">Size</label>
                            <select
                                id="size"
                                name="size"
                                value={filters.size}
                                onChange={handleFilterChange}
                            >
                                <option value="">All Sizes</option>
                                <option value="small">Small</option>
                                <option value="medium">Medium</option>
                                <option value="large">Large</option>
                            </select>
                        </div>

                        <div className="filter-group">
                            <label htmlFor="owned">Ownership</label>
                            <select
                                id="owned"
                                name="owned"
                                value={filters.owned}
                                onChange={handleFilterChange}
                            >
                                <option value="all">All Stamps</option>
                                <option value="owned">Owned</option>
                                <option value="not-owned">Not Owned</option>
                                <option value="wishlist">Wishlist</option>
                            </select>
                        </div>

                        <button className="apply-filters-btn">Apply Filters</button>
                    </div>
                </div>

                <button
                    className={`sidebar-toggle ${sidebarVisible ? '' : 'shifted'}`}
                    onClick={toggleSidebar}
                    aria-label={sidebarVisible ? "Hide sidebar" : "Show sidebar"}
                >
                    {sidebarVisible ? '←' : '→'}
                </button>

                <div className={`catalogue-content ${sidebarVisible ? '' : 'expanded'}`}>

                    {/* Content area that will change based on the view mode and filters */}
                    <div className="catalogue-results">
                        <p>Showing {viewMode} view with selected filters.</p>
                        {/* Stamps will be displayed here */}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Catalogue;
