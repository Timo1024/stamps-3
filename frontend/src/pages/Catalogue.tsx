import React, { useState, useEffect } from 'react';
import './Pages.css';
import './Catalogue.css';

// Define types for the API responses
interface SetStats {
    countries: string[];
    categories: string[];
}

interface DateRange {
    min_date: string | null;
    max_date: string | null;
}

interface ThemesResponse {
    themes: string[];
}

const Catalogue: React.FC = () => {
    const [viewMode, setViewMode] = useState<'individual' | 'country' | 'year'>('individual');
    const [filters, setFilters] = useState({
        country: '',
        yearFrom: '',
        yearTo: '',
        size: '',
        owned: 'all'
    });

    // State for API data
    const [setStats, setSetStats] = useState<SetStats | null>(null);
    const [dateRange, setDateRange] = useState<DateRange | null>(null);
    const [themes, setThemes] = useState<string[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Fetch data from backend
    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            try {
                const apiUrl = `http://${window.location.hostname}:8000`;

                // Fetch set statistics
                const setStatsResponse = await fetch(`${apiUrl}/sets/overall_stats`);
                if (!setStatsResponse.ok) throw new Error('Failed to fetch set stats');
                const setStatsData = await setStatsResponse.json();

                // Fetch date range
                const dateRangeResponse = await fetch(`${apiUrl}/stamps/date_range`);
                if (!dateRangeResponse.ok) throw new Error('Failed to fetch date range');
                const dateRangeData = await dateRangeResponse.json();

                // Fetch themes
                const themesResponse = await fetch(`${apiUrl}/themes/unique`);
                if (!themesResponse.ok) throw new Error('Failed to fetch themes');
                const themesData = await themesResponse.json();

                // Update state with fetched data
                setSetStats(setStatsData);
                setDateRange(dateRangeData);
                setThemes(themesData || []);

                setError(null);
            } catch (err) {
                console.error('Error fetching catalogue data:', err);
                setError(err instanceof Error ? err.message : 'Unknown error occurred');
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []); // Empty dependency array means this runs once when component mounts

    const handleFilterChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value } = e.target;
        setFilters(prev => ({
            ...prev,
            [name]: value
        }));
    };

    return (
        <div className="page catalogue-container">
            <div className='catalogue-heading'>Stamps Catalogue</div>
            <div className="catalogue-content">
                <div className={`catalogue-sidebar`}>
                    <div className="sidebar-section">
                        <div className="sidebar-heading">View Options</div>
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
                        <div className="sidebar-heading">Filters</div>
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

                <div className={`catalogue-content`}>
                    {/* Content area that will change based on the view mode and filters */}
                    <div className="catalogue-results">
                        <p>Showing {viewMode} view with selected filters.</p>

                        {/* show meta results from backend */}
                        <div className="backend-metadata">
                            <h3>Catalogue Metadata</h3>

                            {loading ? (
                                <p>Loading metadata...</p>
                            ) : error ? (
                                <p className="error">Error: {error}</p>
                            ) : (
                                <div className="metadata-grid">
                                    <div className="metadata-section">
                                        <h4>Set Statistics</h4>
                                        {setStats && (
                                            <>
                                                <p>Countries: {setStats.countries.length}</p>
                                                <p>Categories: {setStats.categories.length}</p>
                                                <p>Sample Countries: {setStats.countries.slice(0, 3).join(', ')}{setStats.countries.length > 3 ? '...' : ''}</p>
                                            </>
                                        )}
                                    </div>

                                    <div className="metadata-section">
                                        <h4>Date Range</h4>
                                        {dateRange && (
                                            <p>
                                                {dateRange.min_date && dateRange.max_date ?
                                                    `${new Date(dateRange.min_date).getFullYear()} - ${new Date(dateRange.max_date).getFullYear()}` :
                                                    'No date range available'}
                                            </p>
                                        )}
                                    </div>

                                    <div className="metadata-section">
                                        <h4>Themes</h4>
                                        <p>{themes.length} unique themes</p>
                                        <p>Sample: {themes.slice(0, 3).join(', ')}{themes.length > 3 ? '...' : ''}</p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Catalogue;
