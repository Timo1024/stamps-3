import React, { useState, useEffect } from 'react';
import './Pages.css';
import './Catalogue.css';

import SideBar from '../components/SideBar';

// Define types for the API responses
// interface SetStats {
//     countries: string[];
//     categories: string[];
// }

// interface DateRange {
//     min_date: string | null;
//     max_date: string | null;
// }

// interface ThemesResponse {
//     themes: string[];
// }

const Catalogue: React.FC = () => {
    // State for API data
    // const [setStats, setSetStats] = useState<SetStats | null>(null);
    // const [dateRange, setDateRange] = useState<DateRange | null>(null);
    // const [themes, setThemes] = useState<string[]>([]);
    // const [loading, setLoading] = useState(true);
    // const [error, setError] = useState<string | null>(null);

    return (
        <div className="page catalogue-container">
            <div className='catalogue-heading'>Stamps Catalogue</div>
            <div className="catalogue-content">

                <SideBar />

                <div className={`catalogue-content`}>
                    {/* Content area that will change based on the view mode and filters */}
                    <div className="catalogue-results">
                        <p>Showing view with selected filters.</p>

                        {/* show meta results from backend */}
                        <div className="backend-metadata">
                            <h3>Catalogue Metadata</h3>

                            {/* {loading ? (
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
                            )} */}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Catalogue;
