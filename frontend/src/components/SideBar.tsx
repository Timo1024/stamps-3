import React, { useState, useEffect, useRef } from 'react';
import { Link, useLocation } from 'react-router-dom';
import './SideBar.css';

interface SetStats {
    countries: string[];
    categories: string[];
}

interface DateRange {
    min_date: string | null;
    max_date: string | null;
}

interface SearchParams {
    // username: string;
    show_owned: boolean;
    country: string | null;
    year_from: string | null;
    year_to: string | null;
    denomination: string | null;
    theme: string | null;
    keywords: string[] | null;
    date_of_issue: string | null;
    category: string | null;
    number_issued: string | null;
    perforation_horizontal: string | null;
    perforation_vertical: string | null;
    perforation_keyword: string | null;
    sheet_size_amount: string | null;
    sheet_size_horizontal: string | null;
    sheet_size_vertical: string | null;
    stamp_size_horizontal: string | null;
    stamp_size_vertical: string | null;
    hue: number | null;
    saturation: number | null;
    tolerance: number | null;
    max_results: string;
}

// NavBar component with active link highlighting
const SideBar: React.FC = () => {
    const [searchParams, setSearchParams] = useState<SearchParams>({
        // username: '',
        show_owned: false,
        country: null,
        year_from: null,
        year_to: null,
        denomination: null,
        theme: null,
        keywords: null,
        date_of_issue: null,
        category: null,
        number_issued: null,
        perforation_horizontal: null,
        perforation_vertical: null,
        perforation_keyword: null,
        sheet_size_amount: null,
        sheet_size_horizontal: null,
        sheet_size_vertical: null,
        stamp_size_horizontal: null,
        stamp_size_vertical: null,
        hue: null,
        saturation: null,
        tolerance: 15,
        max_results: '1000'
    });

    // countries
    const [countries, setCountries] = useState<string[]>([]);
    const [filteredCountries, setFilteredCountries] = useState<string[]>([]);
    const [showCountryDropdown, setShowCountryDropdown] = useState(false);
    const countryContainerRef = useRef<HTMLDivElement>(null);

    // categories
    const [categories, setCategories] = useState<string[]>([]);
    const [filteredCategories, setFilteredCategories] = useState<string[]>([]);
    const [showCategoryDropdown, setShowCategoryDropdown] = useState(false);
    const categoryContainerRef = useRef<HTMLDivElement>(null);

    // themes
    const [themes, setThemes] = useState<string[]>([]);
    const [filteredThemes, setFilteredThemes] = useState<string[]>([]);
    const [showThemeDropdown, setShowThemeDropdown] = useState(false);
    const themeContainerRef = useRef<HTMLDivElement>(null);

    // date from to
    const [dateRange, setDateRange] = useState<DateRange | null>(null);

    const [viewMode, setViewMode] = useState<'individual' | 'grouped'>('individual');
    const [filters, setFilters] = useState({
        country: '',
        yearFrom: '',
        yearTo: '',
        size: '',
        owned: 'all'
    });

    // State for API data
    // const [setStats, setSetStats] = useState<SetStats | null>(null);
    // const [dateRange, setDateRange] = useState<DateRange | null>(null);
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
                setCountries(setStatsData.countries || []);
                setFilteredCountries(setStatsData.countries || []);
                setCategories(setStatsData.categories || []);
                setFilteredCategories(setStatsData.categories || []);
                setThemes(themesData || []);
                setFilteredThemes(themesData || []);

                setDateRange(dateRangeData);

                setError(null);
            } catch (err) {
                console.error('Error fetching catalogue data:', err);
                setError(err instanceof Error ? err.message : 'Unknown error occurred');
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    const handleFilterChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value } = e.target;
        setFilters(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleChange = (field: keyof SearchParams, value: any) => {
        setSearchParams((prev) => ({
            ...prev,
            [field]: value,
        }));
    };

    // country

    const handleCountryChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = e.target.value;
        handleChange('country', value);

        // Filter countries based on input
        if (value) {
            const filtered = countries.filter(country =>
                country.toLowerCase().includes(value.toLowerCase())
            );
            setFilteredCountries(filtered);
            setShowCountryDropdown(true);
        } else {
            setFilteredCountries([]);
            setShowCountryDropdown(false);
        }
    };

    const selectCountry = (country: string) => {
        handleChange('country', country);
        setShowCountryDropdown(false);
    };

    // theme

    const handleThemeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = e.target.value;
        handleChange('theme', value);

        // Filter themes based on input
        if (value) {
            const filtered = themes.filter(theme =>
                theme.toLowerCase().includes(value.toLowerCase())
            );
            setFilteredThemes(filtered);
            setShowThemeDropdown(true);
        } else {
            setFilteredThemes([]);
            setShowThemeDropdown(false);
        }
    };

    const selectTheme = (theme: string) => {
        handleChange('theme', theme);
        setShowThemeDropdown(false);
    };

    return (
        <div className={`catalogue-sidebar`}>
            <div className="filter-options">
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
                            className={`sidebar-nav-item ${viewMode === 'grouped' ? 'active' : ''}`}
                            onClick={() => setViewMode('grouped')}
                        >
                            Grouped
                        </li>
                    </ul>
                </div>

                <div className="sidebar-section">
                    <div className="sidebar-heading">Filters</div>

                    {/* countries */}
                    <div className="autocomplete-container" ref={countryContainerRef}>
                        <div className="single-input">
                            <div className={`single-input-title ${searchParams.country ? 'visible' : ''}`}>Country</div>
                            <input
                                type="text"
                                placeholder="Country"
                                value={searchParams.country || ''}
                                onChange={handleCountryChange}
                                onFocus={() => setShowCountryDropdown(true)}
                                autoComplete="new-password"
                                autoCorrect="off"
                                spellCheck="false"
                            />
                        </div>
                        {showCountryDropdown && filteredCountries.length > 0 && (
                            <div className="autocomplete-dropdown">
                                {filteredCountries.map((country, index) => (
                                    <div
                                        key={index}
                                        className="autocomplete-item"
                                        onClick={() => selectCountry(country)}
                                    >
                                        {country}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* themes */}
                    <div className="autocomplete-container" ref={themeContainerRef}>
                        <div className="single-input">
                            <div className={`single-input-title ${searchParams.theme ? 'visible' : ''}`}>Theme</div>
                            <input
                                type="text"
                                placeholder="Theme"
                                value={searchParams.theme || ''}
                                onChange={handleThemeChange}
                                onFocus={() => setShowThemeDropdown(true)}
                                autoComplete="new-password"
                                autoCorrect="off"
                                spellCheck="false"
                            />
                        </div>
                        {showThemeDropdown && filteredThemes.length > 0 && (
                            <div className="autocomplete-dropdown">
                                {filteredThemes.map((theme, index) => (
                                    <div
                                        key={index}
                                        className="autocomplete-item"
                                        onClick={() => selectTheme(theme)}
                                    >
                                        {theme}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* year range */}
                    <div className="input-group">
                        <div className="single-input">
                            <div className={`single-input-title ${searchParams.year_from ? 'visible' : ''}`}>Year From</div>
                            <input
                                type="number"
                                placeholder={dateRange?.min_date?.slice(0, 4) || 'Year From'}
                                value={searchParams.year_from || ''}
                                onChange={(e) => handleChange('year_from', e.target.value)}
                                autoComplete="new-password"
                                autoCorrect="off"
                                spellCheck="false"
                                onKeyDown={(e) => {
                                    if (!/\d/.test(e.key) && !['Backspace', 'Delete', 'ArrowLeft', 'ArrowRight', 'Tab'].includes(e.key)) {
                                        e.preventDefault();
                                    }
                                }}
                            />
                        </div>
                        <div className="single-input">
                            <div className={`single-input-title ${searchParams.year_to ? 'visible' : ''}`}>Year To</div>
                            <input
                                type="number"
                                placeholder={dateRange?.max_date?.slice(0, 4) || 'Year To'}
                                value={searchParams.year_to || ''}
                                onChange={(e) => handleChange('year_to', e.target.value)}
                                autoComplete="new-password"
                                autoCorrect="off"
                                spellCheck="false"
                                onKeyDown={(e) => {
                                    if (!/\d/.test(e.key) && !['Backspace', 'Delete', 'ArrowLeft', 'ArrowRight', 'Tab'].includes(e.key)) {
                                        e.preventDefault();
                                    }
                                }}
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

                </div>
                <div className='apply-filters-btn-wrapper'>
                    <button className="apply-filters-btn">Apply</button>
                </div>
            </div>
        </div>
    );
}

export default SideBar;