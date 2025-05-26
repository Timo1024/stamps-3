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
    date_from: string | null;
    date_to: string | null;
    denomination: string | null;
    theme: string | null;
    keywords: string[] | null;
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
        date_from: null,
        date_to: null,
        denomination: null,
        theme: null,
        keywords: null,
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

    // keywords
    const [keywordsText, setKeywordsText] = useState('');

    // date from to
    const [dateRange, setDateRange] = useState<DateRange | null>(null);

    const [viewMode, setViewMode] = useState<'individual' | 'grouped'>('individual');

    // State for API data
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

                // setError(null);
            } catch (err) {
                console.error('Error fetching catalogue data:', err);
                setError(err instanceof Error ? err.message : 'Unknown error occurred');
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    // log when searchParams change
    useEffect(() => {
        console.log('Search parameters updated:', searchParams);
    }, [searchParams]);

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

    // category

    const handleCategoryChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = e.target.value;
        handleChange('category', value);

        // Filter categories based on input
        if (value) {
            const filtered = categories.filter(category =>
                category.toLowerCase().includes(value.toLowerCase())
            );
            setFilteredCategories(filtered);
            setShowCategoryDropdown(true);
        } else {
            setFilteredCategories([]);
            setShowCategoryDropdown(false);
        }
    };

    const selectCategory = (category: string) => {
        handleChange('category', category);
        setShowCategoryDropdown(false);
    };

    // keyword

    const handleKeywordsChange = (value: string) => {
        setKeywordsText(value);
        // Only split into keywords when submitting or when there's actual content
        const keywords = value.trim() ? value.split(',').map(k => k.trim()).filter(k => k) : [];
        handleChange('keywords', keywords);
    };

    // Format date for input element (YYYY-MM-DD)
    const formatDateForInput = (dateString: string | null): string => {
        if (!dateString) return '';
        // Ensure we have a valid date string
        try {
            const date = new Date(dateString);
            if (isNaN(date.getTime())) return '';

            return date.toISOString().split('T')[0]; // Returns YYYY-MM-DD format
        } catch (e) {
            console.error('Invalid date:', dateString);
            return '';
        }
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

                    {/* show_owned toggle */}
                    <div className="single-input">
                        <label className="checkbox-label">
                            <input
                                type="checkbox"
                                checked={searchParams.show_owned}
                                onChange={(e) => handleChange('show_owned', e.target.checked)}
                            />
                            Show Owned Stamps
                        </label>
                    </div>

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

                    {/* category */}
                    <div className="autocomplete-container" ref={categoryContainerRef}>
                        <div className="single-input">
                            <div className={`single-input-title ${searchParams.category ? 'visible' : ''}`}>Category</div>
                            <input
                                type="text"
                                placeholder="Category"
                                value={searchParams.category || ''}
                                onChange={handleCategoryChange}
                                onFocus={() => setShowCategoryDropdown(true)}
                                autoComplete="new-password"
                                autoCorrect="off"
                                spellCheck="false"
                            />
                        </div>
                        {showCategoryDropdown && filteredCategories.length > 0 && (
                            <div className="autocomplete-dropdown">
                                {filteredCategories.map((category, index) => (
                                    <div
                                        key={index}
                                        className="autocomplete-item"
                                        onClick={() => selectCategory(category)}
                                    >
                                        {category}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* year range - updated to use date inputs */}
                    <div className="input-group">
                        <div className="single-input">
                            <div className={`single-input-title ${searchParams.date_from ? 'visible' : ''}`}>Date From</div>
                            <input
                                type="date"
                                placeholder="Date From"
                                value={searchParams.date_from || ''}
                                min={dateRange?.min_date ? formatDateForInput(dateRange.min_date) : ''}
                                max={dateRange?.max_date ? formatDateForInput(dateRange.max_date) : ''}
                                onChange={(e) => handleChange('date_from', e.target.value)}
                                autoComplete="off"
                            />
                        </div>
                        <div className="single-input">
                            <div className={`single-input-title ${searchParams.date_to ? 'visible' : ''}`}>Date To</div>
                            <input
                                type="date"
                                placeholder="Date To"
                                value={searchParams.date_to || ''}
                                min={dateRange?.min_date ? formatDateForInput(dateRange.min_date) : ''}
                                max={dateRange?.max_date ? formatDateForInput(dateRange.max_date) : ''}
                                onChange={(e) => handleChange('date_to', e.target.value)}
                                autoComplete="off"
                            />
                        </div>
                    </div>

                    {/* denomination */}
                    <div className="single-input">
                        <div className={`single-input-title ${searchParams.denomination ? 'visible' : ''}`}>Denomination</div>
                        <input
                            type="number"
                            placeholder="Denomination"
                            value={searchParams.denomination || ''}
                            onChange={(e) => handleChange('denomination', e.target.value)}
                            autoComplete="new-password"
                            autoCorrect="off"
                            spellCheck="false"
                            onKeyDown={(e) => {
                                if (!/\d/.test(e.key) && !['Backspace', 'Delete', 'ArrowLeft', 'ArrowRight', 'Tab', ".", ","].includes(e.key)) {
                                    e.preventDefault();
                                }
                            }}
                        />
                    </div>

                    <div className="single-input">
                        <div className={`single-input-title ${searchParams.keywords ? 'visible' : ''}`}>Keywords (comma-separated)</div>
                        <textarea
                            placeholder="Keywords (comma-separated)"
                            value={keywordsText}
                            onChange={(e) => {
                                const value = e.target.value;
                                handleKeywordsChange(value);
                                // Auto-adjust height only if content exceeds initial height
                                if (e.target.scrollHeight > 57) {
                                    e.target.style.height = 'auto';
                                    e.target.style.height = `${e.target.scrollHeight + 2}px`;
                                } else {
                                    e.target.style.height = '57px';
                                }
                            }}
                            autoComplete="new-password"
                            autoCorrect="off"
                            spellCheck="false"
                            rows={1}
                            className="keywords-textarea"
                        />
                    </div>

                    {/* number_issued */}
                    <div className="single-input">
                        <div className={`single-input-title ${searchParams.number_issued ? 'visible' : ''}`}>Number Issued</div>
                        <input
                            type="number"
                            placeholder="Number Issued"
                            value={searchParams.number_issued || ''}
                            onChange={(e) => handleChange('number_issued', e.target.value)}
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

                    {/* perforation_horizontal */}
                    <div className="single-input">
                        <div className={`single-input-title ${searchParams.perforation_horizontal ? 'visible' : ''}`}>Perforation Horizontal</div>
                        <input
                            type="number"
                            placeholder="Perforation Horizontal"
                            value={searchParams.perforation_horizontal || ''}
                            onChange={(e) => handleChange('perforation_horizontal', e.target.value)}
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

                    {/* perforation_vertical */}
                    <div className="single-input">
                        <div className={`single-input-title ${searchParams.perforation_vertical ? 'visible' : ''}`}>Perforation Vertical</div>
                        <input
                            type="number"
                            placeholder="Perforation Vertical"
                            value={searchParams.perforation_vertical || ''}
                            onChange={(e) => handleChange('perforation_vertical', e.target.value)}
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

                    {/* sheet_size_amount */}
                    <div className="single-input">
                        <div className={`single-input-title ${searchParams.sheet_size_amount ? 'visible' : ''}`}>Sheet Size Amount</div>
                        <input
                            type="number"
                            placeholder="Sheet Size Amount"
                            value={searchParams.sheet_size_amount || ''}
                            onChange={(e) => handleChange('sheet_size_amount', e.target.value)}
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

                    {/* sheet_size_horizontal */}
                    <div className="single-input">
                        <div className={`single-input-title ${searchParams.sheet_size_horizontal ? 'visible' : ''}`}>Sheet Size Horizontal</div>
                        <input
                            type="number"
                            placeholder="Sheet Size Horizontal"
                            value={searchParams.sheet_size_horizontal || ''}
                            onChange={(e) => handleChange('sheet_size_horizontal', e.target.value)}
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

                    {/* sheet_size_vertical */}
                    <div className="single-input">
                        <div className={`single-input-title ${searchParams.sheet_size_vertical ? 'visible' : ''}`}>Sheet Size Vertical</div>
                        <input
                            type="number"
                            placeholder="Sheet Size Vertical"
                            value={searchParams.sheet_size_vertical || ''}
                            onChange={(e) => handleChange('sheet_size_vertical', e.target.value)}
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

                    {/* stamp_size_horizontal */}
                    <div className="single-input">
                        <div className={`single-input-title ${searchParams.stamp_size_horizontal ? 'visible' : ''}`}>Stamp Size Horizontal</div>
                        <input
                            type="number"
                            placeholder="Stamp Size Horizontal"
                            value={searchParams.stamp_size_horizontal || ''}
                            onChange={(e) => handleChange('stamp_size_horizontal', e.target.value)}
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

                    {/* stamp_size_vertical */}
                    <div className="single-input">
                        <div className={`single-input-title ${searchParams.stamp_size_vertical ? 'visible' : ''}`}>Stamp Size Vertical</div>
                        <input
                            type="number"
                            placeholder="Stamp Size Vertical"
                            value={searchParams.stamp_size_vertical || ''}
                            onChange={(e) => handleChange('stamp_size_vertical', e.target.value)}
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

                    {/* color stuff */}

                    {/* hue */}
                    <div className="single-input">
                        <div className={`single-input-title ${searchParams.hue ? 'visible' : ''}`}>Hue</div>
                        <input
                            type="number"
                            placeholder="Hue (0-360)"
                            value={searchParams.hue || ''}
                            onChange={(e) => handleChange('hue', e.target.value)}
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
                    {/* saturation */}
                    <div className="single-input">
                        <div className={`single-input-title ${searchParams.saturation ? 'visible' : ''}`}>Saturation</div>
                        <input
                            type="number"
                            placeholder="Saturation (0-100)"
                            value={searchParams.saturation || ''}
                            onChange={(e) => handleChange('saturation', e.target.value)}
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
                    {/* tolerance */}
                    <div className="single-input">
                        <div className={`single-input-title ${searchParams.tolerance ? 'visible' : ''}`}>Tolerance</div>
                        <input
                            type="number"
                            placeholder="Tolerance (0-100)"
                            value={searchParams.tolerance || ''}
                            onChange={(e) => handleChange('tolerance', e.target.value)}
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
                <div className='apply-filters-btn-wrapper'>
                    <button className="apply-filters-btn">Apply</button>
                </div>
            </div>
        </div>
    );
}

export default SideBar;