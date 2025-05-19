import React from 'react';
import { useNavigate } from 'react-router-dom';
import './SmallCard.css';

interface SmallCardProps {
    title: string;
    description?: string;
    imageSrc?: string;
    linkTo: string;
    children?: React.ReactNode;
}

const SmallCard: React.FC<SmallCardProps> = ({
    title,
    imageSrc,
    linkTo,
    children
}) => {
    const navigate = useNavigate();

    const handleClick = () => {
        navigate(linkTo);
    };

    return (
        <div className="card-wrapper" onClick={handleClick}>
            {imageSrc && (
                <div className="card-image-container">
                    <img src={imageSrc} alt={title} className="card-image" />
                </div>
            )}
            <div className="card-content">
                <div className="card-title">{title}</div>
                {children}
            </div>
        </div>
    );
};

export default SmallCard;
