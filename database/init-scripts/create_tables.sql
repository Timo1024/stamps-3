CREATE TABLE IF NOT EXISTS sets (
    setid INTEGER PRIMARY KEY,
    country VARCHAR(255),
    category VARCHAR(255),
    year INT,
    url VARCHAR(2083),
    name VARCHAR(2083),
    set_description TEXT
);

CREATE TABLE IF NOT EXISTS stamps (
    stampid INTEGER AUTO_INCREMENT PRIMARY KEY,
    number VARCHAR(255),
    type VARCHAR(255),
    denomination VARCHAR(255),
    color VARCHAR(255),
    description TEXT,
    stamps_issued VARCHAR(255),
    mint_condition VARCHAR(255),
    unused VARCHAR(255),
    used VARCHAR(255),
    letter_fdc VARCHAR(255),
    date_of_issue DATE,
    perforations VARCHAR(255),
    sheet_size VARCHAR(255),
    designed VARCHAR(255),
    engraved VARCHAR(255),
    height_width VARCHAR(255),
    image_accuracy INT,
    perforation_horizontal FLOAT,
    perforation_vertical FLOAT,
    perforation_keyword VARCHAR(2083),
    value_from FLOAT,
    value_to FLOAT,
    number_issued BIGINT,
    mint_condition_float FLOAT,
    unused_float FLOAT,
    used_float FLOAT,
    letter_fdc_float FLOAT,
    sheet_size_amount FLOAT,
    sheet_size_x FLOAT,
    sheet_size_y FLOAT,
    sheet_size_note VARCHAR(2083),
    height FLOAT,
    width FLOAT,
    setid INT,
    FOREIGN KEY (setid) REFERENCES sets(setid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS users (
    userid INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_stamps (
    userid INT NOT NULL,
    stampid INT NOT NULL,
    amount_used INT DEFAULT 0,
    amount_unused INT DEFAULT 0,
    amount_minted INT DEFAULT 0,
    amount_letter_fdc INT DEFAULT 0,
    note TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (userid, stampid),
    FOREIGN KEY (userid) REFERENCES users(userid) ON DELETE CASCADE,
    FOREIGN KEY (stampid) REFERENCES stamps(stampid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS themes (
    themeid INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS stamp_themes (
    stampid INT NOT NULL,
    themeid INT NOT NULL,
    PRIMARY KEY (stampid, themeid),
    FOREIGN KEY (stampid) REFERENCES stamps(stampid) ON DELETE CASCADE,
    FOREIGN KEY (themeid) REFERENCES themes(themeid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS images (
    pathid INT AUTO_INCREMENT PRIMARY KEY,
    path VARCHAR(2083) NOT NULL
);

CREATE TABLE IF NOT EXISTS stamp_images (
    stampid INT NOT NULL,
    pathid INT NOT NULL,
    PRIMARY KEY (stampid, pathid),
    FOREIGN KEY (stampid) REFERENCES stamps(stampid) ON DELETE CASCADE,
    FOREIGN KEY (pathid) REFERENCES images(pathid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS colors (
    colorid INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS stamp_colors (
    stampid INT NOT NULL,
    colorid INT NOT NULL,
    PRIMARY KEY (stampid, colorid),
    FOREIGN KEY (stampid) REFERENCES stamps(stampid) ON DELETE CASCADE,
    FOREIGN KEY (colorid) REFERENCES colors(colorid) ON DELETE CASCADE
);

-- Add indexes to the `stamps` table
CREATE INDEX idx_stamps_setid ON stamps(setid);

-- Add indexes to the `sets` table

-- Add indexes to the `users` table

-- Add indexes to the `user_stamps` table
CREATE INDEX idx_user_stamps_userid ON user_stamps(userid);
CREATE INDEX idx_user_stamps_stampid ON user_stamps(stampid);

-- Add indexes to the `themes` table

-- Add indexes to the `stamp_themes` table
CREATE INDEX idx_stamp_themes_stampid ON stamp_themes(stampid);
CREATE INDEX idx_stamp_themes_themeid ON stamp_themes(themeid);

-- Add indexes to the `images` table

-- Add indexes to the `stamp_images` table
CREATE INDEX idx_stamp_images_stampid ON stamp_images(stampid);
CREATE INDEX idx_stamp_images_pathid ON stamp_images(pathid);

-- Add indexes to the `colors` table

-- Add indexes to the `stamp_colors` table
CREATE INDEX idx_stamp_colors_stampid ON stamp_colors(stampid);
CREATE INDEX idx_stamp_colors_colorid ON stamp_colors(colorid);