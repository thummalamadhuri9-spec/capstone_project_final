import sqlite3
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50

BASE_DIR = Path(__file__).resolve().parent

CSV_FILE = BASE_DIR / "scraped_books.csv"
DB_FILE = BASE_DIR / "books.db"
SQL_OUTPUT_FILE = BASE_DIR / "sql_outputs.txt"


RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


# ============================================================
# HTTP REQUEST
# ============================================================

def get_soup(url):

    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 Zepto-Capstone"
        },
        timeout=30
    )

    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


# ============================================================
# DISCOVER CATEGORIES
# ============================================================

def get_categories():

    soup = get_soup(BASE_URL)

    categories = {}

    links = soup.select(
        "div.side_categories ul li ul li a"
    )

    for link in links:

        category_name = link.get_text(strip=True)

        href = link.get("href")

        if href:

            category_url = requests.compat.urljoin(
                BASE_URL,
                href
            )

            categories[category_name] = category_url

    return categories


# ============================================================
# SCRAPE ONE BOOK
# ============================================================

def scrape_book(book_url, category):

    soup = get_soup(book_url)

    title = soup.select_one(
        "div.product_main h1"
    )

    price = soup.select_one(
        "div.product_main p.price_color"
    )

    rating = soup.select_one(
        "div.product_main p.star-rating"
    )

    availability = soup.select_one(
        "div.product_main p.instock"
    )

    rating_text = None

    if rating:

        classes = rating.get("class", [])

        if len(classes) >= 2:
            rating_text = classes[-1]

    return {

        "title":
            title.get_text(strip=True)
            if title else None,

        "price":
            price.get_text(strip=True)
            if price else None,

        "star_rating":
            rating_text,

        "availability":
            availability.get_text(" ", strip=True)
            if availability else None,

        "category":
            category
    }


# ============================================================
# SCRAPE CATEGORY
# ============================================================

def scrape_category(category, category_url):

    books = []

    current_url = category_url

    while current_url:

        soup = get_soup(current_url)

        book_cards = soup.select(
            "article.product_pod"
        )

        for card in book_cards:

            link = card.select_one("h3 a")

            if not link:
                continue

            book_url = requests.compat.urljoin(
                current_url,
                link.get("href")
            )

            try:

                book = scrape_book(
                    book_url,
                    category
                )

                books.append(book)

            except requests.RequestException as error:

                print(
                    f"Skipping book because of error: {error}"
                )

        next_button = soup.select_one(
            "li.next a"
        )

        if next_button:

            current_url = requests.compat.urljoin(
                current_url,
                next_button.get("href")
            )

        else:

            current_url = None

    return books


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):

    df = df.copy()

    # -------------------------
    # PRICE
    # -------------------------

    df["price_gbp"] = (
        df["price"]
        .astype("string")
        .str.replace("£", "", regex=False)
        .str.strip()
    )

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    # -------------------------
    # STAR RATING
    # -------------------------

    df["rating"] = df["star_rating"].map(
        RATING_MAP
    )

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    # -------------------------
    # AVAILABILITY
    # -------------------------

    df["in_stock"] = (
        df["availability"]
        .astype("string")
        .str.contains(
            "In stock",
            case=False,
            na=False
        )
    )

    # -------------------------
    # NUMERIC IMPUTATION
    # -------------------------

    for column in ["price_gbp", "rating"]:

        if df[column].isna().any():

            median_value = df[column].median()

            df[column] = df[column].fillna(
                median_value
            )

    # -------------------------
    # REQUIRED KEY FIELDS
    # -------------------------

    df = df.dropna(
        subset=["title", "category"]
    )

    # -------------------------
    # TYPES
    # -------------------------

    df["rating"] = (
        df["rating"]
        .round()
        .clip(1, 5)
        .astype(int)
    )

    df["price_gbp"] = (
        df["price_gbp"]
        .astype(float)
    )

    df["in_stock"] = (
        df["in_stock"]
        .astype(bool)
    )

    # -------------------------
    # FIXED PROJECT RATE
    # -------------------------

    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    return df[
        [
            "title",
            "price_gbp",
            "rating",
            "in_stock",
            "price_inr",
            "category"
        ]
    ]


# ============================================================
# CREATE DATABASE
# ============================================================

def create_database(df):

    if DB_FILE.exists():
        DB_FILE.unlink()

    connection = sqlite3.connect(DB_FILE)

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE categories (

            category_id INTEGER PRIMARY KEY AUTOINCREMENT,

            category_name TEXT NOT NULL UNIQUE

        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE books (

            book_id INTEGER PRIMARY KEY AUTOINCREMENT,

            title TEXT NOT NULL,

            price_gbp REAL NOT NULL,

            price_inr REAL NOT NULL,

            rating INTEGER NOT NULL,

            in_stock INTEGER NOT NULL,

            category_id INTEGER NOT NULL,

            FOREIGN KEY(category_id)
                REFERENCES categories(category_id)

        )
        """
    )

    # Insert categories

    categories = sorted(
        df["category"].unique()
    )

    for category in categories:

        cursor.execute(
            """
            INSERT INTO categories(category_name)
            VALUES(?)
            """,
            (category,)
        )

    # Category lookup

    category_lookup = {}

    rows = cursor.execute(
        """
        SELECT category_id, category_name
        FROM categories
        """
    ).fetchall()

    for category_id, category_name in rows:

        category_lookup[
            category_name
        ] = category_id

    # Insert books

    for _, row in df.iterrows():

        cursor.execute(
            """
            INSERT INTO books
            (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )

            VALUES (?, ?, ?, ?, ?, ?)
            """,

            (
                row["title"],
                row["price_gbp"],
                row["price_inr"],
                row["rating"],
                int(row["in_stock"]),
                category_lookup[
                    row["category"]
                ]
            )
        )

    connection.commit()

    connection.close()


# ============================================================
# SQL QUERIES
# ============================================================

def execute_queries():

    connection = sqlite3.connect(
        DB_FILE
    )

    queries = {

        "Q1_SELECT_WHERE":

        """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp > 30;
        """,

        "Q2_ORDER_BY_LIMIT":

        """
        SELECT title, rating, price_inr
        FROM books
        ORDER BY rating DESC, price_inr DESC
        LIMIT 10;
        """,

        "Q3_DISTINCT":

        """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
        """,

        "Q4_BETWEEN":

        """
        SELECT title, price_gbp, rating
        FROM books
        WHERE price_gbp BETWEEN 10 AND 25
        ORDER BY price_gbp;
        """,

        "Q5_JOIN":

        """
        SELECT
            c.category_name,
            b.title,
            b.rating,
            b.price_inr

        FROM books b

        JOIN categories c
            ON b.category_id = c.category_id

        ORDER BY
            b.rating DESC,
            c.category_name,
            b.title

        LIMIT 10;
        """
    }

    with open(
        SQL_OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        for name, query in queries.items():

            result = pd.read_sql(
                query,
                connection
            )

            file.write(
                "\n"
                + "=" * 80
                + "\n"
            )

            file.write(name + "\n")

            file.write(
                "=" * 80
                + "\n"
            )

            file.write(query.strip())
            file.write("\n\n")

            file.write(
                result.to_string(
                    index=False
                )
            )

            file.write("\n")

    connection.close()


# ============================================================
# READ SQL + PANDAS MERGE
# ============================================================

def compare_sql_and_pandas_join():

    connection = sqlite3.connect(
        DB_FILE
    )

    join_query = """

    SELECT
        c.category_name,
        b.title,
        b.rating,
        b.price_inr

    FROM books b

    JOIN categories c
        ON b.category_id = c.category_id

    ORDER BY
        b.rating DESC,
        c.category_name,
        b.title

    LIMIT 10;

    """

    sql_result = pd.read_sql(
        join_query,
        connection
    )

    books_df = pd.read_sql(
        "SELECT * FROM books",
        connection
    )

    categories_df = pd.read_sql(
        "SELECT * FROM categories",
        connection
    )

    connection.close()

    # Pandas equivalent JOIN

    merged = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    pandas_result = (
        merged[
            [
                "category_name",
                "title",
                "rating",
                "price_inr"
            ]
        ]
        .sort_values(
            [
                "rating",
                "category_name",
                "title"
            ],
            ascending=[
                False,
                True,
                True
            ]
        )
        .head(10)
        .reset_index(drop=True)
    )

    sql_result = (
        sql_result
        .reset_index(drop=True)
    )

    pandas_result = (
        pandas_result
        .reset_index(drop=True)
    )

    print("\nSQL RESULT")
    print(sql_result)

    print("\nPANDAS MERGE RESULT")
    print(pandas_result)

    print(
        "\nOutputs equivalent:",
        sql_result.equals(
            pandas_result
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nDiscovering book categories..."
    )

    categories = get_categories()

    print(
        f"Found {len(categories)} categories."
    )

    # At least 3 categories.
    selected_categories = list(
        categories.items()
    )[:4]

    all_books = []

    for category, url in selected_categories:

        print(
            f"\nScraping category: {category}"
        )

        books = scrape_category(
            category,
            url
        )

        all_books.extend(books)

        print(
            f"Books collected: {len(books)}"
        )

    raw_df = pd.DataFrame(
        all_books
    )

    raw_df = raw_df.drop_duplicates(
        subset=["title", "category"]
    )

    clean_df = clean_data(
        raw_df
    )

    # Acceptance criteria

    if len(clean_df) < 60:

        raise RuntimeError(
            f"Only {len(clean_df)} books were collected. "
            "At least 60 are required."
        )

    if clean_df["category"].nunique() < 3:

        raise RuntimeError(
            "At least 3 categories are required."
        )

    # Save CSV

    clean_df.to_csv(
        CSV_FILE,
        index=False
    )

    # Database

    create_database(
        clean_df
    )

    # SQL

    execute_queries()

    # Pandas comparison

    compare_sql_and_pandas_join()

    print("\n================================")
    print("MODULE 1 COMPLETED")
    print("================================")

    print(
        "Rows:",
        len(clean_df)
    )

    print(
        "Categories:",
        clean_df["category"].nunique()
    )

    print(
        "CSV:",
        CSV_FILE
    )

    print(
        "Database:",
        DB_FILE
    )

    print(
        "SQL output:",
        SQL_OUTPUT_FILE
    )


if __name__ == "__main__":
    main()

