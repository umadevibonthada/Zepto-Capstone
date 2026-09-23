import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from pathlib import Path
from urllib.parse import urljoin


# ---------------------------------------------------------
# PROJECT SETTINGS
# ---------------------------------------------------------

BASE_URL = "https://books.toscrape.com/"
FIXED_GBP_TO_INR = 105.50

OUTPUT_DIR = Path(__file__).resolve().parent
CSV_FILE = OUTPUT_DIR / "books_cleaned.csv"
DB_FILE = OUTPUT_DIR / "books.db"
QUERY_OUTPUT_FILE = OUTPUT_DIR / "query_outputs.txt"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# ---------------------------------------------------------
# 1. SCRAPE BOOK DETAIL PAGE
# ---------------------------------------------------------

def scrape_book_details(book_url):
    response = requests.get(book_url, headers=HEADERS, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Category is available in the breadcrumb on the detail page.
    breadcrumb = soup.select("ul.breadcrumb li")

    category = None

    if len(breadcrumb) >= 3:
        category = breadcrumb[-2].get_text(strip=True)

    return category


# ---------------------------------------------------------
# 2. SCRAPE CATALOGUE PAGES
# ---------------------------------------------------------

def scrape_books():

    all_books = []

    for page_number in range(1, 6):

        if page_number == 1:
            url = BASE_URL
        else:
            url = f"{BASE_URL}catalogue/page-{page_number}.html"

        print(f"Scraping catalogue page {page_number}: {url}")

        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        books = soup.find_all("article", class_="product_pod")

        print(f"Books found on page {page_number}: {len(books)}")

        for book in books:

            # -----------------------------
            # TITLE
            # -----------------------------

            title = book.h3.a.get("title")

            # -----------------------------
            # PRICE
            # -----------------------------

            price = book.find(
                "p",
                class_="price_color"
            ).get_text(strip=True)

            # -----------------------------
            # STAR RATING
            # -----------------------------

            rating_classes = book.find(
                "p",
                class_="star-rating"
            ).get("class", [])

            star_rating = None

            if len(rating_classes) >= 2:
                star_rating = rating_classes[1]

            # -----------------------------
            # AVAILABILITY
            # -----------------------------

            availability = book.find(
                "p",
                class_="availability"
            ).get_text(" ", strip=True)

            # -----------------------------
            # BOOK DETAIL URL
            # -----------------------------

          #  relative_url = book.h3.a.get("href")

           # book_url = BASE_URL + relative_url.replace(
           #     "../",
           #     ""
           # )
        
            relative_url = book.h3.a.get("href")
            book_url = urljoin(url, relative_url)

            # -----------------------------
            # CATEGORY
            # -----------------------------

            try:
                category = scrape_book_details(book_url)
            except Exception as error:
                print(
                    f"Category request failed for {title}: {error}"
                )
                category = None

            all_books.append({
                "title": title,
                "price": price,
                "star_rating": star_rating,
                "availability": availability,
                "category": category
            })

    return pd.DataFrame(all_books)


# ---------------------------------------------------------
# 3. CLEAN DATA
# ---------------------------------------------------------

def clean_data(df):

    print("\nRaw rows:", len(df))

    # -----------------------------
    # PRICE → FLOAT
    # -----------------------------

    df["price_gbp"] = (
        df["price"]
        .astype(str)
        .str.replace("£", "", regex=False)
        .str.replace("Â", "", regex=False)
        .str.strip()
    )

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    # Median imputation for invalid numeric price
    price_median = df["price_gbp"].median()

    df["price_gbp"] = df["price_gbp"].fillna(
        price_median
    )

    # -----------------------------
    # RATING TEXT → INTEGER
    # -----------------------------

    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    df["rating"] = df["star_rating"].map(rating_map)

    # Median imputation for invalid rating
    rating_median = round(df["rating"].median())

    df["rating"] = df["rating"].fillna(
        rating_median
    ).astype(int)

    # -----------------------------
    # AVAILABILITY → BOOLEAN
    # -----------------------------

    def parse_stock(value):

        if pd.isna(value):
            return None

        text = str(value).lower()

        if "in stock" in text:
            return True

        if "out of stock" in text:
            return False

        return None

    df["in_stock"] = df["availability"].apply(
        parse_stock
    )

    # -----------------------------
    # DROP ROWS WHERE NON-NUMERIC
    # PARSING CANNOT BE RESOLVED
    # -----------------------------

    before_drop = len(df)

    df = df.dropna(
        subset=["in_stock", "category"]
    ).copy()

    dropped = before_drop - len(df)

    print("Rows dropped because availability/category could not be parsed:",
          dropped)

    df["in_stock"] = df["in_stock"].astype(bool)

    # -----------------------------
    # FIXED PROJECT CONVERSION
    # -----------------------------

    df["price_inr"] = (
        df["price_gbp"] * FIXED_GBP_TO_INR
    ).round(2)

    return df


# ---------------------------------------------------------
# 4. CREATE SQLITE DATABASE
# ---------------------------------------------------------

def create_database(df):

    connection = sqlite3.connect(DB_FILE)

    cursor = connection.cursor()

    # Enable foreign-key enforcement
    cursor.execute("PRAGMA foreign_keys = ON")

    # Drop old tables so script can be rerun
    cursor.execute("DROP TABLE IF EXISTS books")
    cursor.execute("DROP TABLE IF EXISTS categories")

    # -----------------------------
    # CATEGORIES TABLE
    # -----------------------------

    cursor.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE NOT NULL
        )
    """)

    # -----------------------------
    # BOOKS TABLE
    # -----------------------------

    cursor.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    # -----------------------------
    # INSERT CATEGORIES
    # -----------------------------

    categories = sorted(
        df["category"].dropna().unique()
    )

    for category in categories:

        cursor.execute(
            """
            INSERT INTO categories (category_name)
            VALUES (?)
            """,
            (category,)
        )

    # -----------------------------
    # CREATE CATEGORY LOOKUP
    # -----------------------------

    cursor.execute(
        "SELECT category_id, category_name FROM categories"
    )

    category_lookup = {
        name: category_id
        for category_id, name in cursor.fetchall()
    }

    # -----------------------------
    # INSERT BOOKS
    # -----------------------------

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
                category_lookup[row["category"]]
            )
        )

    connection.commit()

    return connection


# ---------------------------------------------------------
# 5. RUN SQL QUERIES
# ---------------------------------------------------------

def run_sql_queries(connection, df):

    queries = {

        "Q1_SELECT_WHERE": """
            SELECT title, price_gbp, rating
            FROM books
            WHERE rating >= 4
        """,

        "Q2_ORDER_BY": """
            SELECT title, price_gbp
            FROM books
            ORDER BY price_gbp DESC
        """,

        "Q3_LIMIT": """
            SELECT title, price_gbp, rating
            FROM books
            ORDER BY price_gbp DESC
            LIMIT 10
        """,

        "Q4_DISTINCT": """
            SELECT DISTINCT category_name
            FROM categories
            ORDER BY category_name
        """,

        "Q5_BETWEEN": """
            SELECT title, price_gbp
            FROM books
            WHERE price_gbp BETWEEN 10 AND 30
            ORDER BY price_gbp
        """,

        "Q6_JOIN": """
            SELECT
                b.title,
                c.category_name,
                b.price_gbp,
                b.rating,
                b.in_stock
            FROM books b
            JOIN categories c
                ON b.category_id = c.category_id
            ORDER BY b.rating DESC, b.price_gbp DESC
            LIMIT 10
        """
    }

    with open(
        QUERY_OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output_file:

        for query_name, query in queries.items():

            print("\n" + "=" * 60)
            print(query_name)
            print("=" * 60)

            print(query.strip())

            result = pd.read_sql(
                query,
                connection
            )

            print(result.to_string(index=False))

            output_file.write("\n")
            output_file.write("=" * 60 + "\n")
            output_file.write(query_name + "\n")
            output_file.write("=" * 60 + "\n")
            output_file.write(query.strip() + "\n\n")
            output_file.write(
                result.to_string(index=False)
            )
            output_file.write("\n")

    # -----------------------------------------------------
    # pd.read_sql — two query results
    # -----------------------------------------------------

    q1_df = pd.read_sql(
        queries["Q1_SELECT_WHERE"],
        connection
    )

    q3_df = pd.read_sql(
        queries["Q3_LIMIT"],
        connection
    )

    print("\nQ1 loaded using pd.read_sql:")
    print(q1_df.head())

    print("\nQ3 loaded using pd.read_sql:")
    print(q3_df.head())

    # -----------------------------------------------------
    # JOIN USING pd.merge
    # -----------------------------------------------------

    books_df = pd.read_sql(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            rating,
            in_stock,
            category_id
        FROM books
        """,
        connection
    )

    categories_df = pd.read_sql(
        """
        SELECT category_id, category_name
        FROM categories
        """,
        connection
    )

    merged_df = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    merged_join_result = (
        merged_df[
            [
                "title",
                "category_name",
                "price_gbp",
                "rating",
                "in_stock"
            ]
        ]
        .sort_values(
            ["rating", "price_gbp"],
            ascending=[False, False]
        )
        .head(10)
        .reset_index(drop=True)
    )

    sql_join_result = pd.read_sql(
        queries["Q6_JOIN"],
        connection
    ).reset_index(drop=True)

    # Make column order/types comparable
    sql_join_result["in_stock"] = (
        sql_join_result["in_stock"].astype(bool)
    )

    merged_join_result["in_stock"] = (
        merged_join_result["in_stock"].astype(bool)
    )

    equivalent = sql_join_result.equals(
        merged_join_result
    )

    print("\n" + "=" * 60)
    print("SQL JOIN vs pandas.merge()")
    print("=" * 60)

    print("\nSQL JOIN result:")
    print(sql_join_result)

    print("\npd.merge() result:")
    print(merged_join_result)

    print("\nEquivalent:", equivalent)

    with open(
        QUERY_OUTPUT_FILE,
        "a",
        encoding="utf-8"
    ) as output_file:

        output_file.write(
            "\n\nSQL JOIN vs pandas.merge()\n"
        )
        output_file.write(
            "\nSQL JOIN result:\n"
        )
        output_file.write(
            sql_join_result.to_string(index=False)
        )
        output_file.write(
            "\n\npandas.merge() result:\n"
        )
        output_file.write(
            merged_join_result.to_string(index=False)
        )
        output_file.write(
            f"\n\nEquivalent: {equivalent}\n"
        )


# ---------------------------------------------------------
# 6. MAIN PIPELINE
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("ZEPT0 DATA PIPELINE")
    print("=" * 60)

    # SCRAPE
    df = scrape_books()

    print("\nTotal scraped rows:", len(df))

    # CLEAN
    df = clean_data(df)

    print("Final cleaned rows:", len(df))

    # CHECK REQUIREMENT
    category_count = df["category"].nunique()

    print("Number of categories:", category_count)

    if len(df) < 60:
        raise ValueError(
            f"Requirement failed: only {len(df)} books found."
        )

    if category_count < 3:
        raise ValueError(
            f"Requirement failed: only {category_count} categories found."
        )

    # SAVE CSV
    df.to_csv(
        CSV_FILE,
        index=False,
        encoding="utf-8"
    )

    print(f"\nCSV saved to: {CSV_FILE}")

    # DATABASE
    connection = create_database(df)

    print(f"SQLite database saved to: {DB_FILE}")

    # SQL
    run_sql_queries(
        connection,
        df
    )

    connection.close()

    print(
        f"\nQuery outputs saved to: {QUERY_OUTPUT_FILE}"
    )

    print("\n" + "=" * 60)
    print("MODULE 1 COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()