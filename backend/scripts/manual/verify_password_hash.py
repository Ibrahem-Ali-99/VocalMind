import bcrypt


HASHED_PASSWORD = "$2b$12$q8lyq/NpKlA80YMdzrKtPuHkg1pG4HIk1zIDPpKu78TPFy3zw6NW6"
PASSWORD = "password"


if __name__ == "__main__":
    match = bcrypt.checkpw(PASSWORD.encode("utf-8"), HASHED_PASSWORD.encode("utf-8"))
    print(f"Match: {match}")
