import bcrypt

hashed = "$2b$12$q8lyq/NpKlA80YMdzrKtPuHkg1pG4HIk1zIDPpKu78TPFy3zw6NW6"
password = "password"

match = bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
print(f"Match: {match}")
