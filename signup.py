# auth/signup.py
from fastapi import HTTPException
from passlib.context import CryptContext
import mysql.connector
from mysql.connector import pooling
from pydantic import BaseModel, EmailStr, validator
import re

# Configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

connection_pool = pooling.MySQLConnectionPool(
    pool_name="auth_pool",
    pool_size=5,
    host="localhost",
    user="root",
    password="karan12345",
    database="auth_db"
)

# Models
class UserSignup(BaseModel):
    username: str
    email: EmailStr
    password: str

    @validator('username')
    def username_must_be_valid(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        return v

    @validator('password')
    def password_must_be_strong(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters')
        if not re.search(r'[0-9!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must include at least one number or symbol')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

# Functions
def init_database():
    """Initialize the database tables"""
    conn = connection_pool.get_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    cursor.close()
    conn.close()

async def create_user(user: UserSignup):
    """Create a new user"""
    hashed_password = pwd_context.hash(user.password)

    try:
        conn = connection_pool.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s",
                      (user.username, user.email))
        existing_user = cursor.fetchone()

        if existing_user:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=400, detail="Username or email already exists")

        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
            (user.username, user.email, hashed_password)
        )
        conn.commit()
        cursor.close()
        conn.close()

        return {"status": "success", "message": "User created successfully"}

    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Database error: {err}")

async def authenticate_user(user: UserLogin):
    """Authenticate user login"""
    try:
        conn = connection_pool.get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM users WHERE email = %s", (user.email,))
        db_user = cursor.fetchone()
        cursor.close()
        conn.close()

        if not db_user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        if not pwd_context.verify(user.password, db_user["password"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        return {"status": "success", "message": "Login successful"}

    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Database error: {err}")