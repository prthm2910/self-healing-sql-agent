import os
import psycopg
from dotenv import load_dotenv
from langgraph.store.postgres import PostgresStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.core.config import settings

# Load environment variables
load_dotenv()

def setup_store():
    db_url = os.getenv("DATABASE_URL")
    
    if not db_url:
        print("❌ Error: DATABASE_URL not found.")
        return

    print("🚀 Initializing PostgresStore Setup with Autocommit...")
    
    try:
        # Use autocommit=True for setup because of CREATE INDEX CONCURRENTLY
        with psycopg.connect(db_url, autocommit=True) as conn:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-2",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                output_dimensionality=1536
            )
            
            index_config = {
                "embed": embeddings,
                "dims": 1536,
                "fields": ["$"]
            }
            
            store = PostgresStore(conn, index=index_config)
            print("📦 Running store.setup()...")
            store.setup()
            print("✅ PostgresStore setup completed successfully!")

            # Verify tables
            with conn.cursor() as cur:
                cur.execute("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public';")
                tables = cur.fetchall()
                print(f"📊 Current tables in 'public' schema: {[t[0] for t in tables]}")

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    setup_store()
