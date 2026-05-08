import os
from dotenv import load_dotenv
from langgraph.store.postgres import PostgresStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import psycopg

# Load environment variables
load_dotenv()

def verify_lessons():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL not found.")
        return

    print("🔍 Inspecting Lessons in PostgresStore...")
    
    # Configure Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        output_dimensionality=1536
    )

    try:
        # Connect to Postgres (Synchronous)
        with psycopg.connect(db_url) as conn:
            store = PostgresStore(conn, index={
                "embed": embeddings,
                "dims": 1536,
                "fields": ["$"]
            })
            
            # Namespaces to check
            namespaces = [
                ("global", "lessons", "pinned"),
                ("global", "lessons", "dynamic")
            ]
            
            total_found = 0
            for ns in namespaces:
                print(f"\n--- Namespace: {ns} ---")
                results = store.search(ns, limit=10)
                if not results:
                    print("No lessons found.")
                for res in results:
                    total_found += 1
                    print(f"ID: {res.key}")
                    print(f"Title: {res.value.get('title')}")
                    print(f"Instruction: {res.value.get('instruction')}")
                    print("-" * 20)
            
            print(f"\n✅ Total lessons found: {total_found}")

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_lessons()
