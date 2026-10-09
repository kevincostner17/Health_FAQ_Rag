import os

# Keep unit-test runs out of the LangSmith project (load_dotenv does not override this).
os.environ["LANGSMITH_TRACING"] = "false"
