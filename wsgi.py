# WSGI configuration for PythonAnywhere
import sys
import os

# Add the project directory to the path
project_home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Add backend directory to path
backend_dir = os.path.join(project_home, 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Import the Flask app
from app import app as application
