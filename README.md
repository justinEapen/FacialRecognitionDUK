# Face Recognition Access Control System

This system provides facial recognition-based access control with a web interface for managing authorized personnel.

## Features

- Face Registration System
- Real-time Face Verification
- Access Control Output
- Logging System
- Admin Interface
- Various Lighting Condition Support
- Low False Positive Rate

## Requirements

- Python 3.8+
- Webcam
- Dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```

## Usage

1. Access the admin interface at http://localhost:5000/admin
2. Register new faces through the registration interface
3. The system will automatically monitor for faces and control access

## Security Note

This system should be used as part of a comprehensive security solution and not as the sole means of access control.
