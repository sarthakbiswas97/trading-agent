#!/bin/bash
# VAPM Setup Script

set -e

echo "🚀 Setting up VAPM - Verifiable AI Portfolio Manager"
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.11+"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker"
    exit 1
fi

echo "✅ Prerequisites OK"
echo ""

# Create .env file if not exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env (edit with your settings)"
else
    echo "✅ .env already exists"
fi
echo ""

# Start infrastructure
echo "🐳 Starting Docker containers (PostgreSQL + Redis)..."
docker-compose up -d postgres redis
echo "⏳ Waiting for containers to be ready..."
sleep 5
echo "✅ Infrastructure running"
echo ""

# Setup Python environment
echo "🐍 Setting up Python environment..."
cd backend

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Created virtual environment"
fi

source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ Python dependencies installed"
echo ""

cd ..

echo "=========================================="
echo "✅ Setup complete!"
echo ""
echo "To run the server:"
echo "  cd backend"
echo "  source venv/bin/activate"
echo "  python -m uvicorn main:app --reload"
echo ""
echo "Then visit: http://localhost:8000/docs"
echo "=========================================="
