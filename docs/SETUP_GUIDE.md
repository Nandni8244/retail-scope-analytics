# RetailScope Analytics - Setup Guide

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Git

### Installation Steps

1. **Clone the repository**

```bash
git clone <repository-url>
cd retailscope-analytics
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Setup environment variables**

```bash
cp .env.example .env
# Edit .env with your database credentials
```

5. **Run the complete pipeline**

```bash
python src/data_pipeline/main.py --mode full
```

## 📋 Detailed Setup

### Database Configuration

1. **Install PostgreSQL**

   - Download from [postgresql.org](https://www.postgresql.org/download/)
   - Create a database named `retailscope_analytics`

2. **Update .env file**

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=retailscope_analytics
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
```

### Pipeline Execution Options

**Full Pipeline (Recommended for first run)**

```bash
python src/data_pipeline/main.py --mode full
```

**Individual Components**

```bash
# Setup database only
python src/data_pipeline/main.py --mode setup

# Generate sample data only
python src/data_pipeline/main.py --mode data

# Run ETL only
python src/data_pipeline/main.py --mode etl

# Run analytics only
python src/data_pipeline/main.py --mode analytics

# Generate reports only
python src/data_pipeline/main.py --mode reports
```

### Starting the API Server

```bash
python src/api/app.py
```

API will be available at `http://localhost:5000`

### Viewing the Dashboard

Open `src/visualization/dashboard.html` in your web browser after starting the API server.

## 🔧 Troubleshooting

### Common Issues

**Database Connection Error**

- Verify PostgreSQL is running
- Check credentials in `.env` file
- Ensure database exists

**Missing Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Permission Errors**

- Ensure proper file permissions
- Run with appropriate user privileges

**Memory Issues**

- Reduce dataset size in `data_generator.py`
- Increase system memory allocation

## 📊 Understanding the Output

### Generated Files

- `data/raw/` - Original CSV files
- `data/processed/` - Cleaned datasets
- `reports/` - PDF and Excel reports
- Database tables with analytics results

### Key Metrics Available

- Customer segmentation (RFM analysis)
- Sales forecasting (30-day predictions)
- Churn prediction (risk scores)
- Product performance analytics

## 🎯 Next Steps

1. **Customize the Analysis**

   - Modify parameters in analytics modules
   - Add new data sources
   - Create custom visualizations

2. **Deploy to Production**

   - Setup cloud database
   - Configure automated scheduling
   - Implement monitoring

3. **Extend Functionality**
   - Add real-time data ingestion
   - Implement A/B testing analytics
   - Create mobile dashboard
