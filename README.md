# Automated Data Labeling Assistant

A powerful, AI-driven system for automatically labeling unstructured text data with human-in-the-loop review capabilities. Built with FastAPI, SQLAlchemy, DistilBERT.

## Features

### AI-Powered Labeling (100% Free!)
- **Multiple LLM Options**: Choose from local models, Hugging Face, Ollama, or free APIs
- **Local Processing**: Run AI models directly on your computer (no API costs)
- **Confidence Scoring**: Each label comes with a confidence score (0.0-1.0)
- **Explanation Generation**: AI provides reasoning for each classification
- **Batch Processing**: Handle multiple texts simultaneously

### Human Review System
- **Confidence Thresholds**: Configurable thresholds for automatic review triggers
- **Review Actions**: Approve, reject, or modify AI-generated labels
- **Audit Trail**: Complete history of all review decisions
- **Reviewer Tracking**: Track who made what changes and when

### Comprehensive Analytics
- **Real-time Statistics**: Dashboard with key metrics
- **Category Distribution**: Visual breakdown of label distribution
- **Performance Metrics**: Auto-labeling rates and review coverage
- **Quality Insights**: Confidence score analysis

### Flexible Database Support
- **SQLite**: Default lightweight database for development
- **PostgreSQL**: Production-ready database support
- **Schema Management**: Automatic table creation and migration
- **Data Integrity**: Proper relationships and constraints

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   FastAPI App   │    │   Database      │
│   (Templates)   │◄──►│   (Main.py)     │◄──►│   (SQLite/PG)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌────────┴────────┐
                       │                 │
              ┌─────────────────┐ ┌─────────────────┐
              │  LLM Labeler    │ │ Review Service  │
              │  (Free Models)  │ │ (Human Review)  │
              └─────────────────┘ └─────────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- **No API keys required!** - Uses free local models
- (Optional) PostgreSQL for production

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd automated-data-labeling
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   # Copy the example environment file
   cp env_example.txt .env
   
   # Edit .env to choose your preferred LLM provider (all free!)
   LLM_PROVIDER=local  # Options: local, huggingface, ollama, free_api
   ```

4. **Run the application**
   ```bash
   python run.py
   ```

5. **Access the web interface**
   - Open your browser to `http://localhost:8000`
   - The API documentation is available at `http://localhost:8000/docs`

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `sqlite:///./data_labeling.db` |
| `LLM_PROVIDER` | AI model provider (local, huggingface, ollama, free_api) | `local` |
| `LOCAL_MODEL_NAME` | Local model name for local/huggingface providers | `distilbert-base-uncased` |
| `USE_GPU` | Use GPU acceleration if available | `False` |
| `DEBUG` | Enable debug mode | `False` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `MIN_CONFIDENCE_THRESHOLD` | Confidence threshold for review | `0.8` |
| `MAX_BATCH_SIZE` | Maximum texts per batch | `100` |

### Database Configuration

#### SQLite (Default)
```bash
DATABASE_URL=sqlite:///./data_labeling.db
```

#### PostgreSQL
```bash
DATABASE_URL=postgresql://username:password@localhost:5432/dbname
```

## Usage

### 1. Labeling Text

#### Single Text
- Navigate to `/label`
- Enter text content and optional source
- Click "Label with AI"
- View the generated category, confidence, and explanation

#### Batch Processing
- Use the batch form to label multiple texts
- Enter one text per line
- Process all texts simultaneously

### 2. Reviewing Labels

#### Access Review Interface
- Navigate to `/review`
- Enter your reviewer name
- Set confidence threshold for review triggers
- Click "Load Pending Reviews"

#### Review Actions
- **Approve**: Accept the AI-generated label
- **Reject**: Provide a new category
- **Modify**: Change category and/or confidence
- Add optional comments for each decision

### 3. Monitoring Progress

#### Dashboard
- View overview statistics at `/`
- See recent pending reviews
- Quick access to key actions

#### Statistics Page
- Detailed analytics at `/statistics`
- Category distribution charts
- Performance metrics
- Review action breakdown

## API Endpoints

### Labeling
- `POST /api/label` - Label single text
- `POST /api/label/batch` - Label multiple texts

### Review Management
- `GET /api/reviews/pending` - Get texts needing review
- `POST /api/reviews/approve` - Approve a label
- `POST /api/reviews/reject` - Reject and provide new category
- `POST /api/reviews/modify` - Modify existing label

### Analytics
- `GET /api/statistics` - Get comprehensive statistics
- `GET /api/categories` - Get available categories

## Database Schema

### Core Tables
- **`text_data`**: Raw text content and metadata
- **`labels`**: AI-generated and human-reviewed labels
- **`reviews`**: Human review decisions and history
- **`categories`**: Available classification categories

### Key Relationships
- Each text can have multiple labels (versioning)
- Each label can have multiple reviews (audit trail)
- Categories are configurable and extensible
- [ ] Docker containerization
- [ ] Kubernetes deployment guides 
