# Free LLM Setup Guide

This guide will help you set up the Automated Data Labeling Assistant with **100% FREE** AI models. No API keys or subscriptions required!

## Quick Start (Recommended)

### Option 1: Local Models (Easiest - No Internet Required)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create environment file:**
   ```bash
   cp env_example.txt .env
   ```

3. **Edit .env file:**
   ```bash
   LLM_PROVIDER=local
   LOCAL_MODEL_NAME=distilbert-base-uncased
   USE_GPU=False
   ```

4. **Run the system:**
   ```bash
   python run.py
   ```

That's it! The system will automatically download the model (~400MB) on first run and work completely offline.

## Advanced Free Options

### Option 2: Hugging Face Models

1. **Set in .env:**
   ```bash
   LLM_PROVIDER=huggingface
   HUGGINGFACE_MODEL=distilbert-base-uncased
   ```

2. **Optional**: Get a free Hugging Face API key for private models:
   - Go to [huggingface.co](https://huggingface.co)
   - Create free account
   - Get API key from settings
   - Add to .env: `HUGGINGFACE_API_KEY=your_key_here`

### Option 3: Ollama (Large Language Models)

1. **Install Ollama:**
   - **Windows**: Download from [ollama.ai](https://ollama.ai)
   - **Mac/Linux**: `curl -fsSL https://ollama.ai/install.sh | sh`

2. **Download a model:**
   ```bash
   ollama pull llama2
   ```

3. **Set in .env:**
   ```bash
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama2
   ```

4. **Start Ollama service:**
   ```bash
   ollama serve
   ```

### Option 4: Free APIs (Minimal Local Resources)

1. **Set in .env:**
   ```bash
   LLM_PROVIDER=free_api
   FREE_API_URL=https://api-inference.huggingface.co/models/
   FREE_API_MODEL=distilbert-base-uncased
   ```

2. **Optional**: Get free Hugging Face API key for higher rate limits

## Model Comparison

| Option | Setup Time | Accuracy | Speed | RAM Usage | Internet |
|--------|------------|----------|-------|-----------|----------|
| **Local** | 5 min | High | High | Medium | No |
| **Hugging Face** | 5 min | Very High | High | Medium | Yes (once) |
| **Ollama** | 15 min | Very High | Medium | Low | No |
| **Free API** | 2 min | High | Medium | Very Low | Yes |

## Recommended for Different Users

### First Time Users
- **Choose**: `LLM_PROVIDER=local`
- **Why**: Easiest setup, works offline, good accuracy
- **Setup Time**: 5 minutes

### Developers/Advanced Users
- **Choose**: `LLM_PROVIDER=ollama`
- **Why**: Best accuracy, large language models, local processing
- **Setup Time**: 15 minutes

### Quick Setup (Internet Required)
- **Choose**: `LLM_PROVIDER=free_api`
- **Why**: Fastest setup, no local resources
- **Setup Time**: 2 minutes

### Privacy-Conscious Users
- **Choose**: `LLM_PROVIDER=local` or `LLM_PROVIDER=ollama`
- **Why**: All processing happens on your computer
- **Setup Time**: 5-15 minutes

## Troubleshooting

### Common Issues

#### "Model failed to load"
- **Solution**: The system will automatically fall back to TF-IDF method
- **Why**: Ensures the system always works, even if models fail

#### "Out of memory"
- **Solution**: Use smaller models or free API option
- **Models by RAM usage**:
  - `distilbert-base-uncased`: ~2GB RAM
  - `sentence-transformers/all-MiniLM-L6-v2`: ~1GB RAM
  - Free API: ~100MB RAM

#### "Download failed"
- **Solution**: Check internet connection or use free API option
- **Alternative**: Download models manually from Hugging Face

### Performance Tips

1. **For better speed**: Use `USE_GPU=True` if you have a GPU
2. **For lower RAM**: Use smaller models like `all-MiniLM-L6-v2`
3. **For offline use**: Download models once, then disconnect internet
4. **For batch processing**: Use local models for best performance

## Switching Between Options

You can easily switch between different LLM providers by changing the `LLM_PROVIDER` in your `.env` file:

```bash
# Switch to local models
LLM_PROVIDER=local

# Switch to free API
LLM_PROVIDER=free_api

# Switch to Ollama
LLM_PROVIDER=ollama
```

The system will automatically detect the change and use the new provider on the next restart.

## Pro Tips

1. **Start with local models** - they're the most reliable and free
2. **Use free API for testing** - fastest setup for trying the system
3. **Consider Ollama for production** - best accuracy for serious use
4. **Monitor RAM usage** - local models need 2-4GB RAM
5. **Backup your models** - once downloaded, you can reuse them

## You're All Set!

With these free options, you can:
- Label thousands of texts without any costs
- Work completely offline (local models)
- Get enterprise-level AI capabilities
- Maintain full privacy and control
- Scale to any size dataset

**No more API costs, no more rate limits, no more internet dependencies!** 