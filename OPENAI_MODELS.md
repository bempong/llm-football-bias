# OpenAI Model Reference

## Available Models for test_script.py

### GPT-4o Models (Recommended - Fast and Cost-Effective)
- `gpt-4o-mini` - Most cost-effective GPT-4 level model
- `gpt-4o` - Latest GPT-4 optimized model

### GPT-4 Turbo Models
- `gpt-4-turbo` - Latest GPT-4 Turbo model
- `gpt-4-turbo-preview` - Preview version

### GPT-4 Models
- `gpt-4` - Original GPT-4 model

### GPT-3.5 Models (Budget Option)
- `gpt-3.5-turbo` - Fast and affordable

## Usage Examples

### Using GPT-4o-mini (Recommended for testing)
```bash
python test_script.py \
    --use-api \
    --api-provider openai \
    --model-name gpt-4o-mini \
    --kaggle-path /path/to/tagged_transcripts.json \
    --qb-n 10 \
    --rb-n 10 \
    --wr-n 10 \
    --def-n 10
```

### Using GPT-4
```bash
python test_script.py \
    --use-api \
    --api-provider openai \
    --model-name gpt-4 \
    --kaggle-path /path/to/tagged_transcripts.json
```

### Using GPT-4 Turbo
```bash
python test_script.py \
    --use-api \
    --api-provider openai \
    --model-name gpt-4-turbo \
    --kaggle-path /path/to/tagged_transcripts.json
```

## Swapping Between Models

To easily swap between models, just change the `--model-name` parameter:

```bash
# Test with gpt-4o-mini first
python test_script.py --use-api --api-provider openai --model-name gpt-4o-mini ...

# Then run with gpt-4 for comparison
python test_script.py --use-api --api-provider openai --model-name gpt-4 ...
```

## Setup

1. Get your OpenAI API key from: https://platform.openai.com/api-keys
2. Add it to `.env` file:
   ```
   OPENAI_API_KEY=sk-...your-key-here...
   ```
3. Install the OpenAI package:
   ```bash
   pip install openai
   ```

## Cost Considerations

- **gpt-4o-mini**: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens (cheapest)
- **gpt-4o**: ~$2.50 per 1M input tokens, ~$10 per 1M output tokens
- **gpt-4-turbo**: ~$10 per 1M input tokens, ~$30 per 1M output tokens
- **gpt-4**: ~$30 per 1M input tokens, ~$60 per 1M output tokens (most expensive)

For experimentation and testing, start with `gpt-4o-mini` to save costs!
