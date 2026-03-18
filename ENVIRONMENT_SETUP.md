# Environment Setup Guide

This guide explains how to configure environment variables for RagWorkbench.

## Quick Start

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your credentials as needed for the metrics you plan to use.**

## Optional Configuration

### watsonx.ai Setup (Optional)

Required only if using watsonx LLM-as-a-Judge evaluation metrics. To use these metrics, you need:

#### 1. IBM Cloud API Key

Get your API key from IBM Cloud:
- Go to [IBM Cloud API Keys](https://cloud.ibm.com/iam/apikeys)
- Click "Create an IBM Cloud API key"
- Copy the key and add it to your `.env` file:
  ```
  WATSONX_APIKEY=your_api_key_here
  ```

#### 2. watsonx.ai Project ID

Get your project ID from watsonx.ai:
- Go to your [watsonx.ai projects](https://dataplatform.cloud.ibm.com/projects)
- Open your project
- Click on "Manage" tab
- Copy the "Project ID" and add it to your `.env` file:
  ```
  WATSONX_PROJECT_ID=your_project_id_here
  ```

#### 3. watsonx.ai URL (Optional)

The default URL is for US South region. If you're using a different region, update:
```
WATSONX_URL=https://your-region.ml.cloud.ibm.com
```

Common regions:
- **US South** (default): `https://us-south.ml.cloud.ibm.com`
- **Frankfurt**: `https://eu-de.ml.cloud.ibm.com`
- **London**: `https://eu-gb.ml.cloud.ibm.com`
- **Tokyo**: `https://jp-tok.ml.cloud.ibm.com`

**Metrics requiring watsonx.ai:**
- `unitxt.answer_correctness.llmaaj_llama` - Uses Llama 3.3 70B on watsonx
- `unitxt.answer_correctness.llmaaj_llama4` - Uses Llama 4 Maverick on watsonx

### Azure OpenAI (for GPT metrics)

If using `unitxt.answer_correctness.llmaaj_gpt`:
```
AZURE_API_KEY=your_azure_key
AZURE_API_BASE=https://your-resource.openai.azure.com
AZURE_API_VERSION=2024-02-15-preview
AZURE_DEPLOYMENT_NAME=gpt-4o
```