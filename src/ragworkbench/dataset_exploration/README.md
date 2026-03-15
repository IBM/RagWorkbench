# Dataset Explorer

A web-based interactive application for exploring and filtering RAG (Retrieval-Augmented Generation) benchmark datasets.

## Overview

The Dataset Explorer is a NiceGUI-based web application that provides an intuitive interface to browse, search, and filter various RAG benchmark datasets. It displays comprehensive metadata about each dataset including domain, retrieval characteristics, modalities, and structural information.

## How to Launch

### Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install nicegui
```

### Running the Application


From the project root directory:

```bash
python -m ragbench.dataset_exploration.dataset_explorer
```


### Accessing the Application

Once launched, the application will be available at:

```
http://localhost:8080
```

The application automatically opens in **dark mode** and features hot-reload for development.

## What It Does

### Main Features

#### 1. **Dataset Table View**
- Displays all available RAG benchmark datasets in a sortable, paginated table (15 items per page)
- Shows key attributes for each dataset:
  - **Dataset Name**: Clickable name with copy-to-clipboard functionality
  - **Domain**: Color-coded badges (Wikipedia, Financial, Scientific Papers, etc.)
  - **Retrieval Hops**: Number of retrieval steps required
  - **Answer Scope**: Scope of the answer (e.g., single document, multiple documents)
  - **Context Dependency**: Whether questions require context
  - **Target Modalities**: Supported data types (TEXT, TABLE, IMAGE)
  - **Document Structure**: Format of documents in the dataset
  - **Description**: Brief description with tooltip for full text

#### 2. **Advanced Filtering System**

Multi-select filters for:
- **Domain**: Filter by dataset domain (Wikipedia, Financial, Scientific Papers, Technical Documentation, Policies, Sales, Other)
- **Retrieval Hops**: Filter by number of retrieval steps (Single Hop, Multi Hop)
- **Answer Scope**: Filter by answer scope characteristics
- **Context Dependency**: Filter by question context requirements
- **Document Structure**: Filter by document format
- **Target Modalities**: Filter by supported modalities (TEXT, TABLE, IMAGE)

All filters support multiple selections and work with OR logic within each filter category.

#### 3. **Search Functionality**
- Global search bar to filter datasets by name or description
- Real-time filtering as you type
- Case-insensitive search


#### 4. **Dataset Details Dialog**
Click on any dataset row to view comprehensive details:
- Full dataset name with copy functionality
- Domain badge
- Retrieval characteristics
- Answer scope and context dependency
- Document structure format
- **Corpus size**: Total number of documents
- **Benchmark sizes**: Training and test set sizes
- **URL**: Link to dataset source (if available)
- Target modalities with color-coded badges
- Full description
