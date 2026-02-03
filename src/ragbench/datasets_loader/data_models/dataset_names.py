from enum import Enum, unique


@unique
class DatasetName(str, Enum):
    """
    Enumeration of available RAG benchmark dataset names.

    This enum defines the supported dataset identifiers used throughout the system.
    Each value represents a unique dataset that can be loaded and processed.
    The @unique decorator ensures no duplicate values exist.

    Attributes:
        AI_ARXIV: ArXiv AI papers dataset identifier.
        BIOASQ: BioASQ biomedical question answering dataset identifier.
        CLAP_NQ: CLAP-NQ dataset identifier.
        HOTPOT_QA: HotpotQA multi-hop question answering dataset identifier.

    Example:
        >>> dataset = DatasetName.AI_ARXIV
        >>> print(dataset.value)
        'ai_arxiv'
        >>> DatasetName("ai_arxiv")
        <DatasetName.AI_ARXIV: 'ai_arxiv'>
    """

    AI_ARXIV = "ai_arxiv"
    BIOASQ = "bioasq"
    CLAP_NQ = "clap_nq"
    HOTPOT_QA = "hotpot_qa"
