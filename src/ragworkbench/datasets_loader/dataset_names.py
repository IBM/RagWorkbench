from enum import StrEnum, unique


@unique
class DatasetName(StrEnum):
    """
    Enumeration of available RAG benchmark dataset names.

    This enum defines the supported dataset identifiers used throughout the system.
    Each value represents a unique dataset that can be loaded and processed.
    The @unique decorator ensures no duplicate values exist.

    Attributes:
        AIT_QA: AIT QA (AI and Technology Question Answering) dataset identifier.
        BIOASQ: BioASQ biomedical question answering dataset identifier.
        CLAP_NQ: CLAP-NQ dataset identifier.
        DA_CODE: DA-Code dataset identifier.
        DABSTEP: DABStep dataset identifier.
        HOTPOT_QA: HotpotQA multi-hop question answering dataset identifier.
        KRAMABENCH: KramaBench dataset identifier.
        MINI_WIKI: Mini Wikipedia RAG dataset identifier.
        MLDR: MLDR (Multilingual Long Document Retrieval) dataset identifier.
        NARRATIVE_QA: NarrativeQA reading comprehension dataset identifier.
        OFFICEQA: OfficeQA dataset identifier.
        QASPER: QASPER (Question Answering on Scientific Papers) dataset identifier.
        SECQUE: SecQue dataset identifier.
        WATSONX_DOCS_QA_TXT: WatsonX DocsQA enterprise documentation dataset identifier.

    Example:
        >>> dataset = DatasetName.AI_ARXIV
        >>> print(dataset.value)
        'ai_arxiv'
        >>> DatasetName("ai_arxiv")
        <DatasetName.AI_ARXIV: 'ai_arxiv'>
        >>> DatasetName.list_all()
        ['ai_arxiv', 'bioasq', 'clap_nq', ...]
        >>> DatasetName.is_valid("bioasq")
        True
    """

    AIT_QA = "ait_qa"
    BIOASQ = "bioasq"
    CLAP_NQ = "clap_nq"
    DA_CODE = "da_code"
    DABSTEP = "dabstep"
    HOTPOT_QA = "hotpot_qa"
    KRAMABENCH = "kramabench"
    MINI_WIKI = "mini_wiki"
    MLDR = "mldr"
    NARRATIVE_QA = "narrative_qa"
    OFFICEQA = "officeqa"
    QASPER = "qasper"
    SECQUE = "secque"
    WATSONX_DOCS_QA_TXT = "watsonx_docs_qa_txt"
    WATSONX_DOCS_QA_HTML = "watsonx_docs_qa_html"
    WATSONX_DOCS_QA_MD = "watsonx_docs_qa_md"
    REAL_MM_FIN_SLIDES = "real_mm_fin_slides"
    REAL_MM_FIN_REPORT = "real_mm_fin_report"
    REAL_MM_TECH_REPORT = "real_mm_tech_report"
    REAL_MM_TECH_SLIDES = "real_mm_tech_slides"

    @classmethod
    def list_all(cls) -> list[str]:
        """
        Return all dataset names as a list of strings.

        Returns:
            List of all available dataset name values.

        Example:
            >>> DatasetName.list_all()
            ['ai_arxiv', 'bioasq', 'clap_nq', ...]
        """
        return [member.value for member in cls]

    @classmethod
    def is_valid(cls, name: str) -> bool:
        """
        Check if a string is a valid dataset name.

        Args:
            name: String to validate.

        Returns:
            True if name is valid, False otherwise.

        Example:
            >>> DatasetName.is_valid("bioasq")
            True
            >>> DatasetName.is_valid("invalid")
            False
        """
        try:
            cls(name)
            return True
        except ValueError:
            return False

    @classmethod
    def from_string(cls, name: str) -> "DatasetName":
        """
        Create DatasetName from string with better error message.

        Args:
            name: Dataset name string.

        Returns:
            DatasetName enum member.

        Raises:
            ValueError: If name is not a valid dataset.

        Example:
            >>> DatasetName.from_string("bioasq")
            <DatasetName.BIOASQ: 'bioasq'>
        """
        try:
            return cls(name)
        except ValueError:
            valid_names = ", ".join(cls.list_all())
            raise ValueError(
                f"Invalid dataset name: '{name}'. " f"Valid options are: {valid_names}"
            ) from None
