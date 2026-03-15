from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ragworkbench import DatasetName


class DatasetDomain(StrEnum):
    """The domain of knowledge used in the dataset."""

    WIKIPEDIA = "wikipedia"
    SCIENTIFIC_PAPERS = "scientific_papers"
    FINANCIAL = "financial"
    POLICIES = "policies"
    SALES = "sales"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    LITERATURE = "literature"
    DATA_SCIENCE = "data_science"
    OTHER = "other"


class DatasetRetrievalHops(StrEnum):
    """The number of retrieval steps required to answer questions in the dataset.\nSINGLE_HOP = the questions can be answered with a single retrieval operation.\nMULTI_HOP = the questions require multiple retrieval operations to collect all necessary information."""

    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"


class DatasetAnswerScope(StrEnum):
    """The scope of information needed to answer questions in the dataset.\nPASSAGE = the questions can be answered using a small number of individual data segments.\nCORPUS = the questions require a comprehensive view, integrating information across many segments of the data."""

    PASSAGE = "passage"
    ARTIFACT = "artifact"
    CORPUS = "corpus"


class DatasetQuestionContextDependency(StrEnum):
    """Indicates whether questions depend on a pre-defined context for unambiguous interpretation.\nCONTEXTUALIZED = the questions are posed within a specific context and may be ambiguous or uninterpretable without that prior context.\nUNCONTEXTUALIZED = the questions are self-contained and can be interpreted and answered unambiguously without any additional contextual information."""

    CONTEXTUALIZED = "contextualized"
    UNCONTEXTUALIZED = "uncontextualized"


class DatasetTargetModality(StrEnum):
    """Specifies which information modalities present in the source documents are required to answer the questions accurately.\nTEXT = textual passages.\nTABLE = tabular data.\nIMAGE = visual content.\n"""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"


class DatasetDocumentStructureFormat(StrEnum):
    """Classifies the structural organization of the source documents based on the presence or absence of predefined, consistent patterns.\nSTRUCTURED = The document content follows a consistent and well-defined schema throughout (e.g., standardized tables, key–value fields, uniform forms).\nUNSTRUCTURED = The document lacks a predefined schema and is composed primarily of free-form content (e.g., narrative "text, arbitrary layouts, images)"""

    UNSTRUCTURED = "unstructured"
    STRUCTURED = "structured"
    # SEMI_STRUCTURED = "semi_structured"


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )


class DatasetRecord(StrictBaseModel):
    name: DatasetName = Field(..., description="The name of the dataset")

    domain: DatasetDomain = Field(
        ...,
        description=DatasetDomain.__doc__,
    )

    retrieval_hops: DatasetRetrievalHops = Field(
        ...,
        description=DatasetRetrievalHops.__doc__,
    )

    answer_scope: DatasetAnswerScope = Field(
        ...,
        description=DatasetAnswerScope.__doc__,
    )

    question_context_dependency: DatasetQuestionContextDependency = Field(
        ...,
        description=DatasetQuestionContextDependency.__doc__,
    )
    targeted_modalities: list[DatasetTargetModality] = Field(
        ...,
        description=DatasetTargetModality.__doc__,
    )

    document_structure_format: DatasetDocumentStructureFormat = Field(
        ...,
        description=DatasetDocumentStructureFormat.__doc__,
    )
    is_private_dataset: bool = Field(
        default=False,
        description="Whether the dataset is private (not publicly available or restricted access).",
    )

    subsets: list[DatasetName] = Field(default_factory=list)

    description: str = Field(
        ...,
        description="A textual description of the dataset.\n",
    )

    corpus_size: int = Field(
        ..., description="The number of documents in the corpus.\n"
    )
    train_size: int = Field(
        ..., description="The number of questions in the train split.\n"
    )
    test_size: int = Field(
        ..., description="The number of questions in the test split.\n"
    )

    url: str | None = Field(
        description="URL of the dataset.", default=None
    )  # Not mandatory


class DatasetSplitStats(StrictBaseModel):
    n_questions: int
    n_documents: int


class DatasetRecordStats(DatasetRecord):
    all_stats: DatasetSplitStats | None = None
    train_stats: DatasetSplitStats
    test_stats: DatasetSplitStats

    @model_validator(mode="after")
    def _set_all_stats(self):
        if not self.all_stats:
            self.all_stats = DatasetSplitStats(
                # the number of questions in the full set is the sum of train and test.
                # as for documents, we assume the same docs are used in both splits so
                # no need to sum here. if that's not the case for all datasets,
                # we need to add some indicator that tells us when to sum the docs
                # and when not to do that.
                n_questions=self.train_stats.n_questions + self.test_stats.n_questions,
                n_documents=self.train_stats.n_documents,
            )
        return self


if __name__ == "__main__":
    for v in DatasetTargetModality:
        print(v)
    print(DatasetTargetModality.__doc__)
