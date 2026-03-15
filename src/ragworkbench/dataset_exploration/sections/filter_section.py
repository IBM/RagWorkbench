from nicegui import ui

from ragworkbench.dataset_exploration.dataset_model import (
    DatasetAnswerScope,
    DatasetDocumentStructureFormat,
    DatasetDomain,
    DatasetQuestionContextDependency,
    DatasetRetrievalHops,
    DatasetTargetModality,
)


def _add_filter(
    filter_name: str, description_str: str | None, enum_class, upper_value: bool = False
):
    with ui.column():
        with ui.row().classes("items-center gap-1 flex-nowrap"):
            ui.label(filter_name).classes("font-medium whitespace-nowrap")
            ui.icon("help_outline").classes(
                "text-grey-6 text-xs"
            ).tooltip(  # Smaller icon
                description_str
            )

        select_name = (
            ui.select(
                options=[
                    (
                        e.value.replace("_", " ").title()
                        if not upper_value
                        else e.value.replace("_", " ").upper()
                    )
                    for e in enum_class
                ],
                multiple=True,
                value=[],
            )
            .props("clearable outlined dense")
            .classes("w-full")
        )
    return select_name


def filter_section():
    with ui.grid(columns=6).classes("gap-6 w-full"):
        domain_select = _add_filter(
            filter_name="Domain",
            description_str=DatasetDomain.__doc__,
            enum_class=DatasetDomain,
        )
        hops_select = _add_filter(
            filter_name="Retrieval Hops",
            description_str=DatasetRetrievalHops.__doc__,
            enum_class=DatasetRetrievalHops,
        )
        scope_select = _add_filter(
            filter_name="Answer Scope",
            description_str=DatasetAnswerScope.__doc__,
            enum_class=DatasetAnswerScope,
        )
        context_select = _add_filter(
            filter_name="Context Dependency",
            description_str=DatasetQuestionContextDependency.__doc__,
            enum_class=DatasetQuestionContextDependency,
        )
        structure_select = _add_filter(
            filter_name="Document Structure",
            description_str=DatasetDocumentStructureFormat.__doc__,
            enum_class=DatasetDocumentStructureFormat,
        )
        modality_select = _add_filter(
            filter_name="Modalities",
            description_str=DatasetTargetModality.__doc__,
            enum_class=DatasetTargetModality,
            upper_value=True,
        )

    return (
        domain_select,
        hops_select,
        scope_select,
        context_select,
        structure_select,
        modality_select,
    )
