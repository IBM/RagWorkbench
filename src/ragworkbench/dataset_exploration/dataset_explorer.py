from nicegui import ui

from ragworkbench.dataset_exploration.dataset_model import (
    DatasetDomain,
    DatasetTargetModality,
)
from ragworkbench.dataset_exploration.javascript_sections.badges import add_badges
from ragworkbench.dataset_exploration.javascript_sections.copy_dataset_name import (
    copy_dataset_name,
)
from ragworkbench.dataset_exploration.rag_datasets import DatasetRegistry
from ragworkbench.dataset_exploration.sections.filter_section import filter_section

# ───── Color mappings for badges ─────
DOMAIN_COLORS: dict[str, str] = {
    DatasetDomain.WIKIPEDIA: "blue-7",
    DatasetDomain.FINANCIAL: "green-7",
    DatasetDomain.SCIENTIFIC_PAPERS: "teal-7",
    DatasetDomain.TECHNICAL_DOCUMENTATION: "deep-purple-7",
    DatasetDomain.POLICIES: "pink-7",
    DatasetDomain.SALES: "red-7",
    DatasetDomain.OTHER: "grey-7",
}


MODALITY_COLORS: dict[str, str] = {
    DatasetTargetModality.TEXT.upper(): "grey-8",
    DatasetTargetModality.TABLE.upper(): "amber-8",
    DatasetTargetModality.IMAGE.upper(): "purple-8",
}

# ───── End Color mappings for badges ─────

dataset_registry = DatasetRegistry()


@ui.page("/")
def main_page():
    ui.add_head_html(
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
    )

    # Start in dark mode!
    dark = ui.dark_mode(value=True)
    include_private_datasets_check_box = False
    with ui.column().classes("w-full max-w-7xl mx-auto p-8 gap-10"):

        # ───── Title and dark switch ─────
        with ui.row().classes("w-full justify-between items-center mb-8"):
            ui.label("Dataset Explorer").classes("text-4xl font-bold text-primary")
            ui.switch("Dark Mode", value=True).bind_value_to(dark, "value")

        # ───── Filters ─────
        (
            domain_select,
            hops_select,
            scope_select,
            context_select,
            structure_select,
            modality_select,
        ) = filter_section()

        # ───── Reset and private datasets checkbox ─────
        with ui.row().classes("justify-between items-center mt-6"):
            ui.button("Reset All Filters", on_click=lambda: reset_filters()).props(
                "outline flat"
            )
            # include_private_datasets_check_box = ui.checkbox(
            #     "Include private datasets"
            # ).classes("text-grey-6")
            count_label = ui.label("").classes("text-lg text-gray-700")

        # ───── Global Search Bar ─────
        search_input = (
            ui.input("Search by name or description", placeholder="Type to filter...")
            .props("outlined dense")
            .classes("w-full max-w-md mx-auto mb-6")
        )

        # ───── Updated Filtering Logic (multi-select support) ─────
        def update_table(*_):
            filtered = dataset_registry.list()

            # Multi-select filters (OR within each filter)
            if domain_select.value:
                selected = [v.lower().replace(" ", "_") for v in domain_select.value]
                filtered = [r for r in filtered if r.domain.value in selected]

            if hops_select.value:
                selected = [v.lower().replace(" ", "_") for v in hops_select.value]
                filtered = [r for r in filtered if r.retrieval_hops.value in selected]

            if scope_select.value:
                selected = [v.lower().replace(" ", "_") for v in scope_select.value]
                filtered = [r for r in filtered if r.answer_scope.value in selected]

            if context_select.value:
                selected = [v.lower().replace(" ", "_") for v in context_select.value]
                filtered = [
                    r
                    for r in filtered
                    if r.question_context_dependency.value in selected
                ]

            if structure_select.value:
                selected = [v.lower().replace(" ", "_") for v in structure_select.value]
                filtered = [
                    r for r in filtered if r.document_structure_format.value in selected
                ]

            if modality_select.value:
                filtered = [
                    r
                    for r in filtered
                    if any(
                        m.value.upper() in modality_select.value
                        for m in r.targeted_modalities
                    )
                ]
            # Private datasets filter
            # if not include_private_datasets_check_box.value:  # type: ignore[attr-defined]
            #     filtered = [r for r in filtered if not r.is_private_dataset]

            # Global search
            search_term = (
                search_input.value.strip().lower() if search_input.value else ""
            )
            if search_term:
                filtered = [
                    r
                    for r in filtered
                    if search_term in str(r.name).lower()
                    or search_term in r.description.lower()
                ]
            rows = []
            for record in filtered:
                domain_val = record.domain.value
                rows.append(
                    {
                        "name": f"{str(record.name.value)}",
                        "domain": domain_val.replace("_", " ").title(),
                        "domain_color": DOMAIN_COLORS.get(domain_val, "grey-7"),
                        "hops": record.retrieval_hops.value.replace("_", " ").title(),
                        "scope": record.answer_scope.value.replace("_", " ").title(),
                        "context_dep": record.question_context_dependency.value.replace(
                            "_", " "
                        ).title(),
                        "modalities": "",  # placeholder - we use modality_list instead
                        "modality_list": [
                            m.value.upper() for m in record.targeted_modalities
                        ],
                        "modality_colors": {
                            m.value.upper(): MODALITY_COLORS.get(
                                m.value.upper(), "grey-8"
                            )
                            for m in record.targeted_modalities
                        },
                        "structure": record.document_structure_format.value.replace(
                            "_", " "
                        ).title(),
                        "description": record.description,
                        "is_private": record.is_private_dataset,
                    }
                )

            table.rows = rows
            table.update()
            count_label.set_text(f"{len(rows)} datasets matched")

        table = (
            ui.table(
                columns=[
                    {
                        "name": "name",
                        "label": "Dataset",
                        "field": "name",
                        "sortable": True,
                        "align": "left",
                    },
                    {"name": "domain", "label": "Domain", "field": "domain"},
                    {
                        "name": "hops",
                        "label": "Hops",
                        "field": "hops",
                        "sortable": True,
                    },
                    {
                        "name": "scope",
                        "label": "Answer Scope",
                        "field": "scope",
                        "sortable": True,
                    },
                    {
                        "name": "context_dep",
                        "label": "Context Dep.",
                        "field": "context_dep",
                        "sortable": True,
                    },
                    {
                        "name": "modalities",
                        "label": "Modalities",
                        "field": "modalities",
                    },
                    {
                        "name": "structure",
                        "label": "Structure",
                        "field": "structure",
                        "sortable": True,
                    },
                    {
                        "name": "description",
                        "label": "Description",
                        "field": "description",
                    },
                ],
                rows=[],
                row_key="name",
                pagination=15,
            )
            .classes("shadow-2xl rounded-xl cursor-pointer")
            .props("dense flat bordered")
        )

        table.add_slot(
            "body",
            """
            <q-tr :props="props" @click="$parent.$emit('row-click', props.row)" class="hover:bg-primary/10">
                <q-td key="name" :props="props">
                    <div class="flex items-center gap-2">
                        <strong>{{ props.row.name }}</strong>
                                                </q-icon>
                                        <q-icon
                    v-if="props.row.is_private"
                    name="lock"
                    size="xs"
                    color="grey-6"
                    class="cursor-default"
                >
                    <q-tooltip>This dataset is private (restricted access)</q-tooltip>
                </q-icon>
                        <q-icon
                            name="content_copy"
                            size="xs"
                            class="text-grey-6 cursor-pointer hover:text-primary"
                            @click.stop="$parent.$emit('copy-name', props.row.name)"
                        >
                            <q-tooltip>Copy dataset name to clipboard</q-tooltip>

                    </div>
                </q-td>
                <q-td key="domain" :props="props">
                    <q-badge :color="props.row.domain_color" text-color="white" class="px-3 py-1">
                        {{ props.row.domain }}
                    </q-badge>
                </q-td>
                <q-td key="hops" :props="props">{{ props.row.hops }}</q-td>
                <q-td key="scope" :props="props">{{ props.row.scope }}</q-td>
                <q-td key="context_dep" :props="props">{{ props.row.context_dep }}</q-td>
                <q-td key="modalities" :props="props">
                    <div class="flex items-center gap-2 flex-nowrap overflow-x-auto py-1">
                        <q-badge v-for="mod in props.row.modality_list" :key="mod"
                                 :color="props.row.modality_colors[mod]" text-color="white"
                                 class="px-3 py-1 text-xs flex-shrink-0">
                            {{ mod }}
                        </q-badge>
                    </div>
                </q-td>
                <q-td key="structure" :props="props">{{ props.row.structure }}</q-td>
                <q-td key="description" :props="props">
                    <div class="ellipsis" style="max-width: 300px;">
                        <q-tooltip max-width="600px">{{ props.row.description }}</q-tooltip>
                        {{ props.row.description }}
                    </div>
                </q-td>
            </q-tr>
        """,
        )

        # Listen for copy event and copy to clipboard
        table.on(
            "copy-name",
            lambda e: ui.run_javascript(f"""
            navigator.clipboard.writeText("{e.args}").then(() => {{
                Quasar.Notify.create({{
                    message: 'Copied "{e.args}" to clipboard!',
                    type: 'positive',
                    position: 'top',
                    timeout: 2000
                }});
            }});
        """),
        )

        # ───── Listen for the custom row-click event ─────
        table.on("row-click", lambda e: show_detail_dialog(e.args))

        add_badges(table)

        # Detail dialog with copy icon
        with ui.dialog() as detail_dialog:
            with ui.card().classes("w-full max-w-3xl"):
                with ui.card_section():
                    # Dataset name with copy icon
                    with ui.row().classes("items-center gap-3 mb-8"):
                        detail_name = ui.label().classes("text-3xl font-bold")
                        detail_private_icon = (
                            ui.icon("lock")
                            .classes("text-grey-6 text-2xl")
                            .tooltip("This dataset is private (restricted access)")
                            .style("display: none;")
                        )  # Hidden by default
                        ui.icon("content_copy").classes(
                            "text-grey-6 cursor-pointer hover:text-primary text-2xl"
                        ).on(
                            "click", lambda: copy_dataset_name(detail_name.text)
                        ).tooltip(
                            "Copy dataset name to clipboard"
                        )

                    with ui.grid(columns=2).classes("gap-x-12 gap-y-6 w-full"):
                        with ui.column().classes("gap-4"):
                            ui.label("Domain").classes("font-medium text-grey-6")
                            detail_domain = ui.html("", sanitize=False)

                            ui.label("Retrieval Hops").classes(
                                "font-medium text-grey-6"
                            )
                            detail_hops = ui.label()

                            ui.label("Answer Scope").classes("font-medium text-grey-6")
                            detail_scope = ui.label()

                            ui.label("Context Dependency").classes(
                                "font-medium text-grey-6"
                            )
                            detail_context = ui.label()

                            ui.label("Document Structure").classes(
                                "font-medium text-grey-6"
                            )
                            detail_structure = ui.label()
                            ui.label("Corpus Size").classes("font-medium text-grey-6")
                            corpus_size = ui.label()
                            ui.label("Benchmark Sizes (train, test)").classes(
                                "font-medium text-grey-6"
                            )
                            with ui.row().classes("gap-8"):
                                with ui.column():
                                    with ui.row():
                                        benchmark_train_size = ui.label()
                                        benchmark_test_size = ui.label()

                            ui.label("Url").classes("font-medium text-grey-6 mt-6")
                            detail_url = ui.html("", sanitize=False)

                        with ui.column().classes("gap-4"):
                            ui.label("Target Modalities").classes(
                                "font-medium text-grey-6"
                            )
                            detail_modalities = ui.html("", sanitize=False)

                            ui.label("Description").classes(
                                "font-medium text-grey-6 mt-4"
                            )
                            detail_description = ui.label().classes(
                                "text-grey-5 leading-relaxed max-w-lg"
                            )

                with ui.card_actions().classes("justify-end"):
                    ui.button("Close", on_click=detail_dialog.close).props(
                        "flat color=primary"
                    )

        # ───── Updated show_detail_dialog ─────
        def show_detail_dialog(row):
            if not row:
                return
            # Find the original record
            record = next(
                r for r in dataset_registry.list() if str(r.name.value) == row["name"]
            )

            detail_name.set_text(str(record.name.value))

            domain_val = record.domain.value
            detail_domain.set_content(f"""
                <q-badge color="{DOMAIN_COLORS.get(domain_val, 'grey-7')}" text-color="white" class="px-4 py-2 text-base">
                    {domain_val.replace('_', ' ').title()}
                </q-badge>
            """)

            detail_hops.set_text(record.retrieval_hops.value.replace("_", " ").title())
            detail_scope.set_text(record.answer_scope.value.replace("_", " ").title())
            detail_context.set_text(
                record.question_context_dependency.value.replace("_", " ").title()
            )
            detail_structure.set_text(
                record.document_structure_format.value.replace("_", " ").title()
            )
            corpus_size.set_text(f"{record.corpus_size:,}")
            benchmark_train_size.set_text(f"{record.train_size:,}")
            benchmark_test_size.set_text(f"{record.test_size:,}")
            if record.url:
                detail_url.set_content(f"""
                                <a href="{record.url}" target="_blank" class="text-primary hover:underline">
                                    {record.url}
                                    <q-icon name="open_in_new" size="sm" class="ml-1"/>
                                </a>
                            """)
            else:
                detail_url.set_content('<span class="text-grey-6">Not available</span>')

            # Modalities
            mods_html = '<div class="flex flex-wrap gap-3 mt-2">'
            for mod in record.targeted_modalities:
                color = MODALITY_COLORS.get(mod.value.upper(), "grey-8")
                mods_html += f"""
                    <q-badge color="{color}" text-color="white" class="px-4 py-2 text-base">
                        {mod.value.upper()}
                    </q-badge>
                """
            mods_html += "</div>"
            detail_modalities.set_content(mods_html)

            detail_description.set_text(record.description)
            if record.is_private_dataset:
                detail_private_icon.style("display: inline-flex;")
            else:
                detail_private_icon.style("display: none;")
            detail_dialog.open()

        # search_input.on("update:model-value", update_table)
        # Bind all filters + search + initial load
        for control in [
            domain_select,
            hops_select,
            scope_select,
            context_select,
            structure_select,
            modality_select,
            search_input,
            # include_private_datasets_check_box,
        ]:
            control.on("update:model-value", update_table)

        def reset_filters():
            domain_select.value = []
            hops_select.value = []
            scope_select.value = []
            context_select.value = []
            structure_select.value = []
            modality_select.value = []
            search_input.value = ""
            include_private_datasets_check_box.value = False  # type: ignore[attr-defined]
            update_table()

        # Initial load
        update_table()


ui.run(title="Dataset Registry Explorer", port=8080, reload=True, favicon="📊")
