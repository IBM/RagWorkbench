from nicegui import ui


def copy_dataset_name(name: str):
    ui.run_javascript(f"""
        navigator.clipboard.writeText("{name}").then(() => {{
            Quasar.Notify.create({{
                message: 'Copied "{name}" to clipboard!',
                type: 'positive',
                position: 'top',
                timeout: 2000
            }});
        }});
    """)
