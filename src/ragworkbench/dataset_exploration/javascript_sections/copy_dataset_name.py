# Copyright 2024 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
