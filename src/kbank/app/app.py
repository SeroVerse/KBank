from pathlib import Path
from importlib.metadata import metadata, version
from email.utils import parseaddr

from shiny import ui, App, Inputs, Outputs, Session
from shinyswatch import theme

from kbank.app.models import Tab
from kbank.app.glycan import GlycanTab


# Constants ------------------------------------------------------------------------------------------------------------
_NAME = "KBank"
_DIST = _NAME.lower()
_METADATA = metadata(_DIST)
__version__ = version(_DIST)
__author_name, __author_email = parseaddr(_METADATA.get("Author-email"))

# Extract GitHub/Repository URL
_urls = dict(u.split(", ", 1) for u in _METADATA.get_all("Project-URL") or [])
__github_url = _urls.get("Repository", _urls.get("Source", _urls.get("GitHub")))


# UI -------------------------------------------------------------------------------------------------------------------
class HomeTab(Tab):
    icon = "house"

    @classmethod
    def _render_ui(cls):
        return ui.markdown(Path('README.md').read_text())


ui = ui.page_navbar(
    HomeTab.panel(),
    GlycanTab.panel(),
    ui.nav_spacer(),  # Pushes everything after this to the right side of the navbar
    ui.nav_control(ui.input_dark_mode(id="dark_mode")),
    theme=theme.cosmo,
    title=_NAME,
    id="main_nav",
    window_title=_NAME,
    footer=ui.TagList(
        ui.div(
            ui.HTML(f"""
                <div style="text-align: center; font-size: 0.85rem; color: #64748B; border-top: 1px solid #1E293B; padding-top: 15px; padding-bottom: 15px; margin-top: auto;">
                    <strong>{_DIST}</strong> v{__version__} &bull; 
                    Built by <a href="mailto:{__author_email}" style="color: #0EA5E9; text-decoration: none;"><i class="bi bi-envelope-fill"></i> {__author_name}</a> &bull; 
                    <a href="{__github_url}" trait="_blank" style="color: #0EA5E9; text-decoration: none;"><i class="bi bi-github"></i> GitHub</a> &bull; 
                    <a href="https://pypi.org/project/{_DIST}/" trait="_blank" style="color: #0EA5E9; text-decoration: none;"><i class="bi bi-box-seam-fill"></i> PyPI</a>
                </div>
            """),
            class_="container-fluid d-flex flex-column justify-content-end",
        ),
    ),
    header=ui.tags.head(ui.tags.script(src="https://code.jquery.com/ui/1.13.2/jquery-ui.min.js")),
)


# Server ---------------------------------------------------------------------------------------------------------------
def server(input: Inputs, output: Outputs, session: Session):
    GlycanTab.server()


# App ------------------------------------------------------------------------------------------------------------------
app = App(ui=ui, server=server)
