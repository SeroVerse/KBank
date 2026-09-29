from pathlib import Path
import polars as pl
from shiny import ui, render, Inputs, Outputs, Session

from kbank.glycan.sugars import Sugars

from kbank.app.icons import icon_lucide
from kbank.app.models import Tab


class GlycanTab(Tab):
    """Glycans"""
    icon = "candy"
    _GLYCAN_PATH = Path('data').joinpath('verified_antigen_glycans').with_suffix('.parquet')
    @classmethod
    def _render_ui(cls):
        return ui.layout_sidebar(
            ui.sidebar(),
            ui.navset_card_tab(
                ui.nav_panel(
                    "Dataframe",
                    ui.output_data_frame("glycan_df"),
                    value="tab_glycan_df",
                    icon=icon_lucide("table"),
                ),
                ui.nav_panel(
                    "Sugars",
                    ui.output_data_frame("sugar_df"),
                    value="tab_sugar_df",
                    icon=icon_lucide("candy-cane"),
                ),
            ),
        )

    @classmethod
    def _register_server(cls, input: Inputs, output: Outputs, session: Session):
        @render.data_frame
        def glycan_df():
            return render.DataTable(pl.read_parquet(cls._GLYCAN_PATH))

        @render.data_frame
        def sugar_df():
            return render.DataTable(pl.DataFrame({
                i: getattr(Sugars, i) for i in Sugars._SOA_ARRAYS
            }))