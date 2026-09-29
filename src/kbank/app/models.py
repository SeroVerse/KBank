from shiny import module, Inputs, Outputs, Session
from shiny import ui as shiny_ui
from kbank.app.icons import icon_lucide


class Tab:
    icon: str | None = None

    @classmethod
    def title(cls) -> str:
        if cls.__doc__:
            summary = cls.__doc__.strip().splitlines()[0].strip().rstrip(".")
            if summary:
                return summary
        return cls.__name__.removesuffix("Tab")

    @classmethod
    def ui(cls):
        @module.ui
        def _ui():
            return cls._render_ui()

        return _ui(cls.__name__)

    @classmethod
    def server(cls, *args, **kwargs):
        @module.server
        def _server(input: Inputs, output: Outputs, session: Session):
            return cls._register_server(input, output, session, *args, **kwargs)

        return _server(cls.__name__)

    @classmethod
    def panel(cls) -> shiny_ui._navs.NavPanel:
        icon_tag = icon_lucide(cls.icon) if cls.icon else None
        return shiny_ui.nav_panel(cls.title(), cls.ui(), icon=icon_tag, value=cls.__name__)

    @classmethod
    def _render_ui(cls):
        raise NotImplementedError

    @classmethod
    def _register_server(cls, input: Inputs, output: Outputs, session: Session, *args, **kwargs):
        pass


# TODO: Implement an easy sub-Tab mechanism, or "hierarchical tabs"