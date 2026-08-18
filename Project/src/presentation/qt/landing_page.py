"""The screen the app opens on: a landing page in front of the workspace.

Launching straight into the experiment form asks the user to already know
what the app does and which of its parts they want. This page answers that
first — it names the modules, says what each one is for, and lets the user
pick one — and only then hands over to the workspace.

It is a *page*, not a separate window. ``MainWindow`` stacks it against the
workspace in a ``QStackedWidget``, so switching between the two costs
nothing, keeps one title bar and one theme, and leaves the status bar (and
therefore a running experiment's progress) visible throughout.

Three things move on this page, and each is drawn rather than animated
through a ``QGraphicsEffect`` — a widget may hold only one effect, and the
cards need their paint pipeline for hover *and* intro at the same time:

- ``_Backdrop`` — slow drifting colour fields behind everything. Rendered
  into a quarter-scale pixmap and scaled up, which both costs a quarter of
  the fill and gives the soft blur for free.
- ``_AdaptationView`` — two point clouds converging. This is the app's own
  subject matter (the icon carries the same motif), so the one animation
  that runs continuously is also the one that explains what the app is for.
- ``NavigationCard`` — a fade-and-rise on first show, staggered across the
  grid, and a hover state that warms the card towards its accent.

Every animation stops when the page is hidden (see ``hideEvent``), so
nothing repaints behind the workspace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import (
    QEasingCurve,
    QElapsedTimer,
    QPointF,
    QPropertyAnimation,
    QRectF,
    Qt,
    QTimer,
    pyqtProperty,
    pyqtSignal,
)
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient
from PyQt5.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

#: Frame interval for the two continuous animations. 33ms (~30fps) is smooth
#: for slow drifting motion; the backdrop's quarter-scale rendering is what
#: keeps that affordable, not a lower frame rate.
_FRAME_MS = 33


@dataclass(frozen=True)
class Module:
    """One navigable part of the app, as advertised on the landing page.

    *key* is what :attr:`LandingPage.navigate` emits — ``MainWindow`` owns
    the mapping from key to action, so this module never imports it.
    """

    key: str
    glyph: str
    title: str
    description: str
    accent: str  # a key into the palette


#: Order matters: this is reading order in the grid, and it runs from "the
#: thing you came here to do" to "the things that support it".
MODULES: tuple[Module, ...] = (
    Module(
        "experiment",
        "▶",
        "New Experiment",
        "Adapt one source domain to one target and evaluate the result.",
        "metric_teal",
    ),
    Module(
        "batch",
        "▦",
        "Batch Experiments",
        "Run every ordered pair across a set of datasets in one go.",
        "metric_purple",
    ),
    Module(
        "comparison",
        "⇄",
        "Method Comparison",
        "Rank methods side by side on the runs you have already recorded.",
        "metric_amber",
    ),
    Module(
        "history",
        "↺",
        "Experiment History",
        "Revisit saved runs, reopen their results, or reuse their settings.",
        "metric_pink",
    ),
    Module(
        "registry",
        "▤",
        "Dataset Registry",
        "Every dataset you have loaded, with its shape and where it came from.",
        "accent",
    ),
    Module(
        "learning",
        "◈",
        "Learning Materials",
        "How the methods, classifiers, and metrics in this app actually work.",
        "success",
    ),
)


def _mix(first: QColor, second: QColor, amount: float) -> QColor:
    """Blend *first* towards *second* by *amount* in 0-1."""
    amount = max(0.0, min(1.0, amount))
    return QColor(
        round(first.red() + (second.red() - first.red()) * amount),
        round(first.green() + (second.green() - first.green()) * amount),
        round(first.blue() + (second.blue() - first.blue()) * amount),
    )


def _with_alpha(color: QColor, alpha: int) -> QColor:
    faded = QColor(color)
    faded.setAlpha(max(0, min(255, alpha)))
    return faded


class NavigationCard(QFrame):
    """A clickable tile for one :class:`Module`, painted end to end.

    Everything — icon tile, title, description, hover glow — is drawn in
    :meth:`paintEvent` rather than composed from child labels. That is what
    makes the intro's rise possible: a translate on the painter moves the
    whole card, whereas child widgets would stay pinned wherever the grid
    layout put them.
    """

    activated = pyqtSignal(str)

    CARD_HEIGHT = 126

    def __init__(self, module: Module, palette: dict[str, str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._module = module
        self._palette = palette
        self._hover = 0.0
        self._intro = 0.0
        self._available = True
        self._note = ""

        self.setFixedHeight(self.CARD_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_Hover, True)
        self.setToolTip(module.description)

        self._hover_animation = QPropertyAnimation(self, b"hoverProgress", self)
        self._hover_animation.setDuration(170)
        self._hover_animation.setEasingCurve(QEasingCurve.OutCubic)

        self._intro_animation = QPropertyAnimation(self, b"introProgress", self)
        self._intro_animation.setDuration(480)
        self._intro_animation.setEasingCurve(QEasingCurve.OutCubic)

        self._intro_timer = QTimer(self)
        self._intro_timer.setSingleShot(True)
        self._intro_timer.timeout.connect(self._begin_intro)

    # -- animated properties -------------------------------------------
    def _get_hover(self) -> float:
        return self._hover

    def _set_hover(self, value: float) -> None:
        self._hover = float(value)
        self.update()

    hoverProgress = pyqtProperty(float, _get_hover, _set_hover)

    def _get_intro(self) -> float:
        return self._intro

    def _set_intro(self, value: float) -> None:
        self._intro = float(value)
        self.update()

    introProgress = pyqtProperty(float, _get_intro, _set_intro)

    # -- state ----------------------------------------------------------
    def set_palette(self, palette: dict[str, str]) -> None:
        self._palette = palette
        self.update()

    def set_available(self, available: bool, note: str = "") -> None:
        """Mark the module reachable or not.

        An unavailable card stays clickable on purpose. "Sign in to see your
        history" is a step the user can take, and a card that swallows the
        click teaches nothing — ``MainWindow`` answers with the fix instead.
        """
        self._available = available
        self._note = note
        self.setToolTip(note if note else self._module.description)
        self.update()

    def play_intro(self, delay_ms: int) -> None:
        self._set_intro(0.0)
        # A timer parented to the card rather than QTimer.singleShot: if the
        # card is destroyed while its turn in the stagger is still pending,
        # the timer dies with it instead of firing at a deleted animation.
        self._intro_timer.start(delay_ms)

    def _begin_intro(self) -> None:
        self._intro_animation.stop()
        self._intro_animation.setStartValue(0.0)
        self._intro_animation.setEndValue(1.0)
        self._intro_animation.start()

    def settle(self) -> None:
        """Skip the intro and sit at the finished state."""
        self._intro_timer.stop()
        self._intro_animation.stop()
        self._set_intro(1.0)

    # -- interaction ------------------------------------------------------
    def _animate_hover(self, target: float) -> None:
        self._hover_animation.stop()
        self._hover_animation.setStartValue(self._hover)
        self._hover_animation.setEndValue(target)
        self._hover_animation.start()

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._animate_hover(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if not self.hasFocus():
            self._animate_hover(0.0)
        super().leaveEvent(event)

    def focusInEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._animate_hover(1.0)
        super().focusInEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if not self.underMouse():
            self._animate_hover(0.0)
        super().focusOutEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.LeftButton and self.rect().contains(event.pos()):
            self.activated.emit(self._module.key)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.activated.emit(self._module.key)
            event.accept()
            return
        super().keyPressEvent(event)

    # -- painting ---------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        intro = max(0.0, min(1.0, self._intro))
        painter.setOpacity(intro)
        # Rises the last 16px into place; combined with the stagger in
        # LandingPage this reads as the grid assembling itself.
        painter.translate(0.0, (1.0 - intro) * 16.0)

        hover = self._hover if self._available else 0.0
        accent = QColor(self._palette[self._module.accent])
        surface = QColor(self._palette["surface"])
        body = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)

        if hover > 0.01:
            # A soft ring outside the border, standing in for a drop shadow
            # (which would need a QGraphicsEffect and cost us the intro).
            glow = QPainterPath()
            glow.addRoundedRect(body.adjusted(-2.0, -2.0, 2.0, 2.0), 14.0, 14.0)
            painter.setPen(QPen(_with_alpha(accent, int(70 * hover)), 3.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(glow)

        background = _mix(surface, QColor(self._palette["accent_soft"]), 0.7 * hover)
        border = _mix(QColor(self._palette["border_strong"]), accent, hover)

        path = QPainterPath()
        path.addRoundedRect(body, 12.0, 12.0)
        painter.fillPath(path, background)
        pen = QPen(border, 1.0)
        if not self._available:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

        self._paint_icon(painter, accent, hover)
        self._paint_text(painter, hover)
        self._paint_arrow(painter, accent, hover)
        painter.end()

    def _paint_icon(self, painter: QPainter, accent: QColor, hover: float) -> None:
        tile = QRectF(18.0, 20.0, 42.0, 42.0)
        tile_path = QPainterPath()
        tile_path.addRoundedRect(tile, 11.0, 11.0)
        strength = 0.16 + 0.18 * hover if self._available else 0.10
        painter.fillPath(tile_path, _with_alpha(accent, int(255 * strength)))

        font = QFont(self.font())
        font.setPixelSize(19)
        painter.setFont(font)
        painter.setPen(accent if self._available else QColor(self._palette["text_muted"]))
        painter.drawText(tile, Qt.AlignCenter, self._module.glyph)

    def _paint_text(self, painter: QPainter, hover: float) -> None:
        left = 74.0
        width = self.width() - left - 20.0

        title_font = QFont(self.font())
        title_font.setPixelSize(15)
        title_font.setWeight(QFont.DemiBold)
        painter.setFont(title_font)
        painter.setPen(
            QColor(self._palette["text"])
            if self._available
            else QColor(self._palette["text_secondary"])
        )
        painter.drawText(QRectF(left, 22.0, width, 22.0), Qt.AlignVCenter | Qt.AlignLeft, self._module.title)

        body_font = QFont(self.font())
        body_font.setPixelSize(12)
        painter.setFont(body_font)
        if self._available:
            painter.setPen(QColor(self._palette["text_secondary"]))
            text = self._module.description
        else:
            painter.setPen(QColor(self._palette["text_muted"]))
            text = self._note or self._module.description
        painter.drawText(
            QRectF(left, 48.0, width, self.CARD_HEIGHT - 48.0 - 16.0),
            Qt.AlignTop | Qt.AlignLeft | Qt.TextWordWrap,
            text,
        )

    def _paint_arrow(self, painter: QPainter, accent: QColor, hover: float) -> None:
        if hover <= 0.01 or not self._available:
            return
        font = QFont(self.font())
        font.setPixelSize(15)
        painter.setFont(font)
        painter.setPen(_with_alpha(accent, int(255 * hover)))
        # Slides in from the left as it fades, so the eye follows it outward.
        offset = (1.0 - hover) * 6.0
        painter.drawText(
            QRectF(self.width() - 40.0 - offset, self.CARD_HEIGHT - 34.0, 24.0, 20.0),
            Qt.AlignCenter,
            "→",
        )


class _AdaptationView(QFrame):
    """Two point clouds pulled into a shared space, on a loop.

    Not decoration: this is the operation the whole app performs. The
    caption tracks the phase, so a first-time user watching it for four
    seconds has seen what "domain adaptation" means before reading a word
    of the documentation.
    """

    #: One full pass: separated, converging, held together, reset.
    CYCLE_SECONDS = 8.0
    POINTS_PER_DOMAIN = 44

    def __init__(self, palette: dict[str, str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._phase = 0.0
        self.setMinimumSize(320, 210)
        # Capped so a tall window grows the whitespace around the hero
        # rather than stretching the point clouds into a thin band.
        self.setMaximumHeight(320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Fixed seed: the shape of the clouds is part of the design, not
        # something that should differ between launches.
        rng = np.random.default_rng(7)
        self._source_start = self._cloud(rng, 0.30, 0.64, 0.105)
        self._target_start = self._cloud(rng, 0.73, 0.33, 0.095)
        # Both domains move: adaptation projects the two into a common
        # latent space rather than dragging one onto the other.
        self._source_end = self._cloud(rng, 0.47, 0.52, 0.145)
        self._target_end = self._cloud(rng, 0.53, 0.48, 0.145)

    @staticmethod
    def _cloud(rng: np.random.Generator, x: float, y: float, spread: float) -> np.ndarray:
        points = rng.normal(loc=(x, y), scale=spread, size=(_AdaptationView.POINTS_PER_DOMAIN, 2))
        return np.clip(points, 0.06, 0.94)

    def set_palette(self, palette: dict[str, str]) -> None:
        self._palette = palette
        self.update()

    def set_phase(self, phase: float) -> None:
        self._phase = phase % 1.0
        self.update()

    def _convergence(self) -> tuple[float, float]:
        """``(mix, opacity)`` for the current phase.

        The cycle deliberately does not run backwards at the end — watching
        adapted domains pull back apart would show something the app never
        does. It fades out and starts over instead.
        """
        phase = self._phase
        if phase < 0.42:  # converging
            eased = QEasingCurve(QEasingCurve.InOutCubic).valueForProgress(phase / 0.42)
            return eased, 1.0
        if phase < 0.78:  # held together
            return 1.0, 1.0
        if phase < 0.88:  # fading out
            return 1.0, 1.0 - (phase - 0.78) / 0.10
        return 0.0, (phase - 0.88) / 0.12  # fading back in, separated again

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        frame = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(frame, 14.0, 14.0)
        painter.fillPath(path, QColor(self._palette["surface"]))
        painter.setPen(QPen(QColor(self._palette["border_strong"]), 1.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

        mix, opacity = self._convergence()
        plot = frame.adjusted(16.0, 42.0, -16.0, -38.0)

        source_color = QColor(self._palette["metric_teal"])
        target_color = QColor(self._palette["metric_purple"])
        painter.setPen(Qt.NoPen)
        painter.setOpacity(max(0.0, min(1.0, opacity)))
        self._paint_cloud(painter, plot, self._source_start, self._source_end, mix, source_color)
        self._paint_cloud(painter, plot, self._target_start, self._target_end, mix, target_color)
        painter.setOpacity(1.0)

        self._paint_legend(painter, frame, source_color, target_color)
        self._paint_caption(painter, frame, mix, opacity)
        painter.end()

    def _paint_cloud(
        self,
        painter: QPainter,
        plot: QRectF,
        start: np.ndarray,
        end: np.ndarray,
        mix: float,
        color: QColor,
    ) -> None:
        positions = start + (end - start) * mix
        painter.setBrush(_with_alpha(color, 205))
        for x, y in positions:
            centre = QPointF(plot.left() + x * plot.width(), plot.top() + y * plot.height())
            painter.drawEllipse(centre, 4.0, 4.0)

    def _paint_legend(
        self, painter: QPainter, frame: QRectF, source: QColor, target: QColor
    ) -> None:
        font = QFont(self.font())
        font.setPixelSize(11)
        font.setWeight(QFont.DemiBold)
        painter.setFont(font)

        x = frame.left() + 18.0
        y = frame.top() + 22.0
        for color, label in ((source, "Source"), (target, "Target")):
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(x + 4.0, y), 4.0, 4.0)
            painter.setPen(QColor(self._palette["text_secondary"]))
            painter.drawText(QRectF(x + 14.0, y - 9.0, 70.0, 18.0), Qt.AlignVCenter | Qt.AlignLeft, label)
            x += 76.0

    def _paint_caption(self, painter: QPainter, frame: QRectF, mix: float, opacity: float) -> None:
        if mix < 0.12:
            caption = "Before adaptation — domains apart"
        elif mix > 0.88:
            caption = "After adaptation — domains aligned"
        else:
            caption = "Adapting — projecting to a shared space"

        font = QFont(self.font())
        font.setPixelSize(11)
        painter.setFont(font)
        painter.setOpacity(max(0.35, min(1.0, opacity)))
        painter.setPen(QColor(self._palette["text_muted"]))
        painter.drawText(
            QRectF(frame.left(), frame.bottom() - 30.0, frame.width(), 22.0),
            Qt.AlignCenter,
            caption,
        )
        painter.setOpacity(1.0)


class LandingPage(QWidget):
    """The app's front door: hero, module grid, and session actions.

    Emits :attr:`navigate` with a module key (see :data:`MODULES`) or one of
    the session keys ``"auth"``, ``"theme"``, ``"exit"``. It holds no
    reference to ``MainWindow`` and knows nothing about what any key does,
    so the routing table stays in one place on the window side.
    """

    navigate = pyqtSignal(str)

    #: Drifting colour fields: (x, y, radius, palette key, x sway, y sway,
    #: angular speed, phase offset). Slow and low-contrast on purpose — this
    #: sits behind text and must never compete with it.
    _BLOBS: tuple[tuple[float, float, float, str, float, float, float, float], ...] = (
        (0.16, 0.20, 0.46, "accent", 0.05, 0.04, 0.09, 0.0),
        (0.84, 0.28, 0.40, "metric_purple", 0.06, 0.05, 0.07, 1.9),
        (0.62, 0.88, 0.48, "metric_teal", 0.05, 0.04, 0.06, 3.4),
    )

    def __init__(self, palette: dict[str, str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("landingPage")
        self._palette = palette
        self._cards: dict[str, NavigationCard] = {}
        self._intro_played = False

        self._clock = QElapsedTimer()
        self._clock.start()
        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_MS)
        self._timer.timeout.connect(self._advance)

        self._build()
        self._apply_palette()

    # -- construction -----------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(38, 26, 38, 22)
        root.setSpacing(16)

        # All spare height goes to the hero, which centres its own column
        # inside whatever it gets — so a tall window grows the artwork
        # instead of stranding the footer below an empty band. Note there is
        # deliberately no trailing addStretch(): an expanding spacer item
        # takes precedence over an expanding *widget* in QBoxLayout's
        # distribution, so one here would swallow the space regardless of
        # the stretch factors and pin the hero at its minimum.
        root.addWidget(self._build_hero(), 1)
        root.addWidget(self._build_section_label())
        root.addLayout(self._build_card_grid())
        root.addWidget(self._build_footer())

    def _build_hero(self) -> QWidget:
        hero = QWidget()
        # Explicitly Expanding: a QVBoxLayout hands spare height only to
        # items that carry the expand flag, so with the trailing stretch in
        # the mix a merely-Preferred hero would never grow and the stretch
        # factors below it would have no effect at all.
        hero.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        layout = QHBoxLayout(hero)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(28)

        # One opacity effect on the column, so the whole block fades as a
        # unit rather than each label arriving separately.
        self._hero_column = QWidget()
        column = QVBoxLayout(self._hero_column)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)
        column.addStretch(1)

        self._eyebrow = QLabel("TRANSFER LEARNING STUDIO")
        self._eyebrow.setObjectName("landingEyebrow")
        column.addWidget(self._eyebrow)

        self._title = QLabel("Adapta")
        self._title.setObjectName("landingTitle")
        column.addWidget(self._title)

        self._tagline = QLabel(
            "Train on the data you have, evaluate on the data you care about."
        )
        self._tagline.setObjectName("landingTagline")
        self._tagline.setWordWrap(True)
        column.addWidget(self._tagline)

        self._blurb = QLabel(
            "Six adaptation methods, five classifiers, and a record of every "
            "run — with the alignment diagnostics to tell you whether the "
            "adaptation actually helped."
        )
        self._blurb.setObjectName("landingBlurb")
        self._blurb.setWordWrap(True)
        column.addWidget(self._blurb)

        column.addSpacing(6)
        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        self._start_button = QPushButton("Start an Experiment  →")
        self._start_button.setObjectName("landingPrimaryButton")
        self._start_button.setCursor(Qt.PointingHandCursor)
        self._start_button.clicked.connect(lambda: self.navigate.emit("experiment"))
        buttons.addWidget(self._start_button)

        self._guide_button = QPushButton("Learning Materials")
        self._guide_button.setObjectName("landingGhostButton")
        self._guide_button.setCursor(Qt.PointingHandCursor)
        self._guide_button.clicked.connect(lambda: self.navigate.emit("learning"))
        buttons.addWidget(self._guide_button)
        buttons.addStretch(1)
        column.addLayout(buttons)
        column.addStretch(1)

        self._hero_opacity = QGraphicsOpacityEffect(self._hero_column)
        self._hero_opacity.setOpacity(1.0)
        self._hero_column.setGraphicsEffect(self._hero_opacity)

        self._adaptation = _AdaptationView(self._palette)

        layout.addWidget(self._hero_column, 3)
        layout.addWidget(self._adaptation, 2)
        return hero

    def _build_section_label(self) -> QWidget:
        self._section_label = QLabel("MODULES")
        self._section_label.setObjectName("landingSection")
        return self._section_label

    def _build_card_grid(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)
        for index, module in enumerate(MODULES):
            card = NavigationCard(module, self._palette)
            card.activated.connect(self.navigate.emit)
            self._cards[module.key] = card
            grid.addWidget(card, index // 3, index % 3)
        for column in range(3):
            grid.setColumnStretch(column, 1)
        return grid

    def _build_footer(self) -> QWidget:
        footer = QFrame()
        footer.setObjectName("landingFooter")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(10)

        self._status_label = QLabel()
        self._status_label.setObjectName("landingStatus")
        layout.addWidget(self._status_label)
        layout.addStretch(1)

        self._auth_button = QPushButton("Log In")
        self._auth_button.setObjectName("linkButton")
        self._auth_button.setCursor(Qt.PointingHandCursor)
        self._auth_button.clicked.connect(lambda: self.navigate.emit("auth"))
        layout.addWidget(self._auth_button)

        self._theme_button = QPushButton("☀  Light")
        self._theme_button.setObjectName("themeToggleButton")
        self._theme_button.setCursor(Qt.PointingHandCursor)
        self._theme_button.setToolTip("Switch between dark and light mode")
        self._theme_button.clicked.connect(lambda: self.navigate.emit("theme"))
        layout.addWidget(self._theme_button)

        self._exit_button = QPushButton("⏻  Exit System")
        self._exit_button.setObjectName("exitButton")
        self._exit_button.setCursor(Qt.PointingHandCursor)
        self._exit_button.setToolTip("Close Adapta and quit the application (Ctrl+Q)")
        self._exit_button.clicked.connect(lambda: self.navigate.emit("exit"))
        layout.addWidget(self._exit_button)
        return footer

    # -- external state ---------------------------------------------------
    def set_palette(self, palette: dict[str, str]) -> None:
        self._palette = palette
        for card in self._cards.values():
            card.set_palette(palette)
        self._adaptation.set_palette(palette)
        self._apply_palette()
        self.update()

    def _apply_palette(self) -> None:
        self._status_label.setStyleSheet(f"color: {self._palette['text_muted']};")

    def set_theme_label(self, text: str) -> None:
        self._theme_button.setText(text)

    def set_session(self, *, has_database: bool, signed_in: bool, user_label: str) -> None:
        """Reflect the current database/auth state on the page.

        Cards whose module needs something the user does not have yet are
        marked unavailable *with the reason* — they still emit ``navigate``
        so the window can offer the missing step.
        """
        self._auth_button.setText("Log Out" if signed_in else "Log In")
        self._status_label.setText(user_label)

        self._cards["registry"].set_available(
            has_database, "Needs a database connection — unavailable right now."
        )
        self._cards["comparison"].set_available(
            has_database, "Needs a database connection — unavailable right now."
        )
        if not has_database:
            self._cards["history"].set_available(
                False, "Needs a database connection — unavailable right now."
            )
        else:
            self._cards["history"].set_available(
                signed_in, "Sign in to see the experiments you have saved."
            )

    # -- animation --------------------------------------------------------
    def play_intro(self) -> None:
        """Fade the hero in and stagger the cards up beneath it."""
        self._hero_opacity.setOpacity(0.0)
        self._hero_animation = QPropertyAnimation(self._hero_opacity, b"opacity", self)
        self._hero_animation.setDuration(520)
        self._hero_animation.setStartValue(0.0)
        self._hero_animation.setEndValue(1.0)
        self._hero_animation.setEasingCurve(QEasingCurve.OutCubic)
        self._hero_animation.start()

        for index, module in enumerate(MODULES):
            self._cards[module.key].play_intro(160 + index * 55)

    def _advance(self) -> None:
        self._adaptation.set_phase(
            (self._clock.elapsed() / 1000.0) / _AdaptationView.CYCLE_SECONDS
        )
        self.update()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().showEvent(event)
        if not self._intro_played:
            self._intro_played = True
            self.play_intro()
        else:
            # Coming back from the workspace: the cards are already known,
            # so replaying the reveal would just delay the user.
            for card in self._cards.values():
                card.settle()
        self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._timer.stop()
        super().hideEvent(event)

    # -- backdrop ---------------------------------------------------------
    def _backdrop(self, width: int, height: int) -> QPixmap:
        """Render the drifting fields at quarter scale.

        Scaling a small pixmap up is both cheaper than filling full-size
        radial gradients every frame and softer, since the smoothing blurs
        the gradient banding away for free.
        """
        small_width = max(1, width // 4)
        small_height = max(1, height // 4)
        pixmap = QPixmap(small_width, small_height)
        pixmap.fill(QColor(self._palette["page"]))

        seconds = self._clock.elapsed() / 1000.0
        # Lighter themes need a fainter wash to keep text contrast.
        alpha = 46 if QColor(self._palette["page"]).lightness() < 128 else 30

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        for x, y, radius, key, sway_x, sway_y, speed, offset in self._BLOBS:
            centre = QPointF(
                (x + sway_x * math.sin(seconds * speed + offset)) * small_width,
                (y + sway_y * math.cos(seconds * speed * 1.3 + offset)) * small_height,
            )
            extent = radius * max(small_width, small_height)
            gradient = QRadialGradient(centre, extent)
            colour = QColor(self._palette[key])
            gradient.setColorAt(0.0, _with_alpha(colour, alpha))
            gradient.setColorAt(1.0, _with_alpha(colour, 0))
            painter.setBrush(gradient)
            painter.drawEllipse(centre, extent, extent)
        painter.end()
        return pixmap

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(self.rect(), self._backdrop(self.width(), self.height()))
        painter.end()


__all__ = ["LandingPage", "Module", "MODULES", "NavigationCard"]
