"""Resolves the analyzer implementation at startup.

If ``MODEL_PATH`` points at a valid inference bundle the concrete
:class:`~app.analysis.vyra_analyzer.VyraAnalyzer` is loaded once here. If the
path is unset the API runs without analysis (``/health`` reports it, POSTs get
501). If the path is set but broken we raise: a deployment that expects a model
should fail loudly, not silently serve a degraded API.
"""

from __future__ import annotations

import logging

from app.analysis.contract import QualityAnalyzer
from app.core.config import Settings

logger = logging.getLogger(__name__)


def load_analyzer(settings: Settings) -> QualityAnalyzer | None:
    """Load the trained model bundle, or ``None`` if analysis is unavailable."""
    if settings.model_path is None:
        logger.warning("MODEL_PATH is not set - image analysis is unavailable.")
        return None

    from app.analysis.vyra_analyzer import VyraAnalyzer

    try:
        return VyraAnalyzer.from_path(settings.model_path)
    except Exception:
        if settings.require_analyzer:
            logger.exception("MODEL_PATH=%s could not be loaded.", settings.model_path)
            raise
        logger.exception(
            "MODEL_PATH=%s could not be loaded; continuing without an analyzer "
            "(set REQUIRE_ANALYZER=true to make this fatal).",
            settings.model_path,
        )
        return None
