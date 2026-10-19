"""Client TMDB minimal pour l'UI : affiche, bande-annonce et lien vers la fiche.

Délai d'attente court et nouvelles tentatives sur les erreurs transitoires
(429, 5xx). Le cache est laissé à l'appelant (`st.cache_data` dans l'app) : ce
module ne dépend pas de Streamlit.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "https://api.themoviedb.org/3/movie/{id}"
POSTER_URL = "https://image.tmdb.org/t/p/w500{path}"
PAGE_URL = "https://www.themoviedb.org/movie/{id}"
YOUTUBE_URL = "https://www.youtube.com/watch?v={key}"


class TMDBError(RuntimeError):
    """Échec d'appel TMDB (réseau, clé invalide, film inconnu…)."""


@dataclass(frozen=True)
class MovieMedia:
    poster_url: str | None
    trailer_url: str | None
    page_url: str


def page_url(movie_id: int) -> str:
    return PAGE_URL.format(id=movie_id)


class TMDBClient:
    def __init__(self, api_key: str, timeout: float = 3.0, retries: int = 2) -> None:
        if not api_key:
            raise ValueError("clé TMDB vide")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        retry = Retry(
            total=retries,
            backoff_factor=0.3,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def media(self, movie_id: int, language: str = "en-US") -> MovieMedia:
        """Affiche + bande-annonce YouTube officielle (si disponible). Lève `TMDBError`."""
        try:
            resp = self.session.get(
                API_URL.format(id=movie_id),
                params={"api_key": self.api_key, "append_to_response": "videos", "language": language},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise TMDBError(f"TMDB {movie_id} : {exc}") from exc

        poster = data.get("poster_path")
        return MovieMedia(
            poster_url=POSTER_URL.format(path=poster) if poster else None,
            trailer_url=pick_trailer(data.get("videos", {}).get("results", [])),
            page_url=page_url(movie_id),
        )


class MediaCache:
    """Cache mémoire des médias TMDB, rempli en parallèle.

    Seuls les succès sont mémorisés : une panne réseau ponctuelle n'est pas figée.
    """

    def __init__(self, client: TMDBClient | None, workers: int = 8) -> None:
        self.client = client
        self.workers = workers
        self._cache: dict[tuple[int, str], MovieMedia] = {}
        self._lock = threading.Lock()

    def get_many(self, movie_ids: list[int], language: str = "en-US") -> dict[int, MovieMedia | None]:
        """Médias de chaque film (None si pas de clé ou échec TMDB)."""
        if self.client is None:
            return dict.fromkeys(movie_ids)
        client = self.client
        missing = [i for i in dict.fromkeys(movie_ids) if (i, language) not in self._cache]

        def fetch(movie_id: int) -> None:
            try:
                media = client.media(movie_id, language)
            except TMDBError:
                return
            with self._lock:
                self._cache[(movie_id, language)] = media

        if missing:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(missing))) as pool:
                list(pool.map(fetch, missing))
        return {i: self._cache.get((i, language)) for i in movie_ids}


def pick_trailer(videos: list[dict[str, object]]) -> str | None:
    """Bande-annonce YouTube, en privilégiant les vidéos officielles."""
    trailers = [v for v in videos if v.get("site") == "YouTube" and v.get("type") == "Trailer" and v.get("key")]
    if not trailers:
        return None
    best = max(trailers, key=lambda v: bool(v.get("official")))
    return YOUTUBE_URL.format(key=best["key"])
