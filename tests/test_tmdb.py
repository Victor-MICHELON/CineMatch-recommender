import pytest
import requests
import responses

from cinematch.tmdb import API_URL, MediaCache, TMDBClient, TMDBError, pick_trailer


def url(movie_id: int) -> str:
    return API_URL.format(id=movie_id)


@pytest.fixture
def client():
    return TMDBClient("key", timeout=1, retries=0)


@responses.activate
def test_media_success(client):
    responses.get(
        url(603),
        json={
            "poster_path": "/m.jpg",
            "videos": {"results": [{"site": "YouTube", "type": "Trailer", "key": "abc", "official": True}]},
        },
    )
    m = client.media(603)
    assert m.poster_url == "https://image.tmdb.org/t/p/w500/m.jpg"
    assert m.trailer_url == "https://www.youtube.com/watch?v=abc"
    assert m.page_url == "https://www.themoviedb.org/movie/603"


@responses.activate
def test_media_without_poster_or_trailer(client):
    responses.get(url(1), json={"poster_path": None})
    m = client.media(1)
    assert m.poster_url is None and m.trailer_url is None


@responses.activate
def test_404_raises(client):
    responses.get(url(2), status=404)
    with pytest.raises(TMDBError):
        client.media(2)


@responses.activate
def test_timeout_raises(client):
    responses.get(url(3), body=requests.exceptions.ConnectTimeout())
    with pytest.raises(TMDBError):
        client.media(3)


def test_empty_key_rejected():
    with pytest.raises(ValueError):
        TMDBClient("")


def test_pick_trailer_prefers_official_youtube():
    videos = [
        {"site": "Vimeo", "type": "Trailer", "key": "v"},
        {"site": "YouTube", "type": "Teaser", "key": "t"},
        {"site": "YouTube", "type": "Trailer", "key": "unofficial", "official": False},
        {"site": "YouTube", "type": "Trailer", "key": "official", "official": True},
    ]
    assert pick_trailer(videos) == "https://www.youtube.com/watch?v=official"
    assert pick_trailer([]) is None


def test_cache_without_key_returns_none():
    assert MediaCache(None).get_many([1, 2]) == {1: None, 2: None}


@responses.activate
def test_cache_keeps_successes_and_retries_failures(client):
    responses.get(url(10), json={"poster_path": "/a.jpg"})
    responses.get(url(11), status=500)
    cache = MediaCache(client)
    first = cache.get_many([10, 11])
    assert first[10] is not None and first[11] is None
    calls = len(responses.calls)
    second = cache.get_many([10, 11])
    assert second[10] == first[10]
    assert len(responses.calls) == calls + 1  # 10 servi par le cache, 11 retenté
