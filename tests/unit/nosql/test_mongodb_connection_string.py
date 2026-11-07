"""The MongoDB URI, which the driver either accepts or refuses outright.

The repository's own compose file starts a server with authentication off, so
the unconfigured case has to produce a URI with no userinfo at all: pymongo
rejects `mongodb://:@host` outright, and spelling the credentials as empty
strings used to make every local run fail before a single query. Each case is
checked by parsing the URI back with the driver's own parser rather than by
string comparison, so the assertion is about what pymongo will read.
"""

from typing import Any

import pytest
from pymongo.errors import InvalidURI
from pymongo.uri_parser import parse_uri

from base.clients.nosql.mongodb import MongoDBClient
from base.services.configuration.mixins import MongoSettingsMixin


def build(make_settings: Any, **values: Any) -> str:
    """The connection string for a settings object carrying `values`."""
    make_settings(MongoSettingsMixin, **values)
    return MongoDBClient._build_connection_string()


def test_a_server_without_credentials_gets_a_uri_with_no_userinfo(
    make_settings: Any,
) -> None:
    uri = build(make_settings, mongodb_host="localhost:27017")
    assert uri == "mongodb://localhost:27017/"
    assert "@" not in uri


def test_the_driver_accepts_the_uri_built_for_an_unauthenticated_server(
    make_settings: Any,
) -> None:
    parsed = parse_uri(build(make_settings, mongodb_host="localhost:27017"))
    assert parsed["username"] is None
    assert parsed["password"] is None
    assert parsed["nodelist"] == [("localhost", 27017)]


def test_a_username_and_password_round_trip_through_the_uri(
    make_settings: Any,
) -> None:
    parsed = parse_uri(
        build(
            make_settings,
            mongodb_host="db:27017",
            mongodb_username="appuser",
            mongodb_password="s3cret",
        )
    )
    assert parsed["username"] == "appuser"
    assert parsed["password"] == "s3cret"


def test_a_username_alone_produces_an_empty_password_the_driver_accepts(
    make_settings: Any,
) -> None:
    parsed = parse_uri(
        build(make_settings, mongodb_host="db:27017", mongodb_username="appuser")
    )
    assert parsed["username"] == "appuser"
    assert parsed["password"] == ""


@pytest.mark.parametrize(
    "password",
    [
        "p@ssw0rd",
        "with/slash",
        "with:colon",
        "with?question",
        "100%sure",
        "with#hash",
        "mixed@:/?#%",
    ],
)
def test_a_password_needing_escaping_survives_the_round_trip(
    make_settings: Any, password: str
) -> None:
    uri = build(
        make_settings,
        mongodb_host="db:27017",
        mongodb_username="appuser",
        mongodb_password=password,
    )
    assert parse_uri(uri)["password"] == password


def test_a_username_needing_escaping_survives_the_round_trip(
    make_settings: Any,
) -> None:
    uri = build(
        make_settings,
        mongodb_host="db:27017",
        mongodb_username="user@tenant",
        mongodb_password="pw",
    )
    assert parse_uri(uri)["username"] == "user@tenant"


def test_a_password_without_a_username_builds_a_uri_the_driver_refuses(
    make_settings: Any,
) -> None:
    """Pins current behaviour: a password alone is not a usable configuration."""
    uri = build(make_settings, mongodb_host="db:27017", mongodb_password="pw")
    assert uri.startswith("mongodb://:")
    with pytest.raises(InvalidURI, match="empty string is not valid username"):
        parse_uri(uri)


def test_custom_params_are_appended_after_the_host(make_settings: Any) -> None:
    uri = build(
        make_settings,
        mongodb_host="db:27017",
        mongodb_params="?replicaSet=rs0&w=majority",
    )
    assert uri == "mongodb://db:27017/?replicaSet=rs0&w=majority"
    assert parse_uri(uri)["options"]["replicaSet"] == "rs0"


def test_custom_params_survive_alongside_credentials(make_settings: Any) -> None:
    parsed = parse_uri(
        build(
            make_settings,
            mongodb_host="db:27017",
            mongodb_username="appuser",
            mongodb_password="s3cret",
            mongodb_params="?authSource=admin",
        )
    )
    assert parsed["username"] == "appuser"
    assert parsed["options"]["authSource"] == "admin"


def test_the_base_path_chooses_the_scheme(make_settings: Any) -> None:
    uri = build(
        make_settings,
        mongodb_base_path="mongodb+srv",
        mongodb_host="cluster.mongodb.net",
    )
    assert uri.startswith("mongodb+srv://")


def test_the_database_name_is_not_part_of_the_connection_string(
    make_settings: Any,
) -> None:
    """The database is chosen per handle, so the URI never names one."""
    uri = build(
        make_settings, mongodb_host="db:27017", mongodb_database="application_db"
    )
    assert "application_db" not in uri
    assert parse_uri(uri)["database"] is None


def test_several_hosts_become_a_node_list(make_settings: Any) -> None:
    parsed = parse_uri(build(make_settings, mongodb_host="a:27017,b:27018"))
    assert parsed["nodelist"] == [("a", 27017), ("b", 27018)]
